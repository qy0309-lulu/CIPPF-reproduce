# First-order state-space point-process filter for hippocampal position decoding.
#
# This script follows the data loading, splitting, evaluation, and plotting
# pattern used in LSTM/hc.py, but replaces the LSTM decoder with a Gaussian
# state-space filter whose observation model is a first-order non-homogeneous
# Poisson GLM.

from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "Neural_Decoding"))

from metrics import get_R2, get_rho
from preprocessing_funcs import get_spikes_with_history


def get_mse_by_axis(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2, axis=0)


def get_nrmse_by_range(y_true, y_pred):
    rmse = np.sqrt(get_mse_by_axis(y_true, y_pred))
    coord_range = np.ptp(y_true, axis=0)
    coord_range[coord_range == 0] = np.nan
    return rmse / coord_range


def get_nmse_by_variance(y_true, y_pred):
    mse = get_mse_by_axis(y_true, y_pred)
    variance = np.var(y_true, axis=0)
    variance[variance == 0] = np.nan
    return mse / variance


class GaussianPointProcessFilter:
    """Gaussian state-space point-process filter.

    State model:
        z_t = A [z_{t-1}, 1]^T + q_t,       q_t ~ N(0, Q)

    Observation model for each neuron j:
        n_{t,j} ~ Poisson(lambda_j(z_t))
        log(lambda_j(z_t)) = beta_j^T phi(z_t)

    The GLM uses a first-order spatial feature map [1, x, y], making the
    conditional intensity a non-homogeneous Poisson process over position.
    """

    def __init__(
        self,
        encoding_model="linear",
        glm_l2=1e-4,
        process_noise_scale=1.0,
        initial_cov_scale=1.0,
        max_newton_iter=8,
        tol=1e-5,
        verbose=1,
    ):
        if encoding_model != "linear":
            raise ValueError("SSPPF_OneOrder only supports encoding_model='linear'")
        self.encoding_model = encoding_model
        self.glm_l2 = glm_l2
        self.process_noise_scale = process_noise_scale
        self.initial_cov_scale = initial_cov_scale
        self.max_newton_iter = max_newton_iter
        self.tol = tol
        self.verbose = verbose

    def _features(self, z):
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            x, y = z
            return np.array([1.0, x, y])

        x = z[:, 0]
        y = z[:, 1]
        return np.column_stack((np.ones(z.shape[0]), x, y))

    def _feature_derivatives(self, z):
        grad_phi = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hess_phi = np.zeros((3, 2, 2))
        return grad_phi, hess_phi

    def _fit_one_glm(self, Phi, spikes):
        mean_rate = np.mean(spikes) + 1e-6
        beta0 = np.zeros(Phi.shape[1])
        beta0[0] = np.log(mean_rate)
        penalty_mask = np.ones(Phi.shape[1])
        penalty_mask[0] = 0.0

        def objective(beta):
            eta = np.clip(Phi @ beta, -20.0, 20.0)
            lam = np.exp(eta)
            nll = np.sum(lam - spikes * eta)
            nll += 0.5 * self.glm_l2 * np.sum((penalty_mask * beta) ** 2)
            grad = Phi.T @ (lam - spikes) + self.glm_l2 * penalty_mask * beta
            return nll, grad

        result = minimize(
            fun=lambda b: objective(b)[0],
            x0=beta0,
            jac=lambda b: objective(b)[1],
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-8},
        )
        if not result.success and self.verbose:
            print("WARNING: GLM fit did not fully converge:", result.message)
        return result.x

    def fit(self, spike_counts_train, y_train):
        self.y_scale_ = np.nanstd(y_train, axis=0)
        self.y_scale_[self.y_scale_ < 1e-8] = 1.0
        z_train = y_train / self.y_scale_
        margin = 0.25 * np.ptp(z_train, axis=0)
        self.state_min_ = np.min(z_train, axis=0) - margin
        self.state_max_ = np.max(z_train, axis=0) + margin

        Phi = self._features(z_train)
        betas = []
        for neuron_idx in range(spike_counts_train.shape[1]):
            if self.verbose and neuron_idx % 10 == 0:
                print("Fitting Poisson GLM neuron {}/{}".format(neuron_idx + 1, spike_counts_train.shape[1]))
            betas.append(self._fit_one_glm(Phi, spike_counts_train[:, neuron_idx]))
        self.betas_ = np.vstack(betas)

        z1 = z_train[:-1]
        z2 = z_train[1:]
        X_aug = np.column_stack((z1, np.ones(z1.shape[0])))
        self.A_aug_ = np.linalg.lstsq(X_aug, z2, rcond=None)[0].T
        residuals = z2 - X_aug @ self.A_aug_.T
        self.Q_ = np.cov(residuals.T) * self.process_noise_scale
        self.Q_ = self._make_pos_def(self.Q_)
        self.initial_cov_ = np.cov(z_train.T) * self.initial_cov_scale
        self.initial_cov_ = self._make_pos_def(self.initial_cov_)
        return self

    @staticmethod
    def _make_pos_def(matrix, jitter=1e-6):
        matrix = np.asarray(matrix, dtype=float)
        matrix = 0.5 * (matrix + matrix.T)
        min_eig = np.min(np.linalg.eigvalsh(matrix))
        if min_eig < jitter:
            matrix = matrix + np.eye(matrix.shape[0]) * (jitter - min_eig)
        return matrix

    def _predict_state(self, mean, cov):
        aug = np.r_[mean, 1.0]
        pred_mean = self.A_aug_ @ aug
        A = self.A_aug_[:, :2]
        pred_cov = A @ cov @ A.T + self.Q_
        return pred_mean, self._make_pos_def(pred_cov)

    def _posterior_mode(self, prior_mean, prior_cov, spikes):
        prior_cov = self._make_pos_def(prior_cov)
        c, lower = cho_factor(prior_cov, lower=True, check_finite=False)
        z = prior_mean.copy()

        for _ in range(self.max_newton_iter):
            phi = self._features(z)
            grad_phi, hess_phi = self._feature_derivatives(z)
            eta = np.clip(self.betas_ @ phi, -20.0, 20.0)
            lam = np.exp(eta)
            diff = z - prior_mean

            grad = cho_solve((c, lower), diff, check_finite=False)
            hess = cho_solve((c, lower), np.eye(2), check_finite=False)

            grad_eta = self.betas_ @ grad_phi
            hess_eta = np.einsum("nf,fij->nij", self.betas_, hess_phi)

            grad += np.sum((lam - spikes)[:, None] * grad_eta, axis=0)
            hess += np.einsum("n,ni,nj->ij", lam, grad_eta, grad_eta)
            hess += np.sum((lam - spikes)[:, None, None] * hess_eta, axis=0)
            hess = self._make_pos_def(hess)

            step = np.linalg.solve(hess, grad)
            step_scale = 1.0
            while step_scale > 1e-3:
                candidate = z - step_scale * step
                if np.all(np.isfinite(candidate)):
                    break
                step_scale *= 0.5
            z_next = np.clip(z - step_scale * step, self.state_min_, self.state_max_)
            if np.linalg.norm(z_next - z) < self.tol:
                z = z_next
                break
            z = z_next

        post_cov = np.linalg.inv(hess)
        return z, self._make_pos_def(post_cov)

    def predict(self, spike_counts, y_initial):
        num_steps = spike_counts.shape[0]
        z_predicted = np.empty((num_steps, 2))
        mean = y_initial / self.y_scale_
        cov = self.initial_cov_.copy()

        for t in range(num_steps):
            if t > 0:
                mean, cov = self._predict_state(mean, cov)
            mean, cov = self._posterior_mode(mean, cov, spike_counts[t])
            z_predicted[t] = mean

        return z_predicted * self.y_scale_


def load_hc_data():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "example_data_hc.pickle",
        Path("/root/example_data_hc.pickle"),
    ]
    data_path = next((path for path in candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("Could not find example_data_hc.pickle")

    with data_path.open("rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def main():
    verbose = 1
    neural_data, pos_binned = load_hc_data()

    bins_before = 4
    bins_current = 1
    bins_after = 5

    nd_sum = np.nansum(neural_data, axis=0)
    rmv_nrn = np.where(nd_sum < 100)
    neural_data = np.delete(neural_data, rmv_nrn, 1)

    X = get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)
    X_flat = X.reshape(X.shape[0], (X.shape[1] * X.shape[2]))
    y = pos_binned

    rmv_time = np.where(np.isnan(y[:, 0]) | np.isnan(y[:, 1]))
    X = np.delete(X, rmv_time, 0)
    X_flat = np.delete(X_flat, rmv_time, 0)
    spike_counts = np.delete(neural_data, rmv_time, 0)
    y = np.delete(y, rmv_time, 0)

    training_range = [0, 0.4]
    testing_range = [0.4, 1.0]

    num_examples = X.shape[0]
    training_set = np.arange(
        int(np.round(training_range[0] * num_examples)) + bins_before,
        int(np.round(training_range[1] * num_examples)) - bins_after,
    )
    testing_set = np.arange(
        int(np.round(testing_range[0] * num_examples)) + bins_before,
        int(np.round(testing_range[1] * num_examples)) - bins_after,
    )
    X_train = X[training_set, :, :]
    X_flat_train = X_flat[training_set, :]
    y_train = y[training_set, :]

    X_test = X[testing_set, :, :]
    X_flat_test = X_flat[testing_set, :]
    y_test = y[testing_set, :]

    spike_counts_train = spike_counts[training_set, :]
    spike_counts_test = spike_counts[testing_set, :]

    X_train_mean = np.nanmean(X_train, axis=0)
    X_train_std = np.nanstd(X_train, axis=0)
    X_train_std[X_train_std == 0] = 1
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    X_flat_train_mean = np.nanmean(X_flat_train, axis=0)
    X_flat_train_std = np.nanstd(X_flat_train, axis=0)
    X_flat_train_std[X_flat_train_std == 0] = 1
    X_flat_train = (X_flat_train - X_flat_train_mean) / X_flat_train_std
    X_flat_test = (X_flat_test - X_flat_train_mean) / X_flat_train_std

    y_train_mean = np.mean(y_train, axis=0)
    y_train = y_train - y_train_mean
    y_test = y_test - y_train_mean

    print("Starting first-order SSPPF training")
    model = GaussianPointProcessFilter(
        encoding_model="linear",
        glm_l2=1e-4,
        process_noise_scale=1.0,
        initial_cov_scale=1.0,
        max_newton_iter=8,
        verbose=verbose,
    )
    model.fit(spike_counts_train, y_train)

    print("Decoding test set with first-order SSPPF")
    predictions = model.predict(spike_counts_test, y_test[0])
    R2_test = get_R2(y_test, predictions)
    rho_test = get_rho(y_test, predictions)
    mse_test = get_mse_by_axis(y_test, predictions)
    nrmse_test = get_nrmse_by_range(y_test, predictions)
    nmse_test = get_nmse_by_variance(y_test, predictions)
    print("\n[SSPPF OneOrder] test R2:", R2_test)
    print("[SSPPF OneOrder] test rho:", rho_test)
    print("[SSPPF OneOrder] test MSE [X, Y]:", mse_test)
    print("[SSPPF OneOrder] test NRMSE_range [X, Y]:", nrmse_test)
    print("[SSPPF OneOrder] test NMSE_var [X, Y]:", nmse_test)

    plot_start = min(2000, max(0, y_test.shape[0] - 1))
    plot_end = min(5000, y_test.shape[0])
    if plot_end <= plot_start:
        plot_start = 0
        plot_end = y_test.shape[0]

    fig_x_ssppf = plt.figure()
    plt.plot(y_test[plot_start:plot_end, 0] + y_train_mean[0], "b", label="test")
    plt.plot(predictions[plot_start:plot_end, 0] + y_train_mean[0], "r", label="prediction")
    plt.xlabel("Time bin")
    plt.ylabel("X position")
    plt.legend()
    fig_x_ssppf.savefig("x_position_decoding_ssppf_oneorder_4V1.jpg", dpi=300)

    fig_y_ssppf = plt.figure()
    plt.plot(y_test[plot_start:plot_end, 1] + y_train_mean[1], "b", label="test")
    plt.plot(predictions[plot_start:plot_end, 1] + y_train_mean[1], "r", label="prediction")
    plt.xlabel("Time bin")
    plt.ylabel("Y position")
    plt.legend()
    fig_y_ssppf.savefig("y_position_decoding_ssppf_oneorder_4V1.jpg", dpi=300)


if __name__ == "__main__":
    main()
