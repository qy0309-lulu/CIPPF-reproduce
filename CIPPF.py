# Neural correlation integrated adaptive point-process filter (CIPPF).
#
# This script follows the hippocampal decoding workflow used by the SSPPF
# examples, but adds the CIPPF functional-connectivity term proposed by
# Li et al. (2025). The posterior combines:
#   1. linear Gaussian state prediction,
#   2. independent single-neuron Poisson GLM tuning,
#   3. a Gaussian state density centered at g(lambda_minus), where
#      lambda_minus is a smoothed population spike pattern and g is an ANN.

from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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


class NeuralCorrelationIntegratedPointProcessFilter:
    """CIPPF decoder with Gaussian posterior approximation.

    State model:
        z_k = A [z_{k-1}, 1]^T + q_k, q_k ~ N(0, R)

    Single-neuron tuning:
        Delta N_{k,i} ~ Poisson(lambda_i(z_k))
        log lambda_i(z_k) = beta_i^T phi(z_k)

    Functional-connectivity term:
        p(z_k | lambda_minus_k) = N(z_k; g(lambda_minus_k), Q_corr)

    Marginal state correction from the CIPPF derivation:
        posterior precision includes -S^{-1}, where S is the empirical
        marginal state covariance.
    """

    def __init__(
        self,
        encoding_model="linear",
        glm_l2=1e-4,
        process_noise_scale=1.0,
        connectivity_cov_scale=1.0,
        marginal_cov_scale=1.0,
        initial_cov_scale=1.0,
        half_gaussian_sigma=5.0,
        half_gaussian_width=30,
        ann_hidden_layers=(64, 32),
        ann_alpha=1e-4,
        ann_max_iter=500,
        max_newton_iter=8,
        tol=1e-5,
        random_state=0,
        verbose=1,
    ):
        if encoding_model not in ("linear", "quadratic"):
            raise ValueError("encoding_model must be 'linear' or 'quadratic'")
        self.encoding_model = encoding_model
        self.glm_l2 = glm_l2
        self.process_noise_scale = process_noise_scale
        self.connectivity_cov_scale = connectivity_cov_scale
        self.marginal_cov_scale = marginal_cov_scale
        self.initial_cov_scale = initial_cov_scale
        self.half_gaussian_sigma = half_gaussian_sigma
        self.half_gaussian_width = half_gaussian_width
        self.ann_hidden_layers = ann_hidden_layers
        self.ann_alpha = ann_alpha
        self.ann_max_iter = ann_max_iter
        self.max_newton_iter = max_newton_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def _features(self, z):
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            x, y = z
            if self.encoding_model == "linear":
                return np.array([1.0, x, y])
            return np.array([1.0, x, y, x * x, y * y, x * y])

        x = z[:, 0]
        y = z[:, 1]
        if self.encoding_model == "linear":
            return np.column_stack((np.ones(z.shape[0]), x, y))
        return np.column_stack((np.ones(z.shape[0]), x, y, x * x, y * y, x * y))

    def _feature_derivatives(self, z):
        x, y = z
        if self.encoding_model == "linear":
            grad_phi = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            hess_phi = np.zeros((3, 2, 2))
            return grad_phi, hess_phi

        grad_phi = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0 * x, 0.0],
                [0.0, 2.0 * y],
                [y, x],
            ]
        )
        hess_phi = np.zeros((6, 2, 2))
        hess_phi[3] = np.array([[2.0, 0.0], [0.0, 0.0]])
        hess_phi[4] = np.array([[0.0, 0.0], [0.0, 2.0]])
        hess_phi[5] = np.array([[0.0, 1.0], [1.0, 0.0]])
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

        bounds = [(None, None)] * Phi.shape[1]
        if self.encoding_model == "quadratic":
            bounds[3] = (None, 0.0)
            bounds[4] = (None, 0.0)

        result = minimize(
            fun=lambda b: objective(b)[0],
            x0=beta0,
            jac=lambda b: objective(b)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-8},
        )
        if not result.success and self.verbose:
            print("WARNING: GLM fit did not fully converge:", result.message)
        return result.x

    def _half_gaussian_kernel(self):
        width = int(max(1, self.half_gaussian_width))
        sigma = max(float(self.half_gaussian_sigma), 1e-6)
        offsets = np.arange(width, dtype=float)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        return kernel / np.sum(kernel)

    def _smooth_spike_patterns(self, spike_counts):
        spike_counts = np.asarray(spike_counts, dtype=float)
        kernel = self._half_gaussian_kernel()
        smoothed = np.empty_like(spike_counts, dtype=float)
        for neuron_idx in range(spike_counts.shape[1]):
            smoothed[:, neuron_idx] = np.convolve(
                spike_counts[:, neuron_idx], kernel, mode="full"
            )[: spike_counts.shape[0]]
        return smoothed

    def _fit_connectivity_map(self, spike_counts_train, z_train):
        lambda_minus = self._smooth_spike_patterns(spike_counts_train)
        self.lambda_scaler_ = StandardScaler()
        lambda_scaled = self.lambda_scaler_.fit_transform(lambda_minus)

        self.connectivity_model_ = MLPRegressor(
            hidden_layer_sizes=self.ann_hidden_layers,
            activation="relu",
            solver="adam",
            alpha=self.ann_alpha,
            learning_rate_init=1e-3,
            max_iter=self.ann_max_iter,
            random_state=self.random_state,
            early_stopping=False,
            n_iter_no_change=20,
            verbose=False,
        )
        self.connectivity_model_.fit(lambda_scaled, z_train)
        z_conn = self.connectivity_model_.predict(lambda_scaled)

        residuals = z_train - z_conn
        corr_cov = np.cov(residuals.T) * self.connectivity_cov_scale
        self.Q_corr_ = self._make_pos_def(corr_cov)
        return z_conn

    def fit(self, spike_counts_train, y_train):
        self.y_scale_ = np.nanstd(y_train, axis=0)
        self.y_scale_[self.y_scale_ < 1e-8] = 1.0
        z_train = y_train / self.y_scale_

        margin = 0.25 * np.ptp(z_train, axis=0)
        self.state_min_ = np.min(z_train, axis=0) - margin
        self.state_max_ = np.max(z_train, axis=0) + margin
        self.marginal_mean_ = np.mean(z_train, axis=0)
        self.S_ = self._make_pos_def(np.cov(z_train.T) * self.marginal_cov_scale)

        Phi = self._features(z_train)
        betas = []
        for neuron_idx in range(spike_counts_train.shape[1]):
            if self.verbose and neuron_idx % 10 == 0:
                print("Fitting Poisson GLM neuron {}/{}".format(neuron_idx + 1, spike_counts_train.shape[1]))
            betas.append(self._fit_one_glm(Phi, spike_counts_train[:, neuron_idx]))
        self.betas_ = np.vstack(betas)

        if self.verbose:
            print("Training neural-correlation map g(lambda_minus)")
        self._fit_connectivity_map(spike_counts_train, z_train)

        z1 = z_train[:-1]
        z2 = z_train[1:]
        X_aug = np.column_stack((z1, np.ones(z1.shape[0])))
        self.A_aug_ = np.linalg.lstsq(X_aug, z2, rcond=None)[0].T
        residuals = z2 - X_aug @ self.A_aug_.T
        self.R_ = self._make_pos_def(np.cov(residuals.T) * self.process_noise_scale)
        self.initial_cov_ = self._make_pos_def(np.cov(z_train.T) * self.initial_cov_scale)
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
        pred_cov = A @ cov @ A.T + self.R_
        return pred_mean, self._make_pos_def(pred_cov)

    def _connectivity_means(self, spike_counts):
        lambda_minus = self._smooth_spike_patterns(spike_counts)
        lambda_scaled = self.lambda_scaler_.transform(lambda_minus)
        z_conn = self.connectivity_model_.predict(lambda_scaled)
        return np.clip(z_conn, self.state_min_, self.state_max_)

    def _posterior_mode(self, prior_mean, prior_cov, spikes, corr_mean):
        prior_cov = self._make_pos_def(prior_cov)
        prior_c, prior_lower = cho_factor(prior_cov, lower=True, check_finite=False)
        corr_c, corr_lower = cho_factor(self.Q_corr_, lower=True, check_finite=False)
        marg_c, marg_lower = cho_factor(self.S_, lower=True, check_finite=False)
        z = prior_mean.copy()

        for _ in range(self.max_newton_iter):
            phi = self._features(z)
            grad_phi, hess_phi = self._feature_derivatives(z)
            eta = np.clip(self.betas_ @ phi, -20.0, 20.0)
            lam = np.exp(eta)

            grad = cho_solve((prior_c, prior_lower), z - prior_mean, check_finite=False)
            hess = cho_solve((prior_c, prior_lower), np.eye(2), check_finite=False)

            grad += cho_solve((corr_c, corr_lower), z - corr_mean, check_finite=False)
            hess += cho_solve((corr_c, corr_lower), np.eye(2), check_finite=False)

            grad -= cho_solve((marg_c, marg_lower), z - self.marginal_mean_, check_finite=False)
            hess -= cho_solve((marg_c, marg_lower), np.eye(2), check_finite=False)

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
        corr_means = self._connectivity_means(spike_counts)

        mean = y_initial / self.y_scale_
        cov = self.initial_cov_.copy()
        for t in range(num_steps):
            if t > 0:
                mean, cov = self._predict_state(mean, cov)
            mean, cov = self._posterior_mode(mean, cov, spike_counts[t], corr_means[t])
            z_predicted[t] = mean

        return z_predicted * self.y_scale_


def load_hc_data():
    candidates = [
        SCRIPT_DIR / "example_data_hc.pickle",
        SCRIPT_DIR / "SSPPF_hc2" / "example_data_hc.pickle",
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

    training_range = [0, 0.5]
    testing_range = [0.5, 1.0]

    num_examples = X.shape[0]
    training_set = np.arange(
        int(np.round(training_range[0] * num_examples)) + bins_before,
        int(np.round(training_range[1] * num_examples)) - bins_after,
    )
    testing_set = np.arange(
        int(np.round(testing_range[0] * num_examples)) + bins_before,
        int(np.round(testing_range[1] * num_examples)) - bins_after,
    )
    y_train = y[training_set, :]
    y_test = y[testing_set, :]

    spike_counts_train = spike_counts[training_set, :]
    spike_counts_test = spike_counts[testing_set, :]

    y_train_mean = np.mean(y_train, axis=0)
    y_train = y_train - y_train_mean
    y_test = y_test - y_train_mean

    print("Starting CIPPF training")
    model = NeuralCorrelationIntegratedPointProcessFilter(
        encoding_model="linear",
        glm_l2=1e-4,
        process_noise_scale=1.0,
        connectivity_cov_scale=1.0,
        marginal_cov_scale=1.0,
        initial_cov_scale=1.0,
        half_gaussian_sigma=5.0,
        half_gaussian_width=30,
        ann_hidden_layers=(64, 32),
        ann_alpha=1e-4,
        ann_max_iter=500,
        max_newton_iter=8,
        random_state=0,
        verbose=verbose,
    )
    model.fit(spike_counts_train, y_train)

    print("Decoding test set with CIPPF")
    predictions = model.predict(spike_counts_test, y_test[0])
    R2_test = get_R2(y_test, predictions)
    rho_test = get_rho(y_test, predictions)
    mse_test = get_mse_by_axis(y_test, predictions)
    nrmse_test = get_nrmse_by_range(y_test, predictions)
    nmse_test = get_nmse_by_variance(y_test, predictions)
    print("\n[CIPPF] test R2:", R2_test)
    print("[CIPPF] test rho:", rho_test)
    print("[CIPPF] test MSE [X, Y]:", mse_test)
    print("[CIPPF] test NRMSE_range [X, Y]:", nrmse_test)
    print("[CIPPF] test NMSE_var [X, Y]:", nmse_test)

    plot_start = min(2000, max(0, y_test.shape[0] - 1))
    plot_end = min(3000, y_test.shape[0])
    if plot_end <= plot_start:
        plot_start = 0
        plot_end = y_test.shape[0]

    fig_x_cippf = plt.figure()
    plt.plot(y_test[plot_start:plot_end, 0] + y_train_mean[0], "b", label="ground_truth")
    plt.plot(predictions[plot_start:plot_end, 0] + y_train_mean[0], "r", label="prediction")
    plt.xlabel("Time bin")
    plt.ylabel("X position")
    plt.legend()
    fig_x_cippf.savefig("x_position_decoding_cippf_1V1.jpg", dpi=300)

    fig_y_cippf = plt.figure()
    plt.plot(y_test[plot_start:plot_end, 1] + y_train_mean[1], "b", label="ground_truth")
    plt.plot(predictions[plot_start:plot_end, 1] + y_train_mean[1], "r", label="prediction")
    plt.xlabel("Time bin")
    plt.ylabel("Y position")
    plt.legend()
    fig_y_cippf.savefig("y_position_decoding_cippf_1V1.jpg", dpi=300)


if __name__ == "__main__":
    main()
