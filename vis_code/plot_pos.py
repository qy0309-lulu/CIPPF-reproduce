from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
CSV_COLUMNS = [
    "train_fraction",
    "test_row",
    "source_time_index",
    "true_x",
    "true_y",
    "pred_x",
    "pred_y",
    "error_x",
    "error_y",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot true, SSPPF, and CIPPF x/y position curves from result CSV files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Experiment result directory or parent directory containing SSPPF/CIPPF result folders.",
    )
    parser.add_argument(
        "--ssppf-csv",
        type=Path,
        default=None,
        help="Optional explicit SSPPF predictions.csv path.",
    )
    parser.add_argument(
        "--cippf-csv",
        type=Path,
        default=None,
        help="Optional explicit CIPPF predictions.csv path.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        help="Train fraction to plot, for example 0.4. Defaults to all common fractions.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start row in the matched test sequence.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End row in the matched test sequence. Defaults to the full sequence.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output figures. Defaults to <results-dir>/pos_plots.",
    )
    return parser.parse_args()


def read_prediction_csv(csv_path):
    csv_path = Path(csv_path)
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for raw_row in reader:
            if not raw_row:
                continue
            if len(raw_row) == len(CSV_COLUMNS):
                values = raw_row
                columns = CSV_COLUMNS
            elif len(raw_row) == len(header):
                values = raw_row
                columns = header
            else:
                raise ValueError(
                    "Unexpected column count in {}: header has {}, row has {}".format(
                        csv_path, len(header), len(raw_row)
                    )
                )
            rows.append({column: float(value) for column, value in zip(columns, values)})

    if not rows:
        raise ValueError("No prediction rows found in {}".format(csv_path))

    return rows


def group_by_train_fraction(rows):
    grouped = {}
    for row in rows:
        train_fraction = round(float(row.get("train_fraction", 0.0)), 6)
        grouped.setdefault(train_fraction, []).append(row)

    for fraction_rows in grouped.values():
        fraction_rows.sort(key=lambda row: (row["source_time_index"], row["test_row"]))
    return grouped


def infer_model_name(csv_path):
    text = str(csv_path).lower()
    if "ssppf" in text:
        return "SSPPF"
    if "cippf" in text:
        return "CIPPF"
    return None


def parse_result_dir_name(result_dir):
    parts = result_dir.name.split("_", 2)
    if len(parts) != 3:
        return None

    method, train_fraction, experiment_time = parts
    method = method.upper()
    if method not in ("SSPPF", "CIPPF"):
        return None

    try:
        train_fraction = round(float(train_fraction), 6)
    except ValueError:
        return None

    return {
        "method": method,
        "train_fraction": train_fraction,
        "experiment_time": experiment_time,
    }


def find_prediction_csvs(results_dir):
    results_dir = Path(results_dir)
    if results_dir.is_file():
        return [results_dir]
    if (results_dir / "predictions.csv").exists():
        return [results_dir / "predictions.csv"]
    return sorted(results_dir.rglob("predictions.csv"))


def discover_prediction_csvs(results_dir):
    discovered = {"SSPPF": {}, "CIPPF": {}}
    for csv_path in find_prediction_csvs(results_dir):
        result_info = parse_result_dir_name(csv_path.parent)
        if result_info is None:
            model_name = infer_model_name(csv_path)
            if model_name is None:
                continue
            rows_by_fraction = group_by_train_fraction(read_prediction_csv(csv_path))
            experiment_time = csv_path.parent.name
            candidates = [
                {
                    "method": model_name,
                    "train_fraction": fraction,
                    "experiment_time": experiment_time,
                }
                for fraction in rows_by_fraction
            ]
        else:
            candidates = [result_info]

        for candidate in candidates:
            method = candidate["method"]
            fraction = candidate["train_fraction"]
            current = discovered[method].get(fraction)
            if current is None or candidate["experiment_time"] > current["experiment_time"]:
                discovered[method][fraction] = {
                    "csv_path": csv_path,
                    "experiment_time": candidate["experiment_time"],
                }

    return discovered


def select_prediction_csvs(results_dir, ssppf_csv=None, cippf_csv=None):
    if ssppf_csv is not None or cippf_csv is not None:
        if ssppf_csv is None or cippf_csv is None:
            raise ValueError("--ssppf-csv and --cippf-csv must be provided together.")
        return {
            "SSPPF": {"explicit": {"csv_path": Path(ssppf_csv), "experiment_time": "explicit"}},
            "CIPPF": {"explicit": {"csv_path": Path(cippf_csv), "experiment_time": "explicit"}},
        }

    discovered = discover_prediction_csvs(results_dir)
    missing = [method for method, items in discovered.items() if not items]
    if missing:
        raise FileNotFoundError(
            "Could not find {} result folders named like "
            "{{method}}_{{train_fraction}}_{{experiment_time}} under {}.".format(
                " and ".join(missing), results_dir
            )
        )
    return discovered


def align_rows(ssppf_rows, cippf_rows):
    ssppf_by_time = {int(row["source_time_index"]): row for row in ssppf_rows}
    cippf_by_time = {int(row["source_time_index"]): row for row in cippf_rows}
    common_times = sorted(set(ssppf_by_time) & set(cippf_by_time))
    if not common_times:
        raise ValueError("SSPPF and CIPPF CSV files have no common source_time_index values.")

    true_x = np.array([ssppf_by_time[t]["true_x"] for t in common_times], dtype=float)
    true_y = np.array([ssppf_by_time[t]["true_y"] for t in common_times], dtype=float)
    ssppf_x = np.array([ssppf_by_time[t]["pred_x"] for t in common_times], dtype=float)
    ssppf_y = np.array([ssppf_by_time[t]["pred_y"] for t in common_times], dtype=float)
    cippf_x = np.array([cippf_by_time[t]["pred_x"] for t in common_times], dtype=float)
    cippf_y = np.array([cippf_by_time[t]["pred_y"] for t in common_times], dtype=float)
    return np.array(common_times), true_x, true_y, ssppf_x, ssppf_y, cippf_x, cippf_y


def plot_axis(time_index, true_values, ssppf_values, cippf_values, axis_name, output_path):
    plt.figure(figsize=(12, 4.8))
    plt.plot(time_index, true_values, color="#ff7f0e", linewidth=1.2, label="True")
    plt.plot(time_index, ssppf_values, color="#1f77b4", linewidth=0.8, label="SSPPF")
    plt.plot(time_index, cippf_values, color="#2ca02c", linewidth=0.8, label="CIPPF")
    plt.xlabel("Source time index")
    plt.ylabel("{} position".format(axis_name.upper()))
    plt.title("{} Position Decoding".format(axis_name.upper()))
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    model_csvs = select_prediction_csvs(args.results_dir, args.ssppf_csv, args.cippf_csv)
    if "explicit" in model_csvs["SSPPF"] and "explicit" in model_csvs["CIPPF"]:
        ssppf_by_fraction = group_by_train_fraction(
            read_prediction_csv(model_csvs["SSPPF"]["explicit"]["csv_path"])
        )
        cippf_by_fraction = group_by_train_fraction(
            read_prediction_csv(model_csvs["CIPPF"]["explicit"]["csv_path"])
        )
        common_fractions = sorted(set(ssppf_by_fraction) & set(cippf_by_fraction))
        csv_pairs = {
            fraction: (
                model_csvs["SSPPF"]["explicit"]["csv_path"],
                model_csvs["CIPPF"]["explicit"]["csv_path"],
                ssppf_by_fraction[fraction],
                cippf_by_fraction[fraction],
            )
            for fraction in common_fractions
        }
    else:
        common_fractions = sorted(set(model_csvs["SSPPF"]) & set(model_csvs["CIPPF"]))
        csv_pairs = {}
        for fraction in common_fractions:
            ssppf_csv = model_csvs["SSPPF"][fraction]["csv_path"]
            cippf_csv = model_csvs["CIPPF"][fraction]["csv_path"]
            ssppf_by_fraction = group_by_train_fraction(read_prediction_csv(ssppf_csv))
            cippf_by_fraction = group_by_train_fraction(read_prediction_csv(cippf_csv))
            if fraction in ssppf_by_fraction and fraction in cippf_by_fraction:
                csv_pairs[fraction] = (
                    ssppf_csv,
                    cippf_csv,
                    ssppf_by_fraction[fraction],
                    cippf_by_fraction[fraction],
                )
        common_fractions = sorted(csv_pairs)

    if args.train_fraction is not None:
        target_fraction = round(args.train_fraction, 6)
        csv_pairs = {
            fraction: pair for fraction, pair in csv_pairs.items() if fraction == target_fraction
        }
        common_fractions = sorted(csv_pairs)
    if not common_fractions:
        raise ValueError("No common train fractions found between SSPPF and CIPPF results.")

    output_dir = args.output_dir
    if output_dir is None:
        base_dir = args.results_dir if args.results_dir.is_dir() else args.results_dir.parent
        output_dir = base_dir / "pos_plots"

    for fraction in common_fractions:
        ssppf_csv, cippf_csv, ssppf_rows, cippf_rows = csv_pairs[fraction]
        (
            time_index,
            true_x,
            true_y,
            ssppf_x,
            ssppf_y,
            cippf_x,
            cippf_y,
        ) = align_rows(ssppf_rows, cippf_rows)

        start = max(args.start, 0)
        end = args.end if args.end is not None else len(time_index)
        time_index = time_index[start:start+2000]
        true_x = true_x[start:start+2000]
        true_y = true_y[start:start+2000]
        ssppf_x = ssppf_x[start:start+2000]
        ssppf_y = ssppf_y[start:start+2000]
        cippf_x = cippf_x[start:start+2000]
        cippf_y = cippf_y[start:start+2000]

        split_name = "train_{:.1f}".format(fraction)
        plot_axis(
            time_index,
            true_x,
            ssppf_x,
            cippf_x,
            "x",
            output_dir / "{}_x_position.png".format(split_name),
        )
        plot_axis(
            time_index,
            true_y,
            ssppf_y,
            cippf_y,
            "y",
            output_dir / "{}_y_position.png".format(split_name),
        )
        print(
            "Saved plots for train_fraction={} to {} using {} and {}".format(
                fraction, output_dir, ssppf_csv, cippf_csv
            )
        )


if __name__ == "__main__":
    main()
