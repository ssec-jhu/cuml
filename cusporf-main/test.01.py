#
# test.01.py
#
# Notes:
#  Here we implement a simple test harness for the cuml SPORF implementation.
#

import os
import sys
import argparse
import re
import csv
import itertools
from pidmon import scaffold as p

from cuml import SPORFClassifier as cuRFC
from cuml import SPORFRegressor as cuRFR

from test_data import test_data as td


#
# load test parameters from a specified .CSV file:
#   - the .CSV file must have column headers
#       - the first column must have a header named "tag"
#       - the remaining columns must have headers whose names correspond to SPORF parameters
#
def loadParams() -> (argparse.Namespace, dict, list[dict]):

    # parse the command line:
    #   - the .CSV filename and the tag string are the first two (unnamed positional) arguments
    #   - rows in the .CSV can also be filtered by specifying additional optional arguments as
    #       name=value
    #     where name is a SPORF parameter name (corresponding to a column name in the .CSV file)
    #     and value is a single value or a range of values separated by a colon
    #     (e.g., "max_depth=16" or "n_samples=1024:4096")
    ap = argparse.ArgumentParser(description="%(prog)s: run SPORF tests")
    ap.add_argument("csvFile", type=str, help="parameter-list .CSV file")  # (positional argument 1)
    ap.add_argument("csvTag", type=str, help="parameter-list ID tag")  # positional argument 2)
    args, params = ap.parse_known_args()

    # build a dict of filter values
    df = {"tag": [args.csvTag, args.csvTag, False]}
    for nvp in params:
        m = re.match(r"(\w+)=(\w+)(:(\w+))?", nvp)
        if not m:
            raise ValueError(f"Unable to parse filter expression as name=value: {nvp}")

        # save a range of values for the parameter to filter
        pname = m.group(1)
        try:
            pvalFrom = float(m.group(2))
            pvalTo = float(m.group(4)) if m.group(4) else pvalFrom
            pvalIsNumeric = True

        except ValueError:
            pvalFrom = m.group(2)
            pvalTo = m.group(4) if m.group(4) else pvalFrom
            pvalIsNumeric = False

        # if only one value is specified, duplicate it to form a single-valued "range"
        df[pname] = [pvalFrom, pvalTo, pvalIsNumeric]

    # build a list of parameter sets to be tested:
    #  - scan the .CSV for the row(s) whose first column contains the specified tag string
    #  - for each optional parameter name, filter the .CSV row by the value in the corresponding column
    ap = []
    with open(args.csvFile, newline="") as csvParams:
        rdr = csv.DictReader(csvParams)

        # validate the specified parameter names (.CSV column names) if any
        for pname in df.keys():
            if not pname in rdr.fieldnames:
                raise ValueError(f"Parameter '{pname}' not in .CSV file {args.csvFile}")

        for r in rdr:

            # filter the row on the specified parameter values
            ok = True
            for pname in df.keys():
                try:
                    if r[pname] < df[pname][0] or r[pname] > df[pname][1]:
                        ok = False
                        break
                except TypeError:
                    fval = float(r[pname])
                    if fval < df[pname][0] or fval > df[pname][1]:
                        ok = False
                        break

            # conditionally append the SPORF parameter set (excluding the 0th column) to the list
            if ok:
                ap.append(dict(itertools.islice(r.items(), 1, len(r))))

    # abort if there are no parameter sets (rows in the .CSV file) that pass the specified filter(s)
    if not ap:
        raise ValueError(
            f"No parameters in {args.csvFile} that match tag '{args.csvTag}' and specified filters."
        )

    # return a tuple consisting of...
    #   - the argparse Namespace (positional command-line parameters)
    #   - the parameter filter criteria (optional command-line parameters)
    #   - the list of test parameter sets that meet all the filter criteria
    return args, df, ap


#
# echo parameters to the output log
#
def logParams(args: argparse.Namespace, df: dict, ap: list[dict]) -> None:

    # echo the .CSV filename and the parameter tag string
    p.dtLog(f"csvFile: {args.csvFile}")
    p.dtLog(f"csvTag : {args.csvTag}")

    p.dtLog(f"Parameters in {args.csvFile} filtered by:")
    for pname in df.keys():
        if df[pname][2]:
            if df[pname][0] == df[pname][1]:
                p.dtLog(f" {pname}: {df[pname][0]:g} (numeric)")
            else:
                p.dtLog(f" {pname}: {df[pname][0]:g}:{df[pname][1]:g} (numeric)")
        else:
            if df[pname][0] == df[pname][1]:
                p.dtLog(f" {pname}: {df[pname][0]} (string)")
            else:
                p.dtLog(f" {pname}: {df[pname][0]}:{df[pname][1]} (string)")

    p.dtLog("")
    p.dtLog(f"Testing {len(ap)} parameter set{'s' if len(ap) != 1 else ''} from {args.csvFile}:")
    return


#
# execute one test iteration with the specified set of parameters
#
def doTest(dp: dict) -> None:

    p.dtLog(f"--- Start test: data={dp["dataset"]}...")

    try:
        # use a function in module 'test_data' to build the test data
        fn = getattr(td, dp["dataset"])

    except AttributeError as ex:
        raise AttributeError(f"Unrecognized dataset name: {dp["dataset"]}") from ex

    X_train, y_train, X_test, y_test = fn(n_samples=int(dp["n_samples"]), n_features=int(dp["n_features"]))

    # TODO: DO IT!

    p.dtLog(f"--- End test: data={dp["dataset"]}.")
    return


if __name__ == "__main__":

    # log interesting process metadata
    sScriptName = os.path.basename(__file__)
    p.dtLog(
        f"Start __name__='{__name__}' for {sScriptName} pid {os.getpid()} in python v{sys.version.split('|')[0]}..."
    )

    # parse the command line and load test parameters
    args, df, ap = loadParams()
    logParams(args, df, ap)

    # hard-coded breakpoint
    p.dtLog("")
    p.dtLog("Breakpoint so we can attach a C++ debugger to the Linux process!")
    breakpoint()
    p.dtLog("Resumed execution after breakpoint")
    p.dtLog("")

    # iterate through the specified sets of parameters
    for dp in ap:
        doTest(dp)

    # that's all, folks!
    p.dtLog(f"Exiting {sScriptName}")
    exit()

exit()


#
#
#


from sklearn.ensemble import RandomForestClassifier as skRFC
from sklearn.ensemble import RandomForestRegressor as skRFR
from sklearn.metrics import mean_squared_error, accuracy_score

import cudf
import json
import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np

from typing import Callable


def do_gpu(
    rf: cuRFC,
    fn_metric: Callable,
    n_est: int,
    X_train: cudf.DataFrame,
    y_train: cudf.Series,
    X_test: cudf.DataFrame,
    y_test: cudf.Series,
):
    gpu_rf_params = {
        "n_estimators": n_est,
        "max_depth": 16,
        "max_features": 5,
        "n_bins": 1024,
        "n_streams": 8,
        # "split_criterion": 1,  # 1: entropy (see algo_helper.h, randomforest_shared.pxd, and randomforest_common.pyx),
        ##  "criterion": "entropy",  # which is it, 'split_criterion' or 'criterion'? how do we know?
        ### SPORF-specific
        "density": 0.56789,
        #### TODO: PUT THIS BACK OR ACCEPT THE DEFAULT:  "histogram_method": 1,
    }

    gpu_rf = rf(**gpu_rf_params)

    # here's where we call our new API
    gpu_rf.fit(X_train, y_train)

    # TODO: HOW DO WE KNOW WHAT HAPPENED???

    # we call our new API again
    return fn_metric(gpu_rf.predict(X_test).to_numpy(), y_test.to_numpy())


def do_cpu(rf, fn_metric, n_est, X_train, y_train, X_test, y_test):
    sk_rf_params = {
        "n_estimators": n_est,
        "max_depth": 16,
        "max_features": 5,
        # 'criterion': 'squared_error',
        "n_jobs": mp.cpu_count(),
    }

    sk_rf = rf(**sk_rf_params)
    sk_rf.fit(X_train, y_train)

    return fn_metric(sk_rf.predict(X_test), y_test)


def test_rf(rf, fn, fn_metric, n_est, X_train, y_train, X_test, y_test):
    start = time.perf_counter()
    score = fn(rf, fn_metric, n_est, X_train, y_train, X_test, y_test)
    end = time.perf_counter()

    return {"score": score, "time": end - start}


def aggregates(results):
    return {
        "cpu_time": np.mean([r["time"] for r in results["cpu"]]),
        "gpu_time": np.mean([r["time"] for r in results["gpu"]]),
        "speedup": np.mean([r["time"] for r in results["cpu"]])
        / np.mean([r["time"] for r in results["gpu"]]),
        "cpu_score": np.mean([r["score"] for r in results["cpu"]]),
        "gpu_score": np.mean([r["score"] for r in results["gpu"]]),
        "gpu_mb": results["gpu_mb"],
    }


def plot_speedup_vs_volume(results_list, n_features, dtype_bytes, dataset_name="Friedman1"):
    """
    Plots speedup vs. data volume and sample count.

    Parameters:
    - results_list: List of (n_samples, results_dict) tuples.
    - n_features: Number of features (default: 27).
    - dtype_bytes: Size of the dtype in bytes (default: 8 for float64).
    """
    # Sort by n_samples for consistency
    results_list = sorted(results_list, key=lambda x: x[0])

    n_samples = np.array([n for n, _ in results_list])
    speedups = np.array([r["speedup"] for _, r in results_list])

    # Approximate total size in MB (X + y; assuming float64)
    data_volumes_MB = np.array(
        [r["gpu_mb"] for _, r in results_list]
    )  # ((n_features + 1) * n_samples * dtype_bytes) / (1024 ** 2)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot with log x-scale
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.plot(n_samples, speedups, marker="o", color="tab:blue")
    ax1.set_ylabel("Speedup (CPU / GPU)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Upper axis for data volume
    ax2 = ax1.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax1.get_xlim())

    # Set tick marks to match both axes
    ax1.set_xticks(n_samples)
    ax2.set_xticks(n_samples)
    ax2.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax2.set_xlabel("Number of Samples")

    ax1.set_xticklabels([f"{v:.1f}" for v in data_volumes_MB])
    ax1.set_xlabel("Data Volume (MB)")

    plt.title(f"GPU Speedup vs Data Volume ({dataset_name}, {n_features} features)")
    plt.tight_layout()
    plt.savefig(f"gpu_speedup_vs_data_volume-{dataset_name}.png", dpi=300)
    plt.show()


def main(dataset_name, f_data, n_features, n_est=100, n_trials=1):
    results_all = []
    for n_samples in [
        1_000,
        2_000,
        4_000,
        8_000,
        16_000,
        32_000,
        64_000,
        128_000,
        256_000,
        512_000,
        1_024_000,
        2_048_000,
        4_096_000,
        8_192_000,
    ]:
        r = f_data(n_samples=n_samples, n_features=n_features, n_est=n_est, n_trials=n_trials)
        agg = aggregates(r)
        print(f"n_samples: {n_samples}")
        print(f"Results: {json.dumps(r, indent=2)}")
        print(f"Aggregates: {json.dumps(agg, indent=2)}")
        results_all.append([n_samples, agg])

    plot_speedup_vs_volume(results_all, n_features=n_features, dtype_bytes=4, dataset_name=dataset_name)


if __name__ == "__main__":

    # we need the linux PID so that we can attach a C++ debugger
    import os

    print(f"python PID: {os.getpid()}")

    # hit a breakpoint here so we can attach
    breakpoint()
    print("Resumed execution after breakpoint")

    import argparse

    parser = argparse.ArgumentParser(description="Run Random Forest tests on GPU and CPU.")

    parser.add_argument(
        "--n_features",
        type=int,
        default=27,
        help="Number of features for the dataset (default: 27)",
    )

    parser.add_argument(
        "--n_est",
        type=int,
        default=100,
        help="Number of estimators (trees) in the forest (default: 100)",
    )

    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run (default: 1)")

    parser.add_argument(
        "--regression",
        action="store_true",
        help="Run regression tests instead of classification",
    )
    args = parser.parse_args()

    if args.regression:
        main(
            dataset_name="Friedman1",
            f_data=friedman,
            n_features=args.n_features,
            n_est=args.n_est,
            n_trials=args.n_trials,
        )
    else:
        main(
            dataset_name="Classification",
            f_data=classification,
            n_features=args.n_features,
            n_est=args.n_est,
            n_trials=args.n_trials,
        )

print("That's all, folks!")
