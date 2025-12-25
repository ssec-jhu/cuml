#
# test_data.py
#

from dataclasses import dataclass
import numpy as np

from sklearn.datasets import make_friedman1, make_classification
import cudf


#
# training and test data
#
@dataclass
class RFTestData:

    X_train: cudf.core.dataframe.DataFrame
    y_train: cudf.core.series.Series
    X_test: cudf.core.dataframe.DataFrame
    y_test: cudf.core.series.Series


#
#
def higgs(testParams: dict) -> RFTestData:

    df_higgs = cudf.read_csv(testParams["higgs_csv"], header=None, nrows=5_000_000)
    df_train = df_higgs.sample(frac=0.95, random_state=int(testParams["random_state"]))
    X_train = df_train.iloc[:, 1:]
    y_train = df_train.iloc[:, 0]

    df_test = df_higgs.drop(index=df_train.index)
    X_test = df_test.iloc[:, 1:]
    y_test = df_test.iloc[:, 0]

    return RFTestData(X_train, y_train, X_test, y_test)


#    N_EST = 100
#    N_TRIALS = 1

#     X_train = X_train_gpu
#     y_train = y_train_gpu
#     X_test = X_test_gpu
#     y_test = y_test_gpu


#   print(f"Forest size = {N_EST}")
#   results_gpu = []
# for i in range(N_TRIALS):
#     print(f"Running gpu trial {i} of {N_TRIALS}")
#     res = test_rf(cuRFC, do_gpu, accuracy_score, N_EST, X_train, y_train, X_test, y_test)
#     results_gpu.append(res)
#     print(f"Results: {json.dumps(res, indent=2)}")

# X_train = X_train_gpu.to_numpy()
# y_train = y_train_gpu.to_numpy()
# X_test = X_test_gpu.to_numpy()
# y_test = y_test_gpu.to_numpy()

# results_sklearn = []
# results_ydf = []
# for i in range(N_TRIALS):
#     print(f"Running sklearn trial {i} of {N_TRIALS}")
#     res = test_rf(skRFC, do_sklearn, accuracy_score, N_EST, X_train, y_train, X_test, y_test)
#     results_sklearn.append(res)
#     print(f"Results: {json.dumps(res, indent=2)}")

#     print(f"Running ydf trial {i} of {N_TRIALS}")
#     res = test_rf(
#         Task.CLASSIFICATION,
#         do_ydf,
#         accuracy_score,
#         N_EST,
#         X_train,
#         y_train.astype(int),
#         X_test,
#         y_test.astype(int),
#     )
#     results_ydf.append(res)
#     print(f"Results: {json.dumps(res, indent=2)}")

# return {
#     "sklearn": results_sklearn,
#     "ydf": results_ydf,
#     "gpu": results_gpu,
#     "gpu_mb": df_higgs.memory_usage(deep=True).sum() / 1024**2,
# }


def classification(testParams: dict) -> RFTestData:

    X, y = make_classification(
        n_samples=int(testParams["n_samples"]),
        n_features=int(testParams["n_features"]),
        n_informative=5,
        n_classes=2,
        random_state=int(testParams["random_state"]),
    )

    dfX = cudf.DataFrame(X.astype(np.float32))
    X_train = dfX.sample(frac=0.95, random_state=int(testParams["random_state"]))
    X_test = dfX.drop(index=X_train.index)

    sy = cudf.Series(y.astype(np.float32))
    y_train = sy[X_train.index]
    y_test = sy[X_test.index]

    return RFTestData(X_train, y_train, X_test, y_test)

    # N_EST = n_est
    # N_TRIALS = n_trials

    # X_train = X_train_gpu
    # y_train = y_train_gpu
    # X_test = X_test_gpu
    # y_test = y_test_gpu
    # print(f"Forest size = {N_EST}")
    # results_gpu = []
    # for i in range(N_TRIALS):
    #     print(f"Running gpu trial {i} of {N_TRIALS}")
    #     res = test_rf(cuRFC, do_gpu, accuracy_score, N_EST, X_train, y_train, X_test, y_test)
    #     results_gpu.append(res)
    #     print(f"Results: {json.dumps(res, indent=2)}")

    # X_train = X_train_gpu.to_numpy()
    # y_train = y_train_gpu.to_numpy()
    # X_test = X_test_gpu.to_numpy()
    # y_test = y_test_gpu.to_numpy()
    # results_sklearn = []
    # results_ydf = []
    # for i in range(N_TRIALS):
    #     print(f"Running cpu trial {i} of {N_TRIALS}")
    #     res = test_rf(skRFC, do_sklearn, accuracy_score, N_EST, X_train, y_train, X_test, y_test)
    #     results_sklearn.append(res)
    #     print(f"Results: {json.dumps(res, indent=2)}")

    #     print(f"Running ydf trial {i} of {N_TRIALS}")
    #     res = test_rf(
    #         Task.CLASSIFICATION,
    #         do_ydf,
    #         accuracy_score,
    #         N_EST,
    #         X_train,
    #         y_train.astype(int),
    #         X_test,
    #         y_test.astype(int),
    #     )
    #     results_ydf.append(res)
    #     print(f"Results: {json.dumps(res, indent=2)}")

    # return {
    #     "sklearn": results_sklearn,
    #     "ydf": results_ydf,
    #     "gpu": results_gpu,
    #     "gpu_mb": (X.memory_usage(deep=True).sum() / 1024**2) + (y.memory_usage(deep=True) / 1024**2),
    # }


def friedman(testParams: dict) -> RFTestData:

    X, y = make_friedman1(
        n_samples=int(testParams["n_samples"]),
        n_features=int(testParams["n_features"]),
        random_state=int(testParams["random_state"]),  # TODO: DEFAULT = 42?
    )

    dfX = cudf.DataFrame(X.astype(np.float32))
    sy = cudf.Series(y.astype(np.float32))

    X_train = dfX.sample(frac=0.95, random_state=42)
    X_test = dfX.drop(index=X_train.index)
    y_train = sy[X_train.index]
    y_test = sy[X_test.index]

    return RFTestData(X_train, y_train, X_test, y_test)

    # N_EST = n_est
    # N_TRIALS = n_trials

    # X_train = X_train_gpu
    # y_train = y_train_gpu
    # X_test = X_test_gpu
    # y_test = y_test_gpu
    # print(f"Forest size = {N_EST}")
    # results_gpu = []
    # for i in range(N_TRIALS):
    #     print(f"Running gpu trial {i} of {N_TRIALS}")
    #     res = test_rf(cuRFR, do_gpu, mean_squared_error, N_EST, X_train, y_train, X_test, y_test)
    #     results_gpu.append(res)
    #     print(f"Results: {json.dumps(res, indent=2)}")

    # X_train = X_train_gpu.to_numpy()
    # y_train = y_train_gpu.to_numpy()
    # X_test = X_test_gpu.to_numpy()
    # y_test = y_test_gpu.to_numpy()
    # ds_ydf = {f"f{i}": X_train[:, i] for i in range(X_train.shape[1])}
    # ds_ydf["l"] = y_train

    # results_sklearn = []
    # results_ydf = []
    # for i in range(N_TRIALS):
    #     print(f"Running cpu trial {i} of {N_TRIALS}")
    #     res = test_rf(skRFR, do_sklearn, mean_squared_error, N_EST, X_train, y_train, X_test, y_test)
    #     results_sklearn.append(res)
    #     print(f"Results: {json.dumps(res, indent=2)}")

    #     print(f"Running ydf trial {i} of {N_TRIALS}")
    #     res = test_rf(Task.REGRESSION, do_ydf, mean_squared_error, N_EST, X_train, y_train, X_test, y_test)
    #     results_ydf.append(res)
    #     print(f"Results: {json.dumps(res, indent=2)}")

    # return {
    #     "sklearn": results_sklearn,
    #     "ydf": results_ydf,
    #     "gpu": results_gpu,
    #     "gpu_mb": (X.memory_usage(deep=True).sum() / 1024**2) + (y.memory_usage(deep=True) / 1024**2),
    # }
