/*
 * sporf.cu
 *
 * Notes:
 *  Based on randomforest.cu
 * 
 *  In .../cuml/cpp/CMakeLists.txt, near line 500, add a reference to this file:
 *
 *    if(all_algo OR randomforest_algo)
 *      target_sources(${CUML_CPP_TARGET}
 *        PRIVATE
 *          src/randomforest/randomforest.cu
 *          src/randomforest/sporf.cu)
 *    endif()
 * 
 * 
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "sporf.cuh"

namespace ML {

using namespace MLCommon;
using namespace std;

/**
 * @brief Check validity of all random forest hyper-parameters.
 * @param[in] rf_params: random forest hyper-parameters
 */
void validity_check(const SPORF_params rf_params)
{
  /* (see the implementation of validity_check() in randomforest.cu) */
  ASSERT((rf_params.n_trees > 0), "Invalid n_trees %d", rf_params.n_trees);
  ASSERT((rf_params.max_samples > 0) && (rf_params.max_samples <= 1.0),
         "max_samples value %f outside permitted (0, 1] range",
         rf_params.max_samples);
}

/*
 * SPORF implementation of set_rf_params()
 *
 * (Original implementation in randomforest.cu)
 */
SPORF_params set_sporf_params(int max_depth,
                              int max_leaves,               // base RF parameters
                              float max_features,
                              int max_n_bins,
                              int min_samples_leaf,
                              int min_samples_split,
                              float min_impurity_decrease,
                              bool bootstrap,
                              int n_trees,
                              float max_samples,
                              uint64_t seed,
                              CRITERION split_criterion,
                              int cfg_n_streams,
                              int max_batch_size,
                              int TBD )                     // SPORF-specific parameters
{
  DT::SPORFDecisionTreeParams tree_params;
  DT::set_tree_params(tree_params,
                      max_depth,
                      max_leaves,
                      max_features,
                      max_n_bins,
                      min_samples_leaf,
                      min_samples_split,
                      min_impurity_decrease,
                      split_criterion,
                      max_batch_size);

  // initialize RF_params members
  tree_params.TBD = TBD;                      
  SPORF_params rf_params;
  rf_params.n_trees     = n_trees;
  rf_params.bootstrap   = bootstrap;
  rf_params.max_samples = max_samples;
  rf_params.seed        = seed;
  rf_params.n_streams   = min(cfg_n_streams, omp_get_max_threads());
  if (n_trees < rf_params.n_streams) rf_params.n_streams = n_trees;

  // initialize SPORF-specific members in the SPORFDecisionTreeParams struct
  tree_params.TBD = TBD;  

  rf_params.tree_params = tree_params;
  validity_check(rf_params);
  return rf_params;
}

/*
  * SPORF implementations of
  *    fit()
  *    predict()
  * (Original implementation in randomforest.cu)
  */

/**
 * @defgroup RandomForestRegressorFit Random Forest Regression - Fit function
 * @brief Build (i.e., fit, train) random forest regressor for input data.
 * @param[in] user_handle: raft::handle_t
 * @param[in,out] forest: CPU pointer to RandomForestMetaData object. User allocated.
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format,
 *   excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (float or double), with one label per
 *   training sample. Device pointer.
 * @param[in] rf_params: Random Forest training hyper parameter struct.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @{
 */
void fit(const raft::handle_t& user_handle,
                SPORFMetaData<float,int>*& forest,     // SPORFClassifierF*& forest,
                float* input,       // (32-bit floating point)
                int n_rows,
                int n_cols,
                int* labels,
                int n_unique_labels,
                SPORF_params rf_params,
                rapids_logger::level_enum verbosity)
{
  printf( "HELLO FROM %s LINE %d\n", __FILE__, __LINE__ );

  raft::common::nvtx::range fun_scope("RF::fit @sporf.cu");
  ML::default_logger().set_level(verbosity);
  ASSERT(forest->trees.empty(), "Cannot fit an existing forest.");
  forest->trees.resize(rf_params.n_trees);
  forest->rf_params = rf_params;

  std::shared_ptr<SPORF<float, int>> rf_classifier =
    std::make_shared<SPORF<float, int>>(rf_params, RF_type::CLASSIFICATION);
  rf_classifier->fit(user_handle, input, n_rows, n_cols, labels, n_unique_labels, forest);
}

void fit(const raft::handle_t& user_handle,
                SPORFMetaData<double,int>*& forest,    // SPORFClassifierD*& forest,
                double* input,       // (64-bit floating point)
                int n_rows,
                int n_cols,
                int* labels,
                int n_unique_labels,
                SPORF_params rf_params,
                rapids_logger::level_enum verbosity)
{
  raft::common::nvtx::range fun_scope("RF::fit @sporf.cu");
  ML::default_logger().set_level(verbosity);
  ASSERT(forest->trees.empty(), "Cannot fit an existing forest.");
  forest->trees.resize(rf_params.n_trees);
  forest->rf_params = rf_params;

  std::shared_ptr<SPORF<double, int>> rf_classifier =
    std::make_shared<SPORF<double, int>>(rf_params, RF_type::CLASSIFICATION);
  rf_classifier->fit(user_handle, input, n_rows, n_cols, labels, n_unique_labels, forest);
}
/** @} */

/**
 * @defgroup RandomForestClassificationPredict Random Forest Classification - Predict function
 * @brief Predict target feature for input data; n-ary classification for
     single feature supported.
 * @param[in] user_handle: raft::handle_t.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @{
 */
void predict(const raft::handle_t& user_handle,
              const SPORFClassifierF* forest,
              const float* input,       // (32-bit floating point)
              int n_rows,
              int n_cols,
              int* predictions,
              rapids_logger::level_enum verbosity)
{
  ASSERT(!forest->trees.empty(), "Cannot predict! No trees in the forest.");
  std::shared_ptr<SPORF<float, int>> rf_classifier =
    std::make_shared<SPORF<float, int>>(forest->rf_params, RF_type::CLASSIFICATION);
  rf_classifier->predict(user_handle, input, n_rows, n_cols, predictions, forest, verbosity);
}

void predict(const raft::handle_t& user_handle,
              const SPORFClassifierD* forest,
              const double* input,        // (64-bit floating point)
              int n_rows,
              int n_cols,
              int* predictions,
              rapids_logger::level_enum verbosity)
{
  ASSERT(!forest->trees.empty(), "Cannot predict! No trees in the forest.");
  std::shared_ptr<SPORF<double, int>> rf_classifier =
    std::make_shared<SPORF<double, int>>(forest->rf_params, RF_type::CLASSIFICATION);
  rf_classifier->predict(user_handle, input, n_rows, n_cols, predictions, forest, verbosity);
}

/**
 * @defgroup RandomForestClassificationScore Random Forest Classification - Score function
 * @brief Compare predicted features validate against ref_labels.
 * @param[in] user_handle: raft::handle_t.
 * @param[in] forest: CPU pointer to SPORFMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @return RF_metrics struct with classification score (i.e., accuracy)
 * @{
 */
RF_metrics score(const raft::handle_t& user_handle,
                 const SPORFClassifierF* forest,
                 const int* ref_labels,
                 int n_rows,
                 const int* predictions,
                 rapids_logger::level_enum verbosity)
{
  RF_metrics classification_score = SPORF<float, int>::score(
    user_handle, ref_labels, n_rows, predictions, verbosity, RF_type::CLASSIFICATION);
  return classification_score;
}

RF_metrics score(const raft::handle_t& user_handle,
                 const SPORFClassifierD* forest,
                 const int* ref_labels,
                 int n_rows,
                 const int* predictions,
                 rapids_logger::level_enum verbosity)
{
  RF_metrics classification_score = SPORF<double, int>::score(
    user_handle, ref_labels, n_rows, predictions, verbosity, RF_type::CLASSIFICATION);
  return classification_score;
}

}  // End namespace ML
