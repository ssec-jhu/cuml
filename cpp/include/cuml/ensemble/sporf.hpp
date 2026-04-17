/*
 * sporf.hpp
 * 
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuml/common/logger.hpp>
#include <cuml/ensemble/treelite_defs.hpp>

#include <cuml/tree/sporfdecisiontree.hpp>

#include <cuml/ensemble/randomforest.hpp>

#include <map>
#include <memory>

namespace raft {
class handle_t;  // forward decl
}

namespace ML {



struct SPORF_params {

  /* we cannot inherit from RF_params because it already includes member DT::DecisionTreeParams,
      so we replicate struct RF_params here (see randomforest.hpp) 
      
     TODO: clean this up
  */
  int n_trees;
  bool bootstrap;
  float max_samples;
  uint64_t seed;
  int n_streams;

  // subclass of DT::DecisionTreeParams
  DT::SPORFDecisionTreeParams tree_params;
};

template <class T, class L>
struct SPORFMetaData {
  std::vector<std::shared_ptr<DT::ObliqueTreeMetaDataNode<T, L>>> trees;
  SPORF_params rf_params;
};


// ----------------------------- Classification ----------------------------------- //

typedef SPORFMetaData<float, int> SPORFClassifierF;
typedef SPORFMetaData<double, int> SPORFClassifierD;

void get_label_metadata(const raft::handle_t& user_handle,
                        const int* labels,
                        int n_rows,
                        int* n_unique_labels,
                        bool* is_dense_zero_based);

void fit(const raft::handle_t& user_handle,
         SPORFClassifierF*& forest,
         float* input,
         int n_rows,
         int n_cols,
         int* labels,
         int n_unique_labels,
         SPORF_params sporf_params,             // TODO: isn't this struct also a member of SPORFClassifierF?!
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

void fit(const raft::handle_t& user_handle,
         SPORFClassifierD*& forest,
         double* input,
         int n_rows,
         int n_cols,
         int* labels,
         int n_unique_labels,
         SPORF_params sporf_params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

void predict(const raft::handle_t& user_handle,
             const SPORFClassifierF* forest,
             const float* input,
             int n_rows,
             int n_cols,
             int* predictions,
             rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

void predict(const raft::handle_t& user_handle,
             const SPORFClassifierD* forest,
             const double* input,
             int n_rows,
             int n_cols,
             int* predictions,
             rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

RF_metrics score(const raft::handle_t& user_handle,
                 const SPORFClassifierF* forest,
                 const int* ref_labels,
                 int n_rows,
                 const int* predictions,
                 rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

RF_metrics score(const raft::handle_t& user_handle,
                 const SPORFClassifierD* forest,
                 const int* ref_labels,
                 int n_rows,
                 const int* predictions,
                 rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

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
                              float density,                // SPORF paramsters
                              DT::HISTOGRAM_METHOD histogram_method );

// ----------------------------- Regression ----------------------------------- //

typedef SPORFMetaData<float, float> SPORFRegressorF;
typedef SPORFMetaData<double, double> SPORFRegressorD;

void fit(const raft::handle_t& user_handle,
         SPORFRegressorF*& forest,
         float* input,
         int n_rows,
         int n_cols,
         float* labels,
         SPORF_params rf_params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

void fit(const raft::handle_t& user_handle,
         SPORFRegressorD*& forest,
         double* input,
         int n_rows,
         int n_cols,
         double* labels,
         SPORF_params rf_params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

void predict(const raft::handle_t& user_handle,
             const SPORFRegressorF* forest,
             const float* input,
             int n_rows,
             int n_cols,
             float* predictions,
             rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

void predict(const raft::handle_t& user_handle,
             const SPORFRegressorD* forest,
             const double* input,
             int n_rows,
             int n_cols,
             double* predictions,
             rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

RF_metrics score(const raft::handle_t& user_handle,
                 const SPORFRegressorF* forest,
                 const float* ref_labels,
                 int n_rows,
                 const float* predictions,
                 rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

RF_metrics score(const raft::handle_t& user_handle,
                 const SPORFRegressorD* forest,
                 const double* ref_labels,
                 int n_rows,
                 const double* predictions,
                 rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);
};  // namespace ML
