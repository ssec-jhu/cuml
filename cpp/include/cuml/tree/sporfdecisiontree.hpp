/*
 * sporfdecisiontree.hpp
 *
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <cuml/random_projection/rproj_c.h>   // TODO: IS THIS NEEDED?

#include "decisiontree.hpp"


namespace ML {

namespace DT {

enum HISTOGRAM_METHOD : int {
  HISTOGRAM_METHOD_EXACT = 0,
  HISTOGRAM_METHOD_SAMPLED = 1
};

struct SPORFDecisionTreeParams : DecisionTreeParams {
  /**
   * Additional parameter(s) required for SPORF
   */
  float density;
  HISTOGRAM_METHOD histogram_method;
};

template <class T, class L>
struct ObliqueTreeMetaDataNode : public TreeMetaDataNode<T, L> {
  std::vector<std::unique_ptr<ML::rand_mat<T>>> projection_vectors;
};

template <typename DataT>
void clone_rand_mat(const ML::rand_mat<DataT>& src, ML::rand_mat<DataT>& dst) {
  dst.type = src.type;
  switch (src.type) {
    case ML::dense:
      dst.dense_data = rmm::device_uvector<DataT>(src.dense_data.size(), dst.stream);
      raft::copy(dst.dense_data.data(), src.dense_data.data(), src.dense_data.size(), dst.stream);
      break;
    case ML::sparse:
      dst.indices     = rmm::device_uvector<int>(src.indices.size(), dst.stream);
      dst.indptr      = rmm::device_uvector<int>(src.indptr.size(), dst.stream);
      dst.sparse_data = rmm::device_uvector<DataT>(src.sparse_data.size(), dst.stream);
      raft::copy(dst.indices.data(), src.indices.data(), src.indices.size(), dst.stream);
      raft::copy(dst.indptr.data(), src.indptr.data(), src.indptr.size(), dst.stream);
      raft::copy(dst.sparse_data.data(), src.sparse_data.data(), src.sparse_data.size(), dst.stream);
      break;
    case ML::unset:
    default: break;
  }
}

template <typename DataT>
ML::rand_mat<DataT> clone_rand_mat(const ML::rand_mat<DataT>& src) {
  ML::rand_mat<DataT> dst(src.stream);
  clone_rand_mat(src, dst);
  return dst;  // move/NRVO, no copy
}

/**
 * @brief Set all DecisionTreeParams members.
 * @param[in,out] params: update with tree parameters
 * @param[in] cfg_max_depth: maximum tree depth; default -1
 * @param[in] cfg_max_leaves: maximum leaves; default -1
 * @param[in] cfg_max_features: maximum number of features; default 1.0f
 * @param[in] cfg_max_n_bins: maximum number of bins; default 128
 * @param[in] cfg_min_samples_leaf: min. rows in each leaf node; default 1
 * @param[in] cfg_min_samples_split: min. rows needed to split an internal node;
 *            default 2
 * @param[in] cfg_min_impurity_decrease: split a node only if its reduction in
 *                                       impurity is more than this value
 * @param[in] cfg_split_criterion: split criterion; default CRITERION_END,
 *            i.e., GINI for classification or MSE for regression
 * @param[in] cfg_max_batch_size: Maximum number of nodes that can be processed
              in a batch. This is used only for batched-level algo. Default
              value 4096.
 */
void set_tree_params(SPORFDecisionTreeParams& params,
                     int cfg_max_depth               = -1,
                     int cfg_max_leaves              = -1,
                     float cfg_max_features          = 1.0f,
                     int cfg_max_n_bins              = 128,
                     int cfg_min_samples_leaf        = 1,
                     int cfg_min_samples_split       = 2,
                     float cfg_min_impurity_decrease = 0.0f,
                     CRITERION cfg_split_criterion   = CRITERION_END,
                     int cfg_max_batch_size          = 4096,
                     float density                   = 1.0,
                     HISTOGRAM_METHOD                = HISTOGRAM_METHOD_EXACT);

/***
 * TODO: maybe define alternate implementations for the following (defined in decisiontree.hpp):
 *        TreeMetaDataNode
 *        get_tree_summary_text
 *        get_tree_text
 *        get_tree_json
 *        TreeClassifierF;
 *        TreeClassifierD;
 *        TreeRegressorF;
 *        TreeRegressorD;
 */

}  // End namespace DT
}  // End namespace ML
