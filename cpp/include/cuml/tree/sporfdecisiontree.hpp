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

#include <cuml/random_projection/rproj_c.h>

#include "decisiontree.hpp"


namespace ML {

namespace DT {

typedef enum {
  HISTOGRAM_METHOD_EXACT = 0,
  HISTOGRAM_METHOD_SAMPLED = 1
} HISTOGRAM_METHOD;

struct SPORFDecisionTreeParams : DecisionTreeParams {
  /**
   * Additional parameter(s) required for SPORF
   */
  float density;
  HISTOGRAM_METHOD histogram_method;
};

template <class T, class L>
struct ObliqueTreeMetaDataNode : public TreeMetaDataNode<T, L> {
  std::vector<std::unique_ptr<ML::rand_mat<T>>> vector_projection_vectors;
};

template <typename DataT>
ML::rand_mat<DataT> clone_rand_mat(const ML::rand_mat<DataT>& src, ML::rand_mat<DataT>& dst) {
  dst.type = src.type;
  switch (src.type) {
    case ML::dense:
      dst.dense_data = rmm::device_uvector<DataT>(src.dense_data.size(), dst.stream);
      raft::copy(dst.dense_data.data(), src.dense_data.data(), src.dense_data.size(), dst.stream);
      break;
    case ML::sparse:
      dst.indices    = rmm::device_uvector<int>(src.indices.size(), dst.stream);
      dst.indptr     = rmm::device_uvector<int>(src.indptr.size(), dst.stream);
      dst.sparse_data= rmm::device_uvector<DataT>(src.sparse_data.size(), dst.stream);
      raft::copy(dst.indices.data(), src.indices.data(), src.indices.size(), dst.stream);
      raft::copy(dst.indptr.data(),  src.indptr.data(),  src.indptr.size(), dst.stream);
      raft::copy(dst.sparse_data.data(), src.sparse_data.data(), src.sparse_data.size(), dst.stream);
      break;
    case ML::unset:
    default: break;
  }
  return dst;
}

template <typename DataT>
ML::rand_mat<DataT> clone_rand_mat(const ML::rand_mat<DataT>& src) {
  ML::rand_mat<DataT> dst(src.stream());
  return clone_rand_mat(src, dst);
}


/***
 * TODO: maybe define alternate implementations for the following (defined in decisiontree.hpp):
 *        set_tree_params
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
