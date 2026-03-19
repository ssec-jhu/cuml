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
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>


namespace ML {

namespace DT {

enum HISTOGRAM_METHOD : int {
  HISTOGRAM_METHOD_EXACT = 0,
  HISTOGRAM_METHOD_SAMPLED = 1
};

template <typename DataT, typename IdxT = int>
struct ProjectionMatrix {
  // TODO: rename n_proj_components to n_output_dims
  IdxT n_proj_components; // number of projection components (columns in projection matrix)
  const IdxT* d_proj_indptr; // projection matrix component non-zero-coefficient counts
  const IdxT* d_proj_indices; // projection matrix component column indices
  const DataT* d_proj_coeffs; // projection matrix component non-zero coefficients
};

static constexpr int BLOCK_TASK_SIZE = 128; // heuristic for number of threads per block for GPU kernels

template <typename IdxT = int>
struct NodeWorkItemChunk {
  IdxT work_item_idx;      // index into the batch of work items for this level of the tree
  IdxT instances_begin;    // start of indices into dataset.row_ids for this block and node
  IdxT instances_count;    // number of indices into dataset.row_ids for this block and node
  IdxT block_task_idx;     // index into the batch of block tasks for this level of the tree
  IdxT thread_local_begin; // starting thread index within this block doing work on this node
  IdxT payload_idx;        // index into array of chunk-specific task info (i.e. projection matrices, partition splits, etc) loaded for this batch
  IdxT nLeft;              // number of left child instances for this work item in this block
  IdxT nRight;             // number of right child instances for this work item in this block
};

template <typename IdxT = int>
struct BlockTask {
  IdxT work_item_chunk_ids[BLOCK_TASK_SIZE]; // index into the batch of work item chunks for this level of the tree
  IdxT count;                                // number of rows in this block, in [0...BLOCK_TASK_SIZE]
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

template <typename DataT>
void clone_column_to_column_vector(const ML::rand_mat<DataT>& src, int src_col_id, ML::rand_mat<DataT>& dst) {
  if (src.type != ML::sparse) {
    throw std::runtime_error("clone_column_to_column_vector expects a sparse source matrix");
  }
  if (src_col_id < 0 || src_col_id + 1 >= static_cast<int>(src.indptr.size())) {
    throw std::runtime_error("clone_column_to_column_vector: source column index out of range");
  }

  int col_ptrs[2]{0, 0};
  raft::update_host(col_ptrs, src.indptr.data() + src_col_id, std::size_t(2), dst.stream);
  if (cudaStreamSynchronize(dst.stream) != cudaSuccess) {
    throw std::runtime_error("clone_column_to_column_vector: failed to synchronize stream");
  }

  int start     = col_ptrs[0];
  int end       = col_ptrs[1];
  int n_nonzero = end - start;
  if (n_nonzero < 0) { throw std::runtime_error("clone_column_to_column_vector: invalid CSC indptr"); }

  dst.type        = ML::sparse;
  dst.indptr      = rmm::device_uvector<int>(2, dst.stream);
  dst.indices     = rmm::device_uvector<int>(n_nonzero, dst.stream);
  dst.sparse_data = rmm::device_uvector<DataT>(n_nonzero, dst.stream);

  int dst_col_ptrs[2]{0, n_nonzero};
  raft::update_device(dst.indptr.data(), dst_col_ptrs, std::size_t(2), dst.stream);

  if (n_nonzero > 0) {
    raft::copy(dst.indices.data(), src.indices.data() + start, n_nonzero, dst.stream);
    raft::copy(dst.sparse_data.data(), src.sparse_data.data() + start, n_nonzero, dst.stream);
  }
}

template <typename DataT>
void print_rand_mat(const ML::rand_mat<DataT>& mat, cudaStream_t stream) {
  switch (mat.type) {
    case ML::dense:
      printf("Dense matrix:\n");
      raft::print_device_vector("data",
                                mat.dense_data.data(),
                                mat.dense_data.size(),
                                std::cout);
      break;
    case ML::sparse:
      printf("Sparse matrix in CSC format:\n");
      raft::print_device_vector(
        "indices", mat.indices.data(), mat.indices.size(), std::cout);
      raft::print_device_vector(
        "indptr", mat.indptr.data(), mat.indptr.size(), std::cout);
      raft::print_device_vector(
        "data", mat.sparse_data.data(), mat.sparse_data.size(), std::cout);
      break;
    case ML::unset:
    default:
      printf("Empty matrix\n");
  }
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
