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

#include "decisiontree.hpp"
#include <cuda_runtime.h>
#include <rmm/device_uvector.hpp>
#include <raft/util/cudart_utils.hpp>
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

template <typename IdxT = int, typename OffsetT = std::size_t>
struct OffsetProjectionMatrix {
  // For persistent device-side tree storage, offsets are relative to the
  // training workspace backing buffers rather than absolute pointers.
  IdxT n_proj_components;
  OffsetT indptr_offset;
  OffsetT indices_offset;
  OffsetT coeffs_offset;
};

template <typename DataT, typename IdxT = int>
struct OwnedProjectionMatrix {
  explicit OwnedProjectionMatrix(cudaStream_t stream)
    : indptr(0, stream), indices(0, stream), coeffs(0, stream), stream(stream)
  {
  }

  rmm::device_uvector<IdxT> indptr;
  rmm::device_uvector<IdxT> indices;
  rmm::device_uvector<DataT> coeffs;
  cudaStream_t stream;

  ProjectionMatrix<DataT, IdxT> view() const
  {
    return ProjectionMatrix<DataT, IdxT>{
      static_cast<IdxT>(indptr.size() > 0 ? indptr.size() - 1 : 0),
      indptr.data(),
      indices.data(),
      coeffs.data()
    };
  }
};

static constexpr int BLOCK_TASK_SIZE = 128; // heuristic for number of threads per block for GPU kernels

template <typename IdxT = int>
struct NodeWorkItemChunk {
  IdxT work_item_idx;      // index into the batch of work items for this level of the tree
  IdxT node_id;            // global tree node id for this chunk/work item
  IdxT instances_begin;    // start of indices into dataset.row_ids for this block and node
  IdxT instances_count;    // number of indices into dataset.row_ids for this block and node
  IdxT block_task_idx;     // index into the batch of block tasks for this level of the tree
  IdxT thread_local_begin; // starting thread index within this block doing work on this node
  IdxT nLeft;              // number of left child instances for this work item in this block
  IdxT nRight;             // number of right child instances for this work item in this block
  IdxT loff;               // offset into the left child partition of the output row_id array for this block and node
  IdxT roff;               // offset into the right child partition of the output row_id array for this block and node
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
struct ObliqueTreeMetaDataNode {
  int treeid;
  int depth_counter;
  int leaf_counter;
  double train_time;
  std::vector<T> vector_leaf;
  std::vector<SparseTreeNode<T, L>> sparsetree;
  int num_outputs;
  std::vector<OffsetProjectionMatrix<int>> projection_vectors;
  std::vector<int> projection_indptr_storage;
  std::vector<int> projection_indices_storage;
  std::vector<T> projection_coeffs_storage;
};

template <typename DataT, typename IdxT = int>
void clone_column_to_column_vector(const OwnedProjectionMatrix<DataT, IdxT>& src,
                                   int src_col_id,
                                   OwnedProjectionMatrix<DataT, IdxT>& dst) {
  if (src_col_id < 0 || src_col_id + 1 >= static_cast<int>(src.indptr.size())) {
    throw std::runtime_error("clone_column_to_column_vector: source column index out of range");
  }

  IdxT col_ptrs[2]{0, 0};
  raft::update_host(col_ptrs, src.indptr.data() + src_col_id, std::size_t(2), dst.stream);
  if (cudaStreamSynchronize(dst.stream) != cudaSuccess) {
    throw std::runtime_error("clone_column_to_column_vector: failed to synchronize stream");
  }

  IdxT start = col_ptrs[0];
  IdxT end = col_ptrs[1];
  IdxT n_nonzero = end - start;
  if (n_nonzero < 0) { throw std::runtime_error("clone_column_to_column_vector: invalid CSC indptr"); }

  dst.indptr = rmm::device_uvector<IdxT>(2, dst.stream);
  dst.indices = rmm::device_uvector<IdxT>(n_nonzero, dst.stream);
  dst.coeffs = rmm::device_uvector<DataT>(n_nonzero, dst.stream);

  IdxT dst_col_ptrs[2]{0, n_nonzero};
  raft::update_device(dst.indptr.data(), dst_col_ptrs, std::size_t(2), dst.stream);

  if (n_nonzero > 0) {
    raft::copy(dst.indices.data(), src.indices.data() + start, n_nonzero, dst.stream);
    raft::copy(dst.coeffs.data(), src.coeffs.data() + start, n_nonzero, dst.stream);
  }
}

template <typename DataT, typename IdxT = int>
void copy_projection_matrix_to_owned(const ProjectionMatrix<DataT, IdxT>& src,
                                     OwnedProjectionMatrix<DataT, IdxT>& dst) {
  auto n_proj_components = src.n_proj_components;
  auto indptr_len = static_cast<size_t>(n_proj_components + 1);
  std::vector<IdxT> h_indptr(indptr_len, 0);
  if (indptr_len > 0) {
    raft::update_host(h_indptr.data(), src.d_proj_indptr, indptr_len, dst.stream);
    if (cudaStreamSynchronize(dst.stream) != cudaSuccess) {
      throw std::runtime_error("copy_projection_matrix_to_owned: failed to synchronize stream");
    }
  }
  auto nnz = indptr_len > 0 ? static_cast<size_t>(h_indptr.back()) : 0;

  dst.indptr = rmm::device_uvector<IdxT>(indptr_len, dst.stream);
  dst.indices = rmm::device_uvector<IdxT>(nnz, dst.stream);
  dst.coeffs = rmm::device_uvector<DataT>(nnz, dst.stream);

  if (indptr_len > 0) { raft::copy(dst.indptr.data(), src.d_proj_indptr, indptr_len, dst.stream); }
  if (nnz > 0) {
    raft::copy(dst.indices.data(), src.d_proj_indices, nnz, dst.stream);
    raft::copy(dst.coeffs.data(), src.d_proj_coeffs, nnz, dst.stream);
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
