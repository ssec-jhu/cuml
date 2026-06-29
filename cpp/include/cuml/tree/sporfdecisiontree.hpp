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

struct SPORFDeviceBatchingPolicy {
  int num_sms = 0;
  int max_threads_per_sm = 0;
  std::size_t max_resident_threads = 0;
  std::size_t target_rows_per_batch = 65536;
  std::size_t target_blocks_per_batch = 0;
};

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
  float density_specified;
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
