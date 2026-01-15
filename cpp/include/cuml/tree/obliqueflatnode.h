/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include "flatnode.h"

// We want to define some functions as usable on device
// But need to guard against this file being compiled by a host compiler
#ifdef __CUDACC__
#define FLATNODE_HD __host__ __device__
#else
#define FLATNODE_HD
#endif

/**
 * A node in Decision Tree.
 * @tparam T data type
 * @tparam L label type
 * @tparam IdxT type used for indexing operations
 */
template <typename DataT, typename LabelT, typename IdxT = int>
struct ObliqueSparseTreeNode : public SparseTreeNode<DataT, LabelT, IdxT> {
 private:
  using Base = SparseTreeNode<DataT, LabelT, IdxT>;
  std::unique_ptr<ML::rand_mat<DataT>> rand_vector;

  FLATNODE_HD ObliqueSparseTreeNode(IdxT colid, DataT quesval, DataT best_metric_val, int64_t left_child_id, IdxT instance_count)
    : Base(colid, quesval, best_metric_val, left_child_id, instance_count), rand_vector(nullptr)
  {
  }

  FLATNODE_HD ObliqueSparseTreeNode(
    IdxT colid,
    DataT quesval,
    DataT best_metric_val,
    int64_t left_child_id,
    IdxT instance_count,
    std::unique_ptr<ML::rand_mat<DataT>> rv)
    : Base(colid, quesval, best_metric_val, left_child_id, instance_count), rand_vector(std::move(rv))
  {
  }

 public:
  FLATNODE_HD const ML::rand_mat<DataT>* RandomVector() const { return rand_vector.get(); }

  FLATNODE_HD static ObliqueSparseTreeNode<DataT, LabelT, IdxT> CreateSplitNode(
    IdxT colid,
    DataT quesval,
    DataT best_metric_val,
    int64_t left_child_id,
    IdxT instance_count,
    std::unique_ptr<ML::rand_mat<DataT>> rv)
  {
    return ObliqueSparseTreeNode<DataT, LabelT, IdxT>{
      colid, quesval, best_metric_val, left_child_id, instance_count, std::move(rv)};
  }
  FLATNODE_HD static ObliqueSparseTreeNode<DataT, LabelT, IdxT> CreateLeafNode(IdxT instance_count)
  {
    return ObliqueSparseTreeNode<DataT, LabelT, IdxT>{0, 0, 0, -1, instance_count, nullptr};
  }
  FLATNODE_HD bool IsLeaf() const { return this->LeftChildId() == -1; }
  bool operator==(const ObliqueSparseTreeNode& other) const
  {
    return (this->ColumnId() == other.ColumnId()) &&
           (this->QueryValue() == other.QueryValue()) &&
           (this->BestMetric() == other.BestMetric()) &&
           (this->LeftChildId() == other.LeftChildId()) &&
           (this->InstanceCount() == other.InstanceCount());
  }
};
