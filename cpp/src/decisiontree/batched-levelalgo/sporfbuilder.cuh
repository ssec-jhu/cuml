/*
 * sporfbuilder.cuh
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

#include "kernels/builder_kernels.cuh"
#include "kernels/sporf_builder_kernels.cuh"

#include <common/Timer.h>

#include <cublas_v2.h>

#include <cuml/common/pinned_host_vector.hpp>
#include <cuml/random_projection/rproj_c.h>
#include <cuml/tree/decisiontree.hpp>
#include <cuml/tree/sporfdecisiontree.hpp>
#include <cuml/tree/flatnode.h>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/matrix/matrix.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <deque>
#include <memory>
#include <utility>

namespace ML {
namespace DT {


/**
 * Structure that manages the iterative batched-level training and building of nodes
 * in the host.
 */
template <typename DataT, typename LabelT>
class SPORFNodeQueue {
  using NodeT = SparseTreeNode<DataT, LabelT>;
  using TreeMetaDataNodeT = DT::ObliqueTreeMetaDataNode<DataT, LabelT>;
  const SPORFDecisionTreeParams params;
  std::shared_ptr<TreeMetaDataNodeT> tree;
  std::vector<SPORFDT::InstanceRange> node_instances_;
  std::deque<SPORFDT::NodeWorkItem> work_items_;

 public:
  SPORFNodeQueue(SPORFDecisionTreeParams params, size_t max_nodes, size_t sampled_rows, int num_outputs)
    : params(params), tree(std::make_shared<TreeMetaDataNodeT>())
  {
    tree->num_outputs = num_outputs;
    tree->sparsetree.reserve(max_nodes);
    tree->sparsetree.emplace_back(NodeT::CreateLeafNode(sampled_rows));
    tree->projection_vectors.reserve(max_nodes);
    tree->projection_vectors.resize(max_nodes);
    tree->leaf_counter  = 1;
    tree->depth_counter = 0;
    node_instances_.reserve(max_nodes);
    node_instances_.emplace_back(SPORFDT::InstanceRange{0, sampled_rows});
    if (this->IsExpandable(tree->sparsetree.back(), 0)) {
      work_items_.emplace_back(SPORFDT::NodeWorkItem{0, 0, 0, node_instances_.back()});
    }
  }

  std::shared_ptr<TreeMetaDataNodeT> GetTree() { return tree; }
  const std::vector<SPORFDT::InstanceRange>& GetInstanceRanges() { return node_instances_; }

  bool HasWork() { return work_items_.size() > 0; }

  auto Pop()
  {
    std::vector<SPORFDT::NodeWorkItem> result;
    result.reserve(std::min(size_t(params.max_batch_size), work_items_.size()));
    while (work_items_.size() > 0 && result.size() < std::size_t(params.max_batch_size)) {
      result.emplace_back(work_items_.front());
      work_items_.pop_front();
    }
    return result;
  }

  // This node is allowed to be expanded further (if its split gain is high enough)
  bool IsExpandable(const NodeT& n, int depth)
  {
    if (depth >= params.max_depth) return false;
    if (int(n.InstanceCount()) < params.min_samples_split) return false;
    if (params.max_leaves != -1 && tree->leaf_counter >= params.max_leaves) return false;
    return true;
  }

  template <typename SplitT>
  void Push(const std::vector<SPORFDT::NodeWorkItem>& work_items, SplitT* h_splits, std::vector<std::unique_ptr<rand_mat<DataT>>>& h_sparse_matrices)
  {
    // Update node queue based on splits
    for (std::size_t i = 0; i < work_items.size(); i++) {
      // printf( "Pushing work item %d at %s LINE %d\n", i, __FILE__, __LINE__ );
      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );

      auto split        = h_splits[i];
      auto item         = work_items[i];
      auto parent_range = node_instances_.at(item.idx);
      if (SplitNotValid(
            split, params.min_impurity_decrease, params.min_samples_leaf, parent_range.count)) {
        continue;
      }
      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );

      if (params.max_leaves != -1 && tree->leaf_counter >= params.max_leaves) break;

      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
      // parent
      tree->sparsetree.at(item.idx) = NodeT::CreateSplitNode(split.colid,
                                                             split.quesval,
                                                             split.best_metric_val,
                                                             int64_t(tree->sparsetree.size()),
                                                             parent_range.count);
      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
      tree->projection_vectors.at(item.idx) = std::make_unique<rand_mat<DataT>>(h_sparse_matrices[i]->stream);
      clone_rand_mat(*h_sparse_matrices[i], *tree->projection_vectors.at(item.idx));
      tree->leaf_counter++;
      // left
      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
      tree->sparsetree.emplace_back(NodeT::CreateLeafNode(split.nLeft));
      node_instances_.emplace_back(SPORFDT::InstanceRange{parent_range.begin, std::size_t(split.nLeft)});
      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );

      // Do not add a work item if this child is definitely a leaf
      if (this->IsExpandable(tree->sparsetree.back(), item.depth + 1)) {
        work_items_.emplace_back(
          SPORFDT::NodeWorkItem{tree->sparsetree.size() - 1, item.depth + 1, 0, node_instances_.back()});
      }
      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );

      // right
      tree->sparsetree.emplace_back(NodeT::CreateLeafNode(parent_range.count - split.nLeft));
      node_instances_.emplace_back(
        SPORFDT::InstanceRange{parent_range.begin + split.nLeft, parent_range.count - split.nLeft});

      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
      // Do not add a work item if this child is definitely a leaf
      if (this->IsExpandable(tree->sparsetree.back(), item.depth + 1)) {
        work_items_.emplace_back(
          SPORFDT::NodeWorkItem{tree->sparsetree.size() - 1, item.depth + 1, 0, node_instances_.back()});
      }

      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
      // update depth
      tree->depth_counter = max(tree->depth_counter, item.depth + 1);
    }
  }
};

/**
 * Internal struct used to do all the heavy-lifting required for tree building
 */
template <typename ObjectiveT>
struct SPORFBuilder {
  typedef typename ObjectiveT::DataT DataT;
  typedef typename ObjectiveT::LabelT LabelT;
  typedef typename ObjectiveT::IdxT IdxT;
  typedef typename ObjectiveT::BinT BinT;
  typedef SparseTreeNode<DataT, LabelT, IdxT> NodeT;
  typedef ObliqueTreeMetaDataNode<DataT, LabelT> TreeMetaDataNodeT;
  typedef Split<DataT, IdxT> SplitT;
  typedef Dataset<DataT, LabelT, IdxT> DatasetT;
  typedef Quantiles<DataT, IdxT> QuantilesT;


  /** default threads per block for most kernels in here */
  static constexpr int TPB_DEFAULT = 128;
  // When n_bins <= TPB, using >1 item per thread makes sorted quantiles
  // vanish into out-of-range slots. Keep this at 1.
  static constexpr int ITEMS_PER_THREAD = 1;
  /** handle to get device properties */
  const raft::handle_t& handle;
  /** stream to launch kernels */
  cudaStream_t builder_stream;
  /** DT params */
  SPORFDecisionTreeParams params;
  /** input dataset */
  DatasetT dataset;
  DatasetT dataset_proj;
  /** quantiles */
  QuantilesT quantiles;
  /** Tree index */
  IdxT treeid;
  /** Seed used for randomization */
  uint64_t seed;
  /** number of nodes created in the current batch */
  IdxT* n_nodes;
  /** buffer of segmented histograms*/
  BinT* histograms;
  /** threadblock arrival count */
  int* done_count;
  /** mutex array used for atomically updating best split */
  int* mutex;
  /** best splits for the current batch of nodes */
  SplitT* splits;
  /** current batch of nodes */
  SPORFDT::NodeWorkItem* d_work_items;
  /** device AOS to map CTA blocks along dimx to nodes of a batch */
  SPORFDT::WorkloadInfo<IdxT>* workload_info;
  /** host AOS to map CTA blocks along dimx to nodes of a batch */
  SPORFDT::WorkloadInfo<IdxT>* h_workload_info;
  /** maximum CTA blocks along dimx */
  int max_blocks_dimx = 0;
  /** host array of splits */
  SplitT* h_splits;
  /** number of blocks used to parallelize column-wise computations */
  int n_blks_for_cols = 10;
  /** Memory alignment value */
  const size_t align_value = 512;
  IdxT* d_colids;
  IdxT* h_colids;
  /** rmm device workspace buffer */
  rmm::device_uvector<char> d_buff;
  /** pinned host buffer to store the trained nodes */
  ML::pinned_host_vector<char> h_buff;
  /**
  device buffer for contiguous input data

  dataset.row_ids is n_sampled_rows long.
  each node gets its own extent within dataset.row_ids, disjoint from other nodes.
  d_contiguous indices correspond to dataset.row_ids indices,
  so that we may copy from dataset.data[dataset.row_ids[i]] into d_contiguous[i],
  and node row_id extents map directly to d_contiguous extents.
  */
  std::vector<IdxT> h_quantile_indices;
  rmm::device_uvector<IdxT> d_quantile_indices;
  rmm::device_uvector<DataT> d_quantiles;
  rmm::device_uvector<DataT> d_contiguous;
  rmm::device_uvector<DataT> d_trans;
  std::vector<std::unique_ptr<rand_mat<DataT>>> h_sparse_matrices;
  DatasetT dataset_trans;

  SPORFBuilder(const raft::handle_t& handle,
          cudaStream_t s,
          IdxT treeid,
          uint64_t seed,
          const SPORFDecisionTreeParams& p,
          const DataT* data,
          const LabelT* labels,
          IdxT n_rows,
          IdxT n_cols,
          rmm::device_uvector<IdxT>* row_ids,
          IdxT n_classes,
          const QuantilesT& q)
    : handle(handle),
      builder_stream(s),
      treeid(treeid),
      seed(seed),
      params(p),
      dataset{data,
              labels,
              n_rows,
              n_cols,
              int(row_ids->size()),
              max(1, IdxT(params.max_features * n_cols)),
              row_ids->data(),
              n_classes
      },
      quantiles(q),
      d_quantile_indices(0, builder_stream),
      d_quantiles(0, builder_stream),
      d_buff(0, builder_stream),
      d_contiguous(0, builder_stream),
      d_trans(0, builder_stream),
      dataset_trans{
        0,
        labels,
        n_rows,
        max(1, IdxT(params.max_features * n_cols)),
        int(row_ids->size()),
        max(1, IdxT(params.max_features * n_cols)),
        row_ids->data(),
        n_classes
      },
      dataset_proj{
        0,
        labels,
        n_rows,
        max(1, IdxT(params.max_features * n_cols)),
        int(row_ids->size()),
        max(1, IdxT(params.max_features * n_cols)),
        row_ids->data(),
        n_classes
      }
  {
    max_blocks_dimx = 1 + params.max_batch_size + dataset.n_sampled_rows / TPB_DEFAULT;
    // ASSERT(q.quantiles_array != nullptr && q.n_bins_array != nullptr,
    //        "Currently quantiles need to be computed before this call!");
    ASSERT(n_classes >= 1, "n_classes should be at least 1");
    ASSERT(TPB_DEFAULT * ITEMS_PER_THREAD >= params.max_n_bins,
      "max_n_bins must be <= 2048 for proper functioning of quantile sorting.");

    h_quantile_indices.resize(params.max_batch_size * dataset.n_sampled_cols * params.max_n_bins);

    size_t req_bytes     = size_t(params.max_batch_size) * size_t(dataset.n_sampled_cols) * params.max_n_bins * sizeof(IdxT);
    size_t aligned_bytes = calculateAlignedBytes(req_bytes);
    size_t aligned_elems = aligned_bytes / sizeof(IdxT);
    d_quantile_indices.resize(aligned_elems, builder_stream);

    req_bytes     = size_t(params.max_batch_size) * size_t(dataset.n_sampled_cols) * params.max_n_bins * sizeof(DataT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(DataT);
    d_quantiles.resize(aligned_elems, builder_stream);

    // allocate d_contiguous with byte-alignment: calculateAlignedBytes works in bytes
    req_bytes     = size_t(dataset.n_sampled_rows) * size_t(n_cols) * sizeof(DataT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(DataT);
    d_contiguous.resize(aligned_elems, builder_stream);

    req_bytes     = size_t(dataset.n_sampled_rows) * size_t(dataset.n_sampled_cols) * sizeof(DataT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(DataT);
    d_trans.resize(aligned_elems, builder_stream);
    dataset_trans.data = d_trans.data();

    // initialize host-owned rand_mat objects so they own their device buffers
    h_sparse_matrices.reserve(params.max_batch_size * dataset.n_sampled_cols);
    for (int i = 0; i < params.max_batch_size; ++i) {
      h_sparse_matrices.emplace_back(std::make_unique<rand_mat<DataT>>(builder_stream));
    }

    auto [device_workspace_size, host_workspace_size] = workspaceSize();
    d_buff.resize(device_workspace_size, builder_stream);
    h_buff.resize(host_workspace_size);
    assignWorkspace(d_buff.data(), h_buff.data());
  }

  /**
   * @brief calculates nearest aligned size of input w.r.t an `align_value`.
   *
   * @param[in] actual_size actual size in bytes of input
   * @return aligned size
   */
  size_t calculateAlignedBytes(const size_t actual_size) const
  {
    return raft::alignTo(actual_size, align_value);
  }

  /**
   * @brief returns maximum nodes possible per tree
   * @return maximum nodes possible per tree
   */
  size_t maxNodes() const
  {
    if (params.max_depth < 13) {
      // Start with allocation for a dense tree for depth < 13
      return pow(2, (params.max_depth + 1)) - 1;
    } else {
      // Start with fixed size allocation for depth >= 13
      return 8191;
    }
  }

  /**
   * @brief calculate the workspace size required
   *
   * @return a pair of device workspace and host workspace size requirements
   */
  auto workspaceSize() const
  {
    size_t d_wsize = 0, h_wsize = 0;
    raft::common::nvtx::range fun_scope("SPORFBuilder::workspaceSize @sporfbuilder.cuh [batched-levelalgo]");
    auto max_batch = params.max_batch_size;
    size_t max_len_histograms =
      max_batch * params.max_n_bins * n_blks_for_cols * dataset.num_outputs;

    d_wsize += calculateAlignedBytes(sizeof(IdxT));                               // n_nodes
    d_wsize += calculateAlignedBytes(sizeof(BinT) * max_len_histograms);          // histograms
    d_wsize += calculateAlignedBytes(sizeof(int) * max_batch * n_blks_for_cols);  // done_count
    d_wsize += calculateAlignedBytes(sizeof(int) * max_batch);                    // mutex
    d_wsize += calculateAlignedBytes(sizeof(SplitT) * max_batch);                 // splits
    d_wsize += calculateAlignedBytes(sizeof(SPORFDT::NodeWorkItem) * max_batch);           // d_work_Items
    d_wsize +=                                                                    // workload_info
      calculateAlignedBytes(sizeof(SPORFDT::WorkloadInfo<IdxT>) * max_blocks_dimx);
    d_wsize += calculateAlignedBytes(sizeof(IdxT) * max_batch * dataset.n_sampled_cols);  // colids

    // all nodes in the tree
    h_wsize +=  // h_workload_info
      calculateAlignedBytes(sizeof(SPORFDT::WorkloadInfo<IdxT>) * max_blocks_dimx);
    h_wsize += calculateAlignedBytes(sizeof(SplitT) * max_batch);  // splits
    h_wsize += calculateAlignedBytes(sizeof(IdxT) * max_batch * dataset.n_sampled_cols);  // colids



    //if( this->builder_stream == handle.get_stream_from_stream_pool(0) )
  // printf( "SPORFBuilder::workspaceSize (%s line %d): d_wsize=%lu h_wsize=%lu\n", __FILE__, __LINE__, d_wsize, h_wsize );


    return std::make_pair(d_wsize, h_wsize);
  }

  /**
   * @brief assign workspace to the current state
   *
   * @param[in] d_wspace device buffer allocated by the user for the workspace.
   *                     Its size should be at least workspaceSize()
   * @param[in] h_wspace pinned host buffer needed to store the learned nodes
   */
  void assignWorkspace(char* d_wspace, char* h_wspace)
  {
    raft::common::nvtx::range fun_scope(
      "SPORFBuilder::assignWorkspace @sporfbuilder.cuh [batched-levelalgo]");
    auto max_batch  = params.max_batch_size;
    auto n_col_blks = n_blks_for_cols;
    size_t max_len_histograms =
      max_batch * (params.max_n_bins) * n_blks_for_cols * dataset.num_outputs;
    // device
    n_nodes = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(IdxT));
    histograms = reinterpret_cast<BinT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(BinT) * max_len_histograms);
    done_count = reinterpret_cast<int*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(int) * max_batch * n_col_blks);
    mutex = reinterpret_cast<int*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(int) * max_batch);
    splits = reinterpret_cast<SplitT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(SplitT) * max_batch);
    d_work_items = reinterpret_cast<SPORFDT::NodeWorkItem*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(SPORFDT::NodeWorkItem) * max_batch);
    workload_info = reinterpret_cast<SPORFDT::WorkloadInfo<IdxT>*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(SPORFDT::WorkloadInfo<IdxT>) * max_blocks_dimx);
    d_colids = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(IdxT) * max_batch * dataset.n_sampled_cols);

    RAFT_CUDA_TRY(
      cudaMemsetAsync(done_count, 0, sizeof(int) * max_batch * n_col_blks, builder_stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, builder_stream));

    // host
    h_workload_info = reinterpret_cast<SPORFDT::WorkloadInfo<IdxT>*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(SPORFDT::WorkloadInfo<IdxT>) * max_blocks_dimx);
    h_splits = reinterpret_cast<SplitT*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(SplitT) * max_batch);
    h_colids = reinterpret_cast<IdxT*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(IdxT) * max_batch * dataset.n_sampled_cols);
  }

  /**
   * @brief trains the tree, builds the nodes
   *
   * @return trained tree structure
   */
  std::shared_ptr<TreeMetaDataNodeT> train()
  {
    raft::common::nvtx::range fun_scope("SPORFBuilder::train @sporfbuilder.cuh [batched-levelalgo]");
    MLCommon::TimerCPU timer;
    SPORFNodeQueue<DataT, LabelT> queue(
      params, this->maxNodes(), dataset.n_sampled_rows, dataset.num_outputs);
    while (queue.HasWork()) {
      // printf( "INVOKING Pop at %s LINE %d\n", __FILE__, __LINE__ );
      auto work_items                      = queue.Pop();
      printf( "\n***INVOKING doSplit at %s LINE %d\n", __FILE__, __LINE__ );
      auto [splits_host_ptr, splits_count] = doSplit(work_items);
      // printf( "INVOKING Push at %s LINE %d\n", __FILE__, __LINE__ );
      queue.Push(work_items, splits_host_ptr, h_sparse_matrices);
    }
    // printf( "INVOKING GetTree at %s LINE %d\n", __FILE__, __LINE__ );
    auto tree = queue.GetTree();
    this->SetLeafPredictions(tree, queue.GetInstanceRanges());
    tree->train_time = timer.getElapsedMilliseconds();

    std::cout << "\n\nOK HERE'S THE TREE:\n";
    for(size_t i = 0; i < tree->sparsetree.size(); i++) {
      NodeT* node = &tree->sparsetree[i];
      if( node->IsLeaf() ) {
        std::cout << "Node " << i << " LEAF instance_count " << node->InstanceCount() << " prediction ";
        for(int o = 0; o < dataset.num_outputs; o++ ) {
          std::cout << tree->vector_leaf[i * dataset.num_outputs + o] << " ";
        }
        std::cout << "\n";
      } else {
        std::cout << "Node " << i << " quesval " << node->QueryValue() << " best_metric_val " << node->BestMetric() << " instance_count " << node->InstanceCount() << "\n projection_vector:\n";
        print_rand_mat(*(tree->projection_vectors.data()[i]), builder_stream);
      }
    }
    printf("\n\n");

    std::vector<DataT> h_input(dataset.M * dataset.N);
    raft::update_host(h_input.data(), dataset.data, dataset.M * dataset.N, builder_stream);
    std::vector<LabelT> h_labels(dataset.M);
    raft::update_host(h_labels.data(), dataset.labels, dataset.M, builder_stream);
    std::vector<IdxT> h_row_ids(dataset.n_sampled_rows);
    raft::update_host(h_row_ids.data(), dataset.row_ids, dataset.n_sampled_rows, builder_stream);
    auto ranges = queue.GetInstanceRanges();
    std::cout << "sparsetree size " << tree->sparsetree.size() << " ranges size " << ranges.size() << " row_ids size: " << dataset.n_sampled_rows << "\n";
    std::cout << "And here's the leaves:\n";
    for(size_t i = 0; i < tree->sparsetree.size(); i++) {
      NodeT* node = &tree->sparsetree[i];
      if( node->IsLeaf() ) {
        std::cout << "Node " << i << " LEAF instance_count " << node->InstanceCount() << std::endl;
        std::cout << "  range begin " << ranges[i].begin << " count " << ranges[i].count << " row_id indices: ";

        for(auto r = ranges[i].begin; r < ranges[i].begin + ranges[i].count; r++ ) {
          // std::cout << " " << dataset.row_ids[r];
          std::cout << " " << r;
        }

        std::cout.flush();
        std::cout << " row_id values: ";
        for(auto r = ranges[i].begin; r < ranges[i].begin + ranges[i].count; r++ ) {
          std::cout << " " << h_row_ids[r];
        }
        std::cout << std::endl;

        std::cout << "  data values:\n";
        for(auto r = ranges[i].begin; r < ranges[i].begin + ranges[i].count; r++ ) {
          auto row_id = h_row_ids[r];
          std::cout << "    row_id " << row_id << ": ";
          for(int c = 0; c < dataset.N; c++ ) {
            std::cout << h_input[c * dataset.M + row_id] << " ";
          }
          std::cout << " | " << h_labels[row_id] << std::endl;
        }
        std::cout << std::endl;
      }
    }
    std::cout << "\n\n";

    std::cout << "HERE'S THE RAW DATA AGAIN:\n";
    for(int i = 0; i < dataset.M; i++) {
      std::cout << "row_id " << i << ": ";
      for(int c = 0; c < dataset.N; c++ ) {
        std::cout << h_input[c * dataset.M + i] << " ";
      }
      std::cout << " | " << h_labels[i] << std::endl;
    }
    return tree;
  }

 private:
  auto updateWorkloadInfo(const std::vector<SPORFDT::NodeWorkItem>& work_items)
  {
    int n_large_nodes = 0;  // large nodes are nodes having training instances larger than block
                            // size, hence require global memory for histogram construction
    int n_blocks_dimx = 0;  // gridDim.x required for computeSplitKernel
    for (std::size_t i = 0; i < work_items.size(); i++) {
      auto item = work_items[i];
      int n_blocks_per_node =
        std::max(raft::ceildiv(item.instances.count, size_t(TPB_DEFAULT)), size_t(1));

      if (n_blocks_per_node > 1) ++n_large_nodes;

      for (int b = 0; b < n_blocks_per_node; b++) {
        h_workload_info[n_blocks_dimx + b] = {int(i), n_large_nodes - 1, b, n_blocks_per_node};
      }
      n_blocks_dimx += n_blocks_per_node;
    }
    raft::update_device(workload_info, h_workload_info, n_blocks_dimx, builder_stream);
    return std::make_pair(n_blocks_dimx, n_large_nodes);
  }

  auto doSplit(const std::vector<SPORFDT::NodeWorkItem>& work_items)
  {
    raft::common::nvtx::range fun_scope("SPORFBuilder::doSplit @sporfbuilder.cuh [batched-levelalgo]");
    // start fresh on the number of *new* nodes created in this batch
    RAFT_CUDA_TRY(cudaMemsetAsync(n_nodes, 0, sizeof(IdxT), builder_stream));
    initSplit<DataT, IdxT, TPB_DEFAULT>(splits, work_items.size(), builder_stream);

    // get the current set of nodes to be worked upon
    raft::update_device(d_work_items, work_items.data(), work_items.size(), builder_stream);

    auto [n_blocks_dimx, n_large_nodes] = this->updateWorkloadInfo(work_items);

    // printf( "HELLO FROM %s LINE %d\n", __FILE__, __LINE__ );

    for(IdxT i = 0; i < params.max_batch_size; i++ ) {
      for( IdxT c = 0; c < dataset.n_sampled_cols; c++ ) {
        h_colids[i * dataset.n_sampled_cols + c] = c;
      }
    }
    raft::update_device(d_colids, h_colids, work_items.size() * dataset.n_sampled_cols, builder_stream);

    // printf( "at %s LINE %d\n", __FILE__, __LINE__ );

    // TODO: parallelize this over work_items
    for (size_t i = 0; i < work_items.size(); i++) {
      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
      auto& begin = work_items[i].instances.begin;
      auto& count = work_items[i].instances.count;

      if (count < static_cast<unsigned long>(params.min_samples_split)) continue;

      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
      raft::matrix::copyRows<DataT, IdxT, size_t>(
        dataset.data,                     // in
        count,                            // number of rows of output matrix
        dataset.N,                        // number of columns of output matrix
        d_contiguous.data() + begin, // d_contiguous.data() + begin * dataset.N,    // out
        dataset.row_ids + begin,        // row indices to copy
        count,
        builder_stream,
        false                             // do-row-major (false, i.e. do column-major)
      );
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
      // TODO: fix rproj to use better types than int for everything
      paramsRPROJ rproj_params{
        static_cast<int>(count), // number of samples
        dataset.N,               // number of features
        1,                       // number of components
        -1.0f,                   // error tolerance (not used)
        false,                   // gaussian or sparse method
        -1.0,                    // auto density (-1: auto-deduction)
        false,                   // not used
        0                        // random seed
      };

      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
      std::vector<IdxT> universe(work_items[i].instances.count);
      std::iota(universe.begin(), universe.end(), 0);

      for(int c = 0; c < dataset.n_sampled_cols; c++) {
        // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
        rand_mat<DataT>& random_matrix = *(h_sparse_matrices[i * dataset.n_sampled_cols + c]);
        random_matrix.reset();
        // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
        auto random_state = static_cast<int>((seed + static_cast<uint64_t>(treeid) + i + c) &
                                            0x7fffffffULL);
      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
        rproj_params.random_state = random_state;
      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
        RPROJfit(handle, &random_matrix, &rproj_params);
        // printf("candidate random projection matrix for node %zu, colid=%d:\n", i, c);
        // print_rand_mat(random_matrix, builder_stream);
      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
        RPROJtransform<DataT>(
          handle,
          d_contiguous.data() + begin,
          &random_matrix,
          d_trans.data() + (c * dataset.n_sampled_rows) + begin,
          &rproj_params
        );

      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
        std::mt19937_64 rng(seed + static_cast<uint64_t>(treeid) + i + c);
      // printf( "at %s LINE %d\n", __FILE__, __LINE__ );
        // TODO: randomize the order here!
        std::sample(
          universe.begin(), universe.end(), h_quantile_indices.begin() + (i * dataset.n_sampled_cols * params.max_n_bins) + (c * params.max_n_bins), params.max_n_bins, rng
        );
      }
    }
    // printf( "at %s LINE %d\n", __FILE__, __LINE__ );

    raft::update_device(d_quantile_indices.data(), h_quantile_indices.data(), h_quantile_indices.size(), builder_stream);

    std::vector<LabelT> h_labels(dataset.M); // labels come from the (full) raw dataset
    RAFT_CUDA_TRY(cudaMemcpyAsync(h_labels.data(), dataset.labels, dataset.M * sizeof(LabelT), cudaMemcpyDeviceToHost, builder_stream));
    std::vector<DataT> h_trans(dataset.n_sampled_cols * dataset.n_sampled_rows); // transformed data is just for the sampled rows and projected columns
    RAFT_CUDA_TRY(cudaMemcpyAsync(h_trans.data(), d_trans.data(), dataset.n_sampled_cols * dataset.n_sampled_rows * sizeof(DataT), cudaMemcpyDeviceToHost, builder_stream));
    std::vector<IdxT> h_row_ids(dataset.n_sampled_rows); // row_ids also just for the sampled rows
    RAFT_CUDA_TRY(cudaMemcpyAsync(h_row_ids.data(), dataset.row_ids, dataset.n_sampled_rows * sizeof(IdxT), cudaMemcpyDeviceToHost, builder_stream));
    for(size_t i = 0; i < work_items.size(); i++ ) {
      std::cout << "Node " << work_items[i].idx << " transformed data: " << std::endl;
      for(int c = 0; c < dataset.n_sampled_cols; c++ ) {
        std::cout << "  col " << c << ": ";
        for(auto r = work_items[i].instances.begin; r < work_items[i].instances.begin + work_items[i].instances.count; r++ ) {
          auto row_id = h_row_ids[r];
          std::cout << "  row " << row_id << ": " << h_trans[c * dataset.n_sampled_rows + r] << " ";
        }
        std::cout << std::endl;
      }
    }

    dataset_proj.data = d_trans.data();

    // printf( "at %s LINE %d\n", __FILE__, __LINE__ );

    // iterate through a batch of columns (to reduce the memory pressure) and
    // compute the best split at the end
    for (IdxT c = 0; c < dataset.n_sampled_cols; c += n_blks_for_cols) {
      printf( "HELLO FROM %s LINE %d\n", __FILE__, __LINE__ );
      printf( "c=%d n_blocks_dimx=%d n_large_nodes=%d\n", c, n_blocks_dimx, n_large_nodes);
      computeSplit(c, dataset_proj, n_blocks_dimx, n_large_nodes);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      // printf( "HELLO FROM %s LINE %d\n", __FILE__, __LINE__ );
      // printf("\n");
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(builder_stream));
    printf( "Completed computeSplit loop at %s LINE %d\n", __FILE__, __LINE__ );

      raft::update_host(h_splits, splits, work_items.size(), builder_stream);
      RAFT_CUDA_TRY(cudaStreamSynchronize(builder_stream));
      for(size_t i = 0; i < work_items.size(); i++ ) {
        printf("winner node %lu: colid=%d, quesval=%f, best_metric_val=%f, nLeft=%d\n", work_items[i].idx, h_splits[i].colid, h_splits[i].quesval, h_splits[i].best_metric_val, h_splits[i].nLeft);
      }
    // YEAH DON'T CALL THE FOLLOWING FUNCTION. EVER. MY EXPERIENCE WITH IT HAS BEEN THAT IT OVERWRITES THE MEMORY THAT YOU'RE TRYING TO PRINT
    // DT::printSplits(splits,
    //                 static_cast<IdxT>(work_items.size()),
    //                 builder_stream);
    // printf( "\n\n");

    // printf( "OUT OF LOOP AT %s LINE %d\n", __FILE__, __LINE__ );

    // create child nodes (or make the current ones leaf)
    raft::common::nvtx::push_range("nodeSplitKernel @sporfbuilder.cuh [batched-levelalgo]");
    SPORFDT::launchNodeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT>(params.max_depth,
                                                            params.min_samples_leaf,
                                                            params.min_samples_split,
                                                            params.max_leaves,
                                                            params.min_impurity_decrease,
                                                            dataset_proj,
                                                            d_work_items,
                                                            work_items.size(),
                                                            splits,
                                                            builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // printf( "HELLO FROM %s LINE %d\n", __FILE__, __LINE__ );

    raft::common::nvtx::pop_range();
    raft::update_host(h_splits, splits, work_items.size(), builder_stream);
    handle.sync_stream(builder_stream);

    for(size_t i = 0; i < work_items.size(); i++ ) {
      printf("random_matrix for node %zu, colid=%d:\n", work_items[i].idx, h_splits[i].colid);
      if(h_splits[i].colid < 0 || h_splits[i].colid >= dataset.n_sampled_cols ) {
        printf("  invalid colid %d\n", h_splits[i].colid);
        continue;
      }
      print_rand_mat(*h_sparse_matrices[i * dataset.n_sampled_cols + h_splits[i].colid], builder_stream);
    }
    // printf( "HELLO FROM %s LINE %d\n", __FILE__, __LINE__ );

    return std::make_tuple(h_splits, work_items.size());
  }

  auto computeSplitSmemSize()
  {
    size_t smem_size_1 =
      params.max_n_bins * dataset.num_outputs * sizeof(BinT) +  // shared_histogram size
      params.max_n_bins * sizeof(DataT) +                       // shared_quantiles size
      sizeof(int);                                              // shared_done size
    // Extra room for alignment (see alignPointer in
    // computeSplitKernel)
    smem_size_1 += sizeof(DataT) + 3 * sizeof(int);
    // Calculate the shared memory needed for evalBestSplit
    size_t smem_size_2 = raft::ceildiv(TPB_DEFAULT, raft::WarpSize) * sizeof(SplitT);
    // Pick the max of two
    auto available_smem = handle.get_device_properties().sharedMemPerBlock;
    size_t smem_size    = std::max(smem_size_1, smem_size_2);
    ASSERT(available_smem >= smem_size, "Not enough shared memory. Consider reducing max_n_bins.");

    // printf( "SPORFBuilder::computeSplitSmemSize: smem_size=%ld\n", static_cast<size_t>(smem_size) );

    return smem_size;
  }

  void computeSplit(IdxT col, DatasetT& dataset, size_t n_blocks_dimx, size_t n_large_nodes)
  {
    // if no instances to split, return
    if (n_blocks_dimx == 0) return;
    raft::common::nvtx::range fun_scope("SPORFBuilder::computeSplit @sporfbuilder.cuh [batched-levelalgo]");
    auto n_bins    = params.max_n_bins;
    auto n_classes = dataset.num_outputs;
    // if columns left to be processed lesser than `n_blks_for_cols`, shrink the blocks along dimy
    auto n_blocks_dimy = std::min(n_blks_for_cols, dataset.n_sampled_cols - col);
    // compute required dynamic shared memory
    auto smem_size = computeSplitSmemSize();
    dim3 grid(n_blocks_dimx, n_blocks_dimy, 1);
    // required total length (in bins) of the global segmented histograms over all
    // classes, features and (large)nodes.
    int len_histograms = n_bins * n_classes * n_blocks_dimy * n_large_nodes;
    RAFT_CUDA_TRY(cudaMemsetAsync(histograms, 0, sizeof(BinT) * len_histograms, builder_stream));
    // create the objective function object
    ObjectiveT objective(dataset.num_outputs, params.min_samples_leaf);
    // call the computeSplitKernel
    // printf( "HELLO FROM %s LINE %d\n", __FILE__, __LINE__ );
    raft::common::nvtx::range kernel_scope("computeSplitKernel @sporfbuilder.cuh [batched-levelalgo]");
    launchComputeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT, ITEMS_PER_THREAD>(histograms,
                                                                    params.max_n_bins,
                                                                    params.max_depth,
                                                                    params.min_samples_split,
                                                                    params.min_samples_leaf,
                                                                    params.max_leaves,
                                                                    dataset,
                                                                    d_quantile_indices.data(),
                                                                    d_work_items,
                                                                    col,
                                                                    d_colids,
                                                                    done_count,
                                                                    mutex,
                                                                    splits,
                                                                    objective,
                                                                    treeid,
                                                                    workload_info,
                                                                    seed,
                                                                    grid,
                                                                    smem_size,
                                                                    builder_stream);
    // printf( "HELLO FROM %s LINE %d\n", __FILE__, __LINE__ );
  }

  // Set the leaf value predictions in batch
  void SetLeafPredictions(std::shared_ptr<TreeMetaDataNodeT> tree,
                          const std::vector<SPORFDT::InstanceRange>& instance_ranges)
  {
    tree->vector_leaf.resize(tree->sparsetree.size() * dataset.num_outputs);
    ASSERT(tree->sparsetree.size() == instance_ranges.size(),
           "Expected instance range for each node");
    // do this in batch to reduce peak memory usage in extreme cases
    std::size_t max_batch_size = min(std::size_t(100000), tree->sparsetree.size());
    rmm::device_uvector<NodeT> d_tree(max_batch_size, builder_stream);
    rmm::device_uvector<SPORFDT::InstanceRange> d_instance_ranges(max_batch_size, builder_stream);
    rmm::device_uvector<DataT> d_leaves(max_batch_size * dataset.num_outputs, builder_stream);

    ObjectiveT objective(dataset.num_outputs, params.min_samples_leaf);
    for (std::size_t batch_begin = 0; batch_begin < tree->sparsetree.size();
         batch_begin += max_batch_size) {
      std::size_t batch_end  = min(batch_begin + max_batch_size, tree->sparsetree.size());
      std::size_t batch_size = batch_end - batch_begin;
      raft::update_device(
        d_tree.data(), tree->sparsetree.data() + batch_begin, batch_size, builder_stream);
      raft::update_device(
        d_instance_ranges.data(), instance_ranges.data() + batch_begin, batch_size, builder_stream);

      RAFT_CUDA_TRY(
        cudaMemsetAsync(d_leaves.data(), 0, sizeof(DataT) * d_leaves.size(), builder_stream));
      size_t smem_size = sizeof(BinT) * dataset.num_outputs;
      SPORFDT::launchLeafKernel(objective,
                       dataset,
                       d_tree.data(),
                       d_instance_ranges.data(),
                       d_leaves.data(),
                       batch_size,
                       smem_size,
                       builder_stream);
      raft::update_host(tree->vector_leaf.data() + batch_begin * dataset.num_outputs,
                        d_leaves.data(),
                        batch_size * dataset.num_outputs,
                        builder_stream);
    }
  }
};  // end Builder

}  // namespace DT
}  // namespace ML
