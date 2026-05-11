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

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

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

namespace detail {

template <typename ScalarT>
inline void append_scalar(std::string& out, const ScalarT& value)
{
  out.append(reinterpret_cast<const char*>(&value), sizeof(ScalarT));
}

template <typename ScalarT>
inline void read_scalar(const char*& cursor, const char* end, ScalarT& value)
{
  if (static_cast<std::size_t>(end - cursor) < sizeof(ScalarT)) {
    throw std::runtime_error("SPORF serialization payload truncated while reading scalar");
  }
  std::memcpy(&value, cursor, sizeof(ScalarT));
  cursor += sizeof(ScalarT);
}

template <typename ScalarT>
inline void append_vector(std::string& out, const std::vector<ScalarT>& values)
{
  std::uint64_t size = static_cast<std::uint64_t>(values.size());
  append_scalar(out, size);
  if (!values.empty()) {
    out.append(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(ScalarT));
  }
}

template <typename ScalarT>
inline void read_vector(const char*& cursor, const char* end, std::vector<ScalarT>& values)
{
  std::uint64_t size = 0;
  read_scalar(cursor, end, size);
  if (size > (static_cast<std::uint64_t>(end - cursor) / sizeof(ScalarT))) {
    throw std::runtime_error("SPORF serialization payload truncated while reading vector");
  }
  values.resize(static_cast<std::size_t>(size));
  if (size > 0) {
    std::memcpy(values.data(), cursor, static_cast<std::size_t>(size) * sizeof(ScalarT));
    cursor += static_cast<std::size_t>(size) * sizeof(ScalarT);
  }
}

template <typename T, typename L>
inline std::string serialize_sporf_forest_impl(const SPORFMetaData<T, L>* forest)
{
  if (forest == nullptr) { throw std::runtime_error("Cannot serialize null SPORF forest"); }

  std::string payload;
  constexpr std::uint32_t k_magic = 0x53504F52;  // "SPOR"
  constexpr std::uint32_t k_version = 1;
  append_scalar(payload, k_magic);
  append_scalar(payload, k_version);

  std::uint64_t n_trees = static_cast<std::uint64_t>(forest->trees.size());
  append_scalar(payload, n_trees);
  for (const auto& tree_ptr : forest->trees) {
    if (!tree_ptr) { throw std::runtime_error("Cannot serialize null SPORF tree"); }
    const auto& tree = *tree_ptr;
    append_scalar(payload, tree.treeid);
    append_scalar(payload, tree.depth_counter);
    append_scalar(payload, tree.leaf_counter);
    append_scalar(payload, tree.train_time);
    append_scalar(payload, tree.num_outputs);
    append_vector(payload, tree.vector_leaf);

    std::uint64_t n_nodes = static_cast<std::uint64_t>(tree.sparsetree.size());
    append_scalar(payload, n_nodes);
    for (const auto& node : tree.sparsetree) {
      std::uint8_t is_leaf = node.IsLeaf() ? 1 : 0;
      append_scalar(payload, is_leaf);
      auto colid = node.ColumnId();
      auto quesval = node.QueryValue();
      auto best_metric = node.BestMetric();
      auto left_child_id = static_cast<std::int64_t>(node.LeftChildId());
      auto instance_count = node.InstanceCount();
      append_scalar(payload, colid);
      append_scalar(payload, quesval);
      append_scalar(payload, best_metric);
      append_scalar(payload, left_child_id);
      append_scalar(payload, instance_count);
    }

    std::uint64_t n_proj_vectors = static_cast<std::uint64_t>(tree.projection_vectors.size());
    append_scalar(payload, n_proj_vectors);
    for (const auto& proj : tree.projection_vectors) {
      append_scalar(payload, proj.n_proj_components);
      append_scalar(payload, proj.indptr_offset);
      append_scalar(payload, proj.indices_offset);
      append_scalar(payload, proj.coeffs_offset);
    }

    append_vector(payload, tree.projection_indptr_storage);
    append_vector(payload, tree.projection_indices_storage);
    append_vector(payload, tree.projection_coeffs_storage);
  }

  return payload;
}

template <typename T, typename L>
inline void deserialize_sporf_forest_impl(SPORFMetaData<T, L>* forest, const std::string& payload)
{
  if (forest == nullptr) { throw std::runtime_error("Cannot deserialize into null SPORF forest"); }

  const char* cursor = payload.data();
  const char* end = cursor + payload.size();

  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  read_scalar(cursor, end, magic);
  read_scalar(cursor, end, version);
  if (magic != 0x53504F52) { throw std::runtime_error("Invalid SPORF serialization payload"); }
  if (version != 1) { throw std::runtime_error("Unsupported SPORF serialization payload version"); }

  forest->trees.clear();
  std::uint64_t n_trees = 0;
  read_scalar(cursor, end, n_trees);
  forest->trees.reserve(static_cast<std::size_t>(n_trees));

  for (std::uint64_t tree_idx = 0; tree_idx < n_trees; ++tree_idx) {
    auto tree = std::make_shared<DT::ObliqueTreeMetaDataNode<T, L>>();
    read_scalar(cursor, end, tree->treeid);
    read_scalar(cursor, end, tree->depth_counter);
    read_scalar(cursor, end, tree->leaf_counter);
    read_scalar(cursor, end, tree->train_time);
    read_scalar(cursor, end, tree->num_outputs);
    read_vector(cursor, end, tree->vector_leaf);

    std::uint64_t n_nodes = 0;
    read_scalar(cursor, end, n_nodes);
    tree->sparsetree.clear();
    tree->sparsetree.reserve(static_cast<std::size_t>(n_nodes));
    for (std::uint64_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
      std::uint8_t is_leaf = 0;
      int colid = 0;
      T quesval = T(0);
      T best_metric = T(0);
      std::int64_t left_child_id = -1;
      int instance_count = 0;
      read_scalar(cursor, end, is_leaf);
      read_scalar(cursor, end, colid);
      read_scalar(cursor, end, quesval);
      read_scalar(cursor, end, best_metric);
      read_scalar(cursor, end, left_child_id);
      read_scalar(cursor, end, instance_count);

      if (is_leaf != 0) {
        tree->sparsetree.push_back(SparseTreeNode<T, L>::CreateLeafNode(instance_count));
      } else {
        tree->sparsetree.push_back(
          SparseTreeNode<T, L>::CreateSplitNode(
            colid, quesval, best_metric, left_child_id, instance_count));
      }
    }

    std::uint64_t n_proj_vectors = 0;
    read_scalar(cursor, end, n_proj_vectors);
    tree->projection_vectors.resize(static_cast<std::size_t>(n_proj_vectors));
    for (auto& proj : tree->projection_vectors) {
      read_scalar(cursor, end, proj.n_proj_components);
      read_scalar(cursor, end, proj.indptr_offset);
      read_scalar(cursor, end, proj.indices_offset);
      read_scalar(cursor, end, proj.coeffs_offset);
    }

    read_vector(cursor, end, tree->projection_indptr_storage);
    read_vector(cursor, end, tree->projection_indices_storage);
    read_vector(cursor, end, tree->projection_coeffs_storage);
    forest->trees.push_back(std::move(tree));
  }

  if (cursor != end) {
    throw std::runtime_error("SPORF serialization payload had trailing bytes");
  }
}

}  // namespace detail


// ----------------------------- Classification ----------------------------------- //

typedef SPORFMetaData<float, int> SPORFClassifierF;
typedef SPORFMetaData<double, int> SPORFClassifierD;

void get_label_metadata(const raft::handle_t& user_handle,
                        const int* labels,
                        int n_rows,
                        int* n_unique_labels,
                        bool* is_dense_zero_based);

void get_unique_labels(const raft::handle_t& user_handle,
                       const int* labels,
                       int n_rows,
                       int* unique_labels_out,
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

inline std::string serialize(const SPORFClassifierF* forest)
{
  return detail::serialize_sporf_forest_impl(forest);
}

inline std::string serialize(const SPORFClassifierD* forest)
{
  return detail::serialize_sporf_forest_impl(forest);
}

inline void deserialize(SPORFClassifierF* forest, const std::string& payload)
{
  detail::deserialize_sporf_forest_impl(forest, payload);
}

inline void deserialize(SPORFClassifierD* forest, const std::string& payload)
{
  detail::deserialize_sporf_forest_impl(forest, payload);
}

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
