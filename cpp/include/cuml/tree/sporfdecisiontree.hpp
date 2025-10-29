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

namespace ML {

namespace DT {

typedef enum {
  HISTOGRAM_METHOD_EXACT = 0,
  HISTOGRAM_METHOD_SAMPLED
} HISTOGRAM_METHOD;

struct SPORFDecisionTreeParams : DecisionTreeParams {
  /**
   * Additional parameter(s) required for SPORF
   */
  float density;
  HISTOGRAM_METHOD histogram_method;
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
