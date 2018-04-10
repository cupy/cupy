/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#include "tc/core/mapping_options.pb.h"

#include <array>
#include <iostream>
#include <string>
#include <vector>

#include "tc/external/isl.h"

#include "tc/core/flags.h"
#include "tc/core/rtc.h"

/// \file mapping_options.h
/// A set of classes that act as the in-memory interface to control polyhedral
/// scheduling and mapping.  Storage is provided by protocol buffers.
///
/// The interface is based on a concept of "view" classes that provide a
/// different interface to (parts of) underlying protocol buffers.  Each view
/// has a mutable reference to a protocol buffer message, which may be a part of
/// a larger message.  It also provides a set of convenience functions to
/// inspect and modify the underlying message.  All modifications are
/// immediately stored in the protocol buffers, up to the top-level message.
/// Because views only represent a (part of a) message, which they do not own,
/// they cannot be constructed from values.  However, they can be
/// copy-constructed, with a copy referring to the same underlying object, and
/// assigned, with all fields of the underlying object assigned.  Views can be
/// constructed given a protocol buffer message, to which they hold a reference.
/// The caller is responsible for ensuring the actual message lives at least as
/// long as the view.
///
/// All views come with a "materialized" counterpart that owns the underlying
/// message.  They can be constructed from a set of values, from another view or
/// from a protocol buffer.  In the two latter cases, they make a deep copy of
/// the message.  "Materialized" view classes derive from views making it
/// possible to assign a view referring to a part of top-level message from a
/// "materialized" temporary.  For example,
///
///     MappingOptions mo;
///     // Copy of a view refers to the same object.
///     CudaDimView view = mo.block;
///     // Ultimately assigns mo.proto.mutable_block.
///     view = CudaDim(42, 100, 2);
///
/// is equivalent to
///
///     MappingOptions mo;
///     mo.proto.mutable_block()->set_x(42);
///     mo.proto.mutable_block()->set_y(100);
///     mo.proto.mutable_block()->set_z(2);
///
/// References to underlying protocol buffers message objects are exposed
/// publicly.  They can be changed directly, and changes are immediately visible
/// through all views referring to (a part of) the message.  For example,
///
///     MappingOptions mo;
///     mo.proto.mutable_block()->set_x(42);
///     cout << mo.block[0];    // outputs 42;
///
/// "Materialized" views do not expose the message they own, only a modifiable
/// reference through the view interface.
///
/// The top-level interface (MappingOptions) owns and publicly exposes the
/// top-level protocol buffer message along with views to its sub-messages.

namespace tc {

/// Simple template class to wrap getters and by-value setters.  Instances of
/// this class can be implicitly converted to the template parameter type by
/// calling the provided getter function.  They can be assigned from an instance
/// of the template parameter type by calling the setter function provided in
/// the constructor.
///
/// Note that this class does not in any sense extend the lifetime of the
/// accessed object.  Make sure that getters and setters actually change the
/// object, e.g., capture by-reference in lambdas.
template <typename T>
class ValueAccessor {
 public:
  using Setter = std::function<void(T)>;
  using Getter = std::function<T()>;

  ValueAccessor(const Setter& s, const Getter& g) : setter_(s), getter_(g) {}
  ValueAccessor(const ValueAccessor&) = default;

  operator T() const {
    return getter_();
  }

  ValueAccessor& operator=(const T& t) {
    setter_(t);
    return *this;
  }

 private:
  Setter setter_;
  Getter getter_;
};

/// View of a CudaDimProto.
///
/// Provides sequence container-like access to a CudaDimProto, which holds at
/// least one (x) and at most three (x,y,z) values.
class CudaDimView {
 private:
  CudaDimView() = default;

 public:
  /// Construct a view that refers to a protocol buffers message.
  CudaDimView(const CudaDimView&) = default;
  explicit CudaDimView(CudaDimProto& buf) : proto(buf) {}

  /// Number of values held.
  inline size_t size() const;

  /// Return a copy of values as std::vector.
  inline std::vector<uint64_t> extractVector() const;

  /// Return a copy of values as std::array of size 3 padded with defaultDim.
  inline std::array<uint64_t, 3> extractDefaultedArray() const;

  /// Return a modifiable object which replicates assignments back to the
  /// underlying protocol buffers message.
  inline ValueAccessor<uint64_t> operator[](size_t i);

  /// Access the values positionally (x=0, y=1, z=2).
  inline uint64_t operator[](size_t i) const;

  /// Assign the values from another view.
  inline CudaDimView& operator=(const CudaDimView& view);

  /// Compare the values with those from another view.
  inline bool operator==(const CudaDimView& view) const;
  inline bool operator!=(const CudaDimView& view) const;

  /// Conversion to string and output operators.
  std::string toCommaSeparatedString() const;
  friend std::ostream& operator<<(std::ostream& os, const CudaDimView& view);

 public:
  CudaDimProto& proto;

  static const uint64_t defaultDim = 1;
};

/// "Materialized" CudaDimView.
///
/// When constructed from values, ignores trailing defaultDim, e.g.,
///
///   CudaDim(42, defaultDim);
///
/// will only set x, but
///
///   CudaDim(42, defaultDim, 32);
///
/// will x, y and z.
class CudaDim : public CudaDimView {
 public:
  CudaDim() : CudaDimView(ownedProto_) {}
  CudaDim(const CudaDim& cudaDim)
      : CudaDimView(ownedProto_), ownedProto_(cudaDim.proto) {}
  CudaDim(const CudaDimProto& proto)
      : CudaDimView(ownedProto_), ownedProto_(proto) {}
  CudaDim(const CudaDimView& view)
      : CudaDimView(ownedProto_), ownedProto_(view.proto) {}
  inline CudaDim(std::initializer_list<uint64_t> il);
  inline CudaDim(std::vector<uint64_t> il);
  inline CudaDim(
      uint64_t x,
      uint64_t y = CudaDimView::defaultDim,
      uint64_t z = CudaDimView::defaultDim);

  using CudaDimView::operator=;

 private:
  CudaDimProto ownedProto_;
};

/// Specializing CudaDim to differentiate between Block and Grid sizes.
class Block : public CudaDim {
 public:
  Block() = default;
  Block(const CudaDimView& view) : CudaDim(view.proto) {}
  Block(const CudaDimProto& proto) : CudaDim(proto) {}
  Block(std::initializer_list<uint64_t> il) : CudaDim(il) {}
  Block(std::vector<uint64_t> il) : CudaDim(il) {}

  using CudaDimView::operator=;
};

/// Specializing CudaDim to differentiate between Block and Grid sizes.
class Grid : public CudaDim {
 public:
  Grid() = default;
  Grid(const CudaDimView& view) : CudaDim(view.proto) {}
  Grid(const CudaDimProto& proto) : CudaDim(proto) {}
  Grid(std::initializer_list<uint64_t> il) : CudaDim(il) {}
  Grid(std::vector<uint64_t> il) : CudaDim(il) {}

  using CudaDimView::operator=;
};

/// View of a TilingProto.
///
/// Provides sequence container-like access to TilingProto.
class TilingView {
 private:
  TilingView() = default;

 public:
  /// Construct a view that refers to a protocol buffers message.
  TilingView(const TilingView&) = default;
  explicit TilingView(TilingProto& p) : proto(p) {}

  /// Return a copy of values as std::vector.
  inline std::vector<uint64_t> extractVector() const;

  /// Number of values held.
  inline size_t size() const;

  /// Return a modifiable object which replicates assignments back to the
  /// underlying protocol buffers message.
  inline ValueAccessor<uint64_t> operator[](size_t i);

  /// Access the values positionally (x=0, y=1, z=2).
  inline uint64_t operator[](size_t i) const;

  /// Assign the values from another view.
  inline TilingView& operator=(const TilingView& view);

  /// Compare the values with those from another view.
  inline bool operator==(const TilingView& view) const;
  inline bool operator!=(const TilingView& view) const;

  /// Conversion to string and output operators.
  std::string toCommaSeparatedString() const;
  friend std::ostream& operator<<(std::ostream& os, const TilingView& view);

 public:
  TilingProto& proto;
};

/// "Materialized" TilingView.
class Tiling : public TilingView {
 public:
  Tiling() : TilingView(ownedProto_) {}
  Tiling(const Tiling& t) : TilingView(ownedProto_), ownedProto_(t.proto) {}
  Tiling(const TilingProto& proto)
      : TilingView(ownedProto_), ownedProto_(proto) {}
  Tiling(const TilingView& view)
      : TilingView(ownedProto_), ownedProto_(view.proto) {}
  inline Tiling(std::initializer_list<uint64_t> il);
  inline Tiling(const std::vector<uint64_t>& sizes);

 private:
  TilingProto ownedProto_;
};

//// View of a SchedulerOptionsProto.
///
/// Provides isl callbacks based on the options.
class SchedulerOptionsView {
 public:
  /// isl scheduler callback types.
  using MergeCallback = std::function<
      isl_bool(isl_union_map*, isl_union_map*, int, int, int, void*)>;
  using ConstraintsCallback = std::function<isl_basic_set*(
      isl_basic_set*,
      int,
      int,
      isl_id_list*,
      int*,
      int*,
      void*)>;

 private:
  SchedulerOptionsView() = default;

 public:
  /// Construct a view that refers to a protocol buffers message.
  SchedulerOptionsView(const SchedulerOptionsView&) = default;
  SchedulerOptionsView(SchedulerOptionsProto& buf) : proto(buf) {}

  /// Assign the values from another view.
  inline SchedulerOptionsView& operator=(const SchedulerOptionsView&);

  /// Compare the values with those from another view.
  inline bool operator==(const SchedulerOptionsView& view) const;
  inline bool operator!=(const SchedulerOptionsView& view) const;

  /// Output operators.
  friend std::ostream& operator<<(
      std::ostream& os,
      const SchedulerOptionsView& options);

 public:
  SchedulerOptionsProto& proto;
};

/// "Materialized" SchedulerOptionsView.
class SchedulerOptions : public SchedulerOptionsView {
 public:
  SchedulerOptions() : SchedulerOptionsView(ownedProto_) {}
  SchedulerOptions(const SchedulerOptions& options)
      : SchedulerOptionsView(ownedProto_), ownedProto_(options.proto) {}
  explicit SchedulerOptions(const SchedulerOptionsProto& proto)
      : SchedulerOptionsView(ownedProto_), ownedProto_(proto) {}
  explicit SchedulerOptions(const SchedulerOptionsView& view)
      : SchedulerOptionsView(ownedProto_), ownedProto_(view.proto) {}

 private:
  SchedulerOptionsProto ownedProto_;
};

/// Top-level interface to MappingOptionsProto.
///
/// Contains views of the sub-messages (scheduler options, tiling, grid and
/// block sizes).  Provides static constructors for common operator options.
/// Provides fluent (chainable) API for progressively modifying the options.
class MappingOptions {
 private:
  inline MappingOptions();
  static MappingOptions makeUnmappedMappingOptions();

 public:
  /// Construct a deep copy of the options.
  inline MappingOptions(const MappingOptions& options);
  inline explicit MappingOptions(const MappingOptionsProto& buf);

  /// Construct from a serialized protocol buffer message.
  inline explicit MappingOptions(const std::string& str);

  /// Assign from another message.
  MappingOptions& operator=(const MappingOptions&) = default;

  /// Compare with another message.
  inline bool operator==(const MappingOptions& options) const;
  inline bool operator!=(const MappingOptions& options) const;

  /// Get a string with a serialized protocol buffers message.
  inline std::string toProtobufSerializedString() const;

  /**
   * @name Chainable Modifiers
   * See protobuf for documentation on each option.
   * @{
   */
  inline MappingOptions& tile(const std::vector<uint64_t>& sizes);
  inline MappingOptions& tile(std::initializer_list<uint64_t> sizes);
  MappingOptions& tile(const std::string& commaSeparatedSizes);
  inline MappingOptions& tile(const char* commaSeparatedSizes);
  template <typename... Args>
  MappingOptions& tile(Args...);

  inline MappingOptions& mapToThreads(std::initializer_list<uint64_t> threads);
  inline MappingOptions& mapToThreads(
      uint64_t x,
      uint64_t y = CudaDimView::defaultDim,
      uint64_t z = CudaDimView::defaultDim);
  inline MappingOptions& mapToThreads(const std::vector<uint64_t>& threads);
  MappingOptions& mapToThreads(const std::string& commaSeparatedSizes);

  inline MappingOptions& mapToBlocks(std::initializer_list<uint64_t> blocks);
  inline MappingOptions& mapToBlocks(
      uint64_t x,
      uint64_t y = CudaDimView::defaultDim,
      uint64_t z = CudaDimView::defaultDim);
  inline MappingOptions& mapToBlocks(const std::vector<uint64_t>& blocks);
  MappingOptions& mapToBlocks(const std::string& commaSeparatedSizes);

  inline MappingOptions& unroll(uint64_t size);

  inline MappingOptions& useSharedMemory(bool b);
  inline MappingOptions& usePrivateMemory(bool b);
  inline MappingOptions& maxSharedMemory(uint64_t size);
  inline MappingOptions& fixParametersBeforeScheduling(bool b);
  inline MappingOptions& unrollCopyShared(bool b);
  inline MappingOptions& tileImperfectlyNested(bool b);
  inline MappingOptions& matchLibraryCalls(bool b);
  ///@}

  /// Set single fusion strategy.
  ///@{
  inline MappingOptions& scheduleFusionStrategy(FusionStrategy fs);
  inline MappingOptions& scheduleFusionStrategy(const std::string& str);
  ///@}

  /// Set fusion strategy for outer scheduling.
  ///@{
  inline MappingOptions& outerScheduleFusionStrategy(FusionStrategy fs);
  inline MappingOptions& outerScheduleFusionStrategy(const std::string& str);
  inline MappingOptions& outerScheduleAllowSkewing(bool b);
  inline MappingOptions& outerSchedulePositiveOrthant(bool b);
  ///@}

  /// Set fusion strategy for intra-tile scheduling.
  ///@{
  inline MappingOptions& intraTileScheduleFusionStrategy(FusionStrategy fs);
  inline MappingOptions& intraTileScheduleFusionStrategy(
      const std::string& str);
  inline MappingOptions& intraTileScheduleAllowSkewing(bool b);
  inline MappingOptions& intraTileSchedulePositiveOrthant(bool b);
  ///@}

  /// Static constructors for predefined strategies.
  ///@{
  static MappingOptions makeNaiveMappingOptions();
  static MappingOptions makeSingleThreadMappingOptions();
  static MappingOptions makePointwiseMappingOptions();
  static MappingOptions makeMlpMappingOptions();
  static MappingOptions makeConvolutionMappingOptions();
  static MappingOptions makeGroupConvolutionMappingOptions();
  ///@}

  /// Output operator.
  friend std::ostream& operator<<(
      std::ostream& os,
      const MappingOptions& options);

 public:
  MappingOptionsProto proto;

  // Views of sub-messages.
  CudaDimView block;
  CudaDimView grid;
  TilingView tiling;
  SchedulerOptionsView outerScheduleOptions;
  SchedulerOptionsView intraTileScheduleOptions;
};

namespace callbacks {
__isl_give isl_basic_set* AddPositiveCoefficientConstraints(
    __isl_take isl_basic_set* lp,
    int n_param,
    int dim,
    __isl_keep isl_id_list* stmt_ids,
    int* node_n_params,
    int* node_n_dims,
    void*);

isl_bool FuseAllPreserve3Coincident(
    __isl_take isl_union_map* original_schedule,
    __isl_take isl_union_map* updated_schedule,
    int n_updated_coincident,
    int n_original_coincident,
    int is_along_edge,
    void*);

isl_bool FuseAll(
    __isl_take isl_union_map* original_schedule,
    __isl_take isl_union_map* updated_schedule,
    int n_updated_coincident,
    int n_original_coincident,
    int is_along_edge,
    void*);

isl_bool FuseNone(
    __isl_take isl_union_map* original_schedule,
    __isl_take isl_union_map* updated_schedule,
    int n_updated_coincident,
    int n_original_coincident,
    int is_along_edge,
    void*);
} // namespace callbacks

} // namespace tc

#include "tc/core/mapping_options-inl.h"
