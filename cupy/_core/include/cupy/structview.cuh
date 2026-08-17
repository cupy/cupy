#ifndef STRUCT_VIEW_H_
#define STRUCT_VIEW_H_

#include "cupy/carray.cuh"
#include "cupy/cuda_workaround.h"
#ifndef __CUDACC_RTC__
#include <utility>
#endif


namespace cupy {

// Forward declaration for SFINAE
template<typename, typename...>
class StructView;

// Trait to detect StructView
template<typename T>
struct is_struct_view {
  static constexpr bool value = false;
};

template<typename StorageType, typename... Fields>
struct is_struct_view<StructView<StorageType, Fields...>> {
  static constexpr bool value = true;
};

template<size_t Size, size_t Align>
struct alignas(Align) raw_structview_storage {
  static_assert(Align != 0);
  char _data[Size];
};

// Field descriptor: type and offset only
template<typename T, size_t Offset>
struct Field {
  using type = T;
  static constexpr size_t offset = Offset;
};

// Metaprogramming helper: get field type by index in O(1) depth.
template<size_t I, typename T>
struct Indexed { using type = T; };

template<typename Indices, typename... Ts>
struct IndexedTuple;

template<size_t... Is, typename... Ts>
struct IndexedTuple<std::index_sequence<Is...>, Ts...> : Indexed<Is, Ts>... {};

template<size_t I, typename T>
static Indexed<I, T> type_at_index(Indexed<I, T>);

template<size_t Index, typename... Fields>
struct FieldAtImpl {
  using type = typename decltype(type_at_index<Index>(
      IndexedTuple<std::make_index_sequence<sizeof...(Fields)>, Fields...>{}))::type;
};

// StructView represents a NumPy structured dtype as a C struct.
// Generally, NumPy structured dtypes operate by field index `.at<0>()`
// will fetch the first index and casts/assignment only use fields.
template<typename StorageType, typename... Fields>
class StructView {
  template<size_t Index>
  using FieldAt = typename FieldAtImpl<Index, Fields...>::type;

public:
  using storage_type = StorageType;
  static constexpr size_t size = sizeof(storage_type);
  static constexpr size_t alignment = alignof(storage_type);
  static constexpr size_t field_count = sizeof...(Fields);

  // initialize by initializing the storage, we could initialize
  // individual fields, but assume this would also zero them.
  __device__ StructView() = default;

  __device__ StructView(
      const StructView<storage_type, Fields...>& other) : data_{} {
    assign(other);  // non-trivial to try and honor "holes"
  }

  // Construct from other struct
  template<typename OtherStorageType, typename... OtherFields>
  explicit __device__ StructView(
      const StructView<OtherStorageType, OtherFields...>& other)
      : data_{} {
    static_assert(sizeof...(Fields) == sizeof...(OtherFields), "Field count must match");
    assign(other);
  }

  // Construct from "scalar" by broadcasting.
  template<typename T>
  explicit __device__ StructView(const T& value) : data_{} {
    assign_broadcast(value);
  }

  // Index-based field access
  template<size_t Index>
  __device__ auto& at() {
    using FT = FieldAt<Index>;
    using T = typename FT::type;
    return *reinterpret_cast<T*>(reinterpret_cast<char*>(&data_) + FT::offset);
  }

  template<size_t Index>
  __device__ const auto& at() const {
    using FT = FieldAt<Index>;
    using T = typename FT::type;
    return *reinterpret_cast<const T*>(
        reinterpret_cast<const char*>(&data_) + FT::offset);
  }

  // Cross-type equality (only requires operator== on field types)
  // Note: Like NumPy structured dtypes, only == and != are supported
  template<typename OtherStorageType, typename... OtherFields>
  __device__ bool operator==(
      const StructView<OtherStorageType, OtherFields...>& other) const {
    static_assert(sizeof...(Fields) == sizeof...(OtherFields), "Field count must match");
    return compare_eq(other);
  }

  template<typename OtherStorageType, typename... OtherFields>
  __device__ bool operator!=(
      const StructView<OtherStorageType, OtherFields...>& other) const {
    return !(*this == other);
  }

  // Same as constructor (but more interesting as we actually need to omit holes).
  __device__ StructView& operator=(
      const StructView<storage_type, Fields...>& other) {
    assign(other);
    return *this;
  }

  template<typename OtherStorageType, typename... OtherFields>
  __device__ StructView& operator=(
      const StructView<OtherStorageType, OtherFields...>& other) {
    static_assert(sizeof...(Fields) == sizeof...(OtherFields), "Field count must match");
    assign(other);
    return *this;
  }

  // Broadcast assignment - assign single value to all fields.
  // NOTE(seberg): Can omitting enable_if lead to ambiguity?
  template<typename T>
  __device__ StructView& operator=(const T& value) {
    assign_broadcast(value);
    return *this;
  }

private:
  storage_type data_{};

  template<typename O, typename... OF, size_t... Is>
  __device__ bool compare_eq_impl(
      const StructView<O, OF...>& other, std::index_sequence<Is...>) const {
    bool eq = true;
    ((eq = eq && (at<Is>() == other.template at<Is>())), ...);
    return eq;
  }
  template<typename O, typename... OF>
  __device__ bool compare_eq(const StructView<O, OF...>& other) const {
    return compare_eq_impl(other, std::make_index_sequence<sizeof...(Fields)>{});
  }

  template<typename O, typename... OF, size_t... Is>
  __device__ void assign_impl(
      const StructView<O, OF...>& other, std::index_sequence<Is...>) {
    ((at<Is>() = other.template at<Is>()), ...);
  }
  template<typename O, typename... OF>
  __device__ void assign(const StructView<O, OF...>& other) {
    assign_impl(other, std::make_index_sequence<sizeof...(Fields)>{});
  }

  template<typename T, size_t... Is>
  __device__ void assign_broadcast_impl(
      const T& value, std::index_sequence<Is...>) {
    ((at<Is>() = value), ...);
  }
  template<typename T>
  __device__ void assign_broadcast(const T& value) {
    assign_broadcast_impl(value, std::make_index_sequence<sizeof...(Fields)>{});
  }
};

} // namespace cupy

#endif // STRUCT_VIEW_H_
