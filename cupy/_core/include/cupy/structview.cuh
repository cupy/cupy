#ifndef STRUCT_VIEW_H_
#define STRUCT_VIEW_H_

#include "cupy/carray.cuh"

//#include <cstddef>
//#include <type_traits>
//#include <utility>

namespace sv {

// Hack to replace std::false_type and std::true_type
template<bool Value>
struct my_integral_constant {
    static constexpr bool value = Value;
    using type = my_integral_constant;
    constexpr operator bool() const noexcept { return value; }
};

// Forward declaration for SFINAE
template<typename, size_t, typename...>
class StructView;

// Trait to detect StructView
template<typename T>
struct is_struct_view : my_integral_constant<false> {};

template<typename StructType, size_t Size, typename... Fields>
struct is_struct_view<StructView<StructType, Size, Fields...>> : my_integral_constant<true> {};

template<typename T>
inline constexpr bool is_struct_view_v = is_struct_view<T>::value;

// Field descriptor: type and offset only
template<typename T, size_t Offset>
struct Field {
  using type = T;
  static constexpr size_t offset = Offset;
};

// Metaprogramming helper: get field type by index
template<size_t Index, typename... Fields>
struct FieldAtImpl;
template<typename First, typename... Rest>
struct FieldAtImpl<0, First, Rest...> { using type = First; };
template<size_t Index, typename First, typename... Rest>
struct FieldAtImpl<Index, First, Rest...> { using type = typename FieldAtImpl<Index - 1, Rest...>::type; };

// ============================================================================
// StructView - inherits from any struct (real or dummy)
// ============================================================================
// Provides:
// 1. Direct field access: view.x (via struct inheritance)
// 2. Index-based access: view.at<0>()
// 3. Cross-type comparisons and assignments (by field order)
// ============================================================================
template<typename StructType, size_t Size, typename... Fields>
class StructView : public StructType {
  template<size_t Index>
  using FieldAt = typename FieldAtImpl<Index, Fields...>::type;

  // Compile-time verification
  static_assert(sizeof(StructType) == Size, "StructType size must match Size parameter");

public:
  using struct_type = StructType;
  static constexpr size_t size = Size;
  static constexpr size_t field_count = sizeof...(Fields);

  // Constructors
  StructView() : StructType{} {}

  template<typename... Args>
  explicit StructView(Args&&... args) : StructType{std::forward<Args>(args)...} {}

  StructView(const StructType& s) : StructType(s) {}
  StructView(StructType&& s) : StructType(std::move(s)) {}

  // Access underlying struct
  StructType& as_struct() { return *this; }
  const StructType& as_struct() const { return *this; }

  void* data() { return this; }
  const void* data() const { return this; }

  // Index-based field access
  template<size_t Index>
  auto& at() {
    using FT = FieldAt<Index>;
    using T = typename FT::type;
    return *reinterpret_cast<T*>(reinterpret_cast<char*>(this) + FT::offset);
  }

  template<size_t Index>
  const auto& at() const {
    using FT = FieldAt<Index>;
    using T = typename FT::type;
    return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(this) + FT::offset);
  }

  // Cross-type equality (only requires operator== on field types)
  // Note: Like NumPy structured dtypes, only == and != are supported
  template<typename OtherStruct, size_t OtherSize, typename... OtherFields>
  bool operator==(const StructView<OtherStruct, OtherSize, OtherFields...>& other) const {
    static_assert(sizeof...(Fields) == sizeof...(OtherFields), "Field count must match");
    return compare_eq<0>(other);
  }

  template<typename OtherStruct, size_t OtherSize, typename... OtherFields>
  bool operator!=(const StructView<OtherStruct, OtherSize, OtherFields...>& other) const {
    return !(*this == other);
  }

  // Cross-type assignment (by field order)
  template<typename OtherStruct, size_t OtherSize, typename... OtherFields>
  StructView& operator=(const StructView<OtherStruct, OtherSize, OtherFields...>& other) {
    static_assert(sizeof...(Fields) == sizeof...(OtherFields), "Field count must match");
    assign<0>(other);
    return *this;
  }

  // Broadcast assignment - assign single value to all fields
  // (disabled if T is another StructView to avoid ambiguity)
  // std::enable_if_t<!is_struct_view_v<std::decay_t<T>>>
  template<typename T>
  StructView& operator=(const T& value) {
    assign_broadcast<0>(value);
    return *this;
  }

private:
  // Equality comparison helper (only requires operator==)
  template<size_t Index, typename OtherStruct, size_t OtherSize, typename... OtherFields>
  bool compare_eq(const StructView<OtherStruct, OtherSize, OtherFields...>& other) const {
    if constexpr (Index < sizeof...(Fields)) {
      if (!(at<Index>() == other.template at<Index>())) return false;
      return compare_eq<Index + 1>(other);
    }
    return true;
  }

  // Assignment helper (field-by-field from another StructView)
  template<size_t Index, typename OtherStruct, size_t OtherSize, typename... OtherFields>
  void assign(const StructView<OtherStruct, OtherSize, OtherFields...>& other) {
    if constexpr (Index < sizeof...(Fields)) {
      at<Index>() = other.template at<Index>();
      assign<Index + 1>(other);
    }
  }

  // Broadcast assignment helper (assign same value to all fields)
  template<size_t Index, typename T>
  void assign_broadcast(const T& value) {
    if constexpr (Index < sizeof...(Fields)) {
      at<Index>() = value;
      assign_broadcast<Index + 1>(value);
    }
  }
};

} // namespace sv

#endif // STRUCT_VIEW_H_
