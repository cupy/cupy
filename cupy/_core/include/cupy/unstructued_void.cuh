#ifndef UNSTRUCTURED_VOID_H_
#define UNSTRUCTURED_VOID_H_


namespace cupy {

template<size_t Size>
class UnstructuredVoid {
public:
  char data[Size];

  UnstructuredVoid() = default;

  template<typename OtherType>
  explicit UnstructuredVoid(const OtherType& other) {
    *this = other;
  }

  bool operator==(const UnstructuredVoid<Size>& other) const {
    for (size_t i = 0; i < Size; ++i) {
        if (data[i] != other.data[i]) return false;
    }
    return true;
  }

  bool operator!=(const UnstructuredVoid<Size>& other) const {
    return !(*this == other);
  }

  template<typename OtherType>
  UnstructuredVoid& operator=(const OtherType& other) {
    // NumPy allows anything to avoid really, just by copying the bytes
    // (and filling up with zeros).
    size_t min_length = Size < sizeof(OtherType) ? Size : sizeof(OtherType);
    const char* other_bytes = reinterpret_cast<const char*>(&other);
    memcpy(data, other_bytes, min_length);
    if (Size > min_length) {
      memset(data + min_length, '\0', Size - min_length);
    }
    return *this;
  }
};

} // namespace scupy

#endif // UNSTRUCTURED_VOID_H_
