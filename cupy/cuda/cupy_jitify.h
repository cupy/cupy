#ifndef INCLUDE_GUARD_CUPY_JITIFY_H
#define INCLUDE_GUARD_CUPY_JITIFY_H

#define _str_(s) #s
#define _xstr_(s) _str_(s)

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include <cupy/jitify/jitify.hpp>
namespace jitify {
namespace detail {
const char* jitify_ver = _xstr_(CUPY_JITIFY_VERSION_CODE);
}  // namespace detail
}  // namespace jitify

#else

namespace jitify {
namespace detail {

const char* jitify_ver = _xstr_(CUPY_JITIFY_VERSION_CODE);
std::map<std::string, std::string>& get_jitsafe_headers_map();
const int preinclude_jitsafe_headers_count = 0;
const char* preinclude_jitsafe_header_names[] = {};
void load_program(std::string&,
                  std::vector<std::string>&,
                  void*,
                  std::vector<std::string>*,
                  std::map<std::string, std::string>*,
                  std::vector<std::string>*,
                  std::string*);

}  // namespace detail
}  // namespace jitify

#endif

#undef _xstr_
#undef _str_

#endif  // INCLUDE_GUARD_CUPY_JITIFY_H
