/*
 * Trampoline DLL for __cu_demangle from NVIDIA's libcufilt.
 *
 * On Windows, cufilt.lib is compiled with /MT (static CRT) while Python
 * extensions require /MD (dynamic CRT).  Linking them directly causes
 * LNK2038.  This trampoline is built as a standalone DLL with /MT so it
 * can safely link cufilt.lib.  It exports cupy_cu_demangle (passthrough)
 * and cupy_free (frees using the same CRT that allocated the buffer),
 * avoiding any cross-heap issues.
 *
 * On Linux this file is not used; libcufilt.a is linked directly into
 * the function extension.
 */

#include "nv_decode.h"
#include <stdlib.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

EXPORT char* cupy_cu_demangle(
        const char* id, char* output_buffer,
        size_t* length, int* status) {
    return __cu_demangle(id, output_buffer, length, status);
}

EXPORT void cupy_free(char* ptr) {
    free(ptr);
}

#ifdef __cplusplus
}
#endif
