#ifndef INCLUDE_GUARD_ASCEND_CUPY_PROFILER_H
#define INCLUDE_GUARD_ASCEND_CUPY_PROFILER_H

// cudaProfilerStart/cudaProfilerStop implemented on top of the AscendCL
// profiler (aclprof*, see <acl/acl_prof.h>).
//
// CUDA semantics vs. CANN semantics
// ---------------------------------
// CUDA:  cudaProfilerStart() just toggles collection, the tool that was
//        attached (nsys/nvprof) already owns the output directory, so the
//        call is cheap, idempotent and side-effect free.
// CANN:  a session has to be opened explicitly and lives in a state machine:
//
//          aclprofInit(path)                      # choose result dir
//          cfg = aclprofCreateConfig(devs, n,     # devices + metrics + modules
//                  aicoreMetrics, NULL, dataType)
//          aclprofStart(cfg)  ... kernels ...  aclprofStop(cfg)
//          aclprofDestroyConfig(cfg)
//          aclprofFinalize()                      # flush to <path>
//
//        aclprofStart(NULL) (what the old stub attempted) is invalid and
//        returns ACL_ERROR_INVALID_PARAM (100000), which is why it was
//        disabled here; the config object is what was missing.
//
// This header keeps that state machine, so the CUDA-facing API stays a pair of
// `Start`/`Stop` calls (cupy.cuda.runtime.profilerStart /
// cupyx.profiler.profile) and the result lands under CUPY_ASCEND_PROFILER_PATH.
//
// Configuration (environment, read on every session start)
// -------------------------------------------------------
//   CUPY_ASCEND_PROFILER=0|1               master switch (default 1)
//   CUPY_ASCEND_PROFILER_PATH=<dir>        result dir (default ./cupy_ascend_prof)
//   CUPY_ASCEND_PROFILER_DATA_TYPE=<int>   aclprof dataTypeConfig, 0x.. ok
//                                          default ACL_PROF_ACL_API |
//                                          ACL_PROF_TASK_TIME |
//                                          ACL_PROF_AICORE_METRICS |
//                                          ACL_PROF_RUNTIME_API
//   CUPY_ASCEND_PROFILER_AICORE_METRICS=<int>  aclprofAicoreMetrics value
//                                          (default ACL_AICORE_ARITHMETIC_UTILIZATION)
//
// Link note: aclprof* are *not* exported by libascendcl.so on CANN 9.x (it only
// records DT_NEEDED libmsprofiler.so), so a `-lascendcl` link cannot resolve
// them and adding `-lmsprofiler` would make the build depend on a library that
// some CANN releases may not ship. The entry points are therefore resolved
// lazily with dlopen/dlsym; when the profiler library is missing the calls
// degrade to no-ops (the same behaviour CUDA has when no profiler is attached).

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "cupy_ascend_common.h"
#include "acl/acl_prof.h"

namespace cupy_ascend_profiler {

// ---------------------------------------------------------------------------
// aclprof entry points (resolved at runtime)
// ---------------------------------------------------------------------------
typedef aclError (*aclprofInit_t)(const char*, size_t);
typedef aclError (*aclprofFinalize_t)();
typedef aclprofConfig* (*aclprofCreateConfig_t)(
    uint32_t*, uint32_t, aclprofAicoreMetrics, const aclprofAicoreEvents*,
    uint64_t);
typedef aclError (*aclprofDestroyConfig_t)(const aclprofConfig*);
typedef aclError (*aclprofStart_t)(const aclprofConfig*);
typedef aclError (*aclprofStop_t)(const aclprofConfig*);

struct Api {
    aclprofInit_t Init = nullptr;
    aclprofFinalize_t Finalize = nullptr;
    aclprofCreateConfig_t CreateConfig = nullptr;
    aclprofDestroyConfig_t DestroyConfig = nullptr;
    aclprofStart_t Start = nullptr;
    aclprofStop_t Stop = nullptr;
    bool available = false;
};

struct State {
    bool inited = false;            // aclprofInit() done, not finalized
    bool started = false;           // aclprofStart() done, not stopped
    bool disabled = false;          // no usable config -> stop retrying
    aclprofConfig* config = nullptr;  // owned, destroyed on stop
};

inline void* open_profiler_library() {
    // libascendcl.so already pulls libmsprofiler.so in, so the soname lookup
    // normally just returns the already-loaded object.
    void* handle = dlopen("libmsprofiler.so", RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
        return handle;
    }
    const char* home = getenv("ASCEND_HOME_PATH");
    if (home == nullptr || home[0] == '\0') {
        return nullptr;
    }
    const std::string root(home);
    const char* const suffixes[] = {
        "/lib64/libmsprofiler.so",
        "/x86_64-linux/lib64/libmsprofiler.so",
        "/aarch64-linux/lib64/libmsprofiler.so",
    };
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); ++i) {
        const std::string path = root + suffixes[i];
        handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (handle != nullptr) {
            return handle;
        }
    }
    return nullptr;
}

inline Api load_api() {
    Api api;
    void* handle = open_profiler_library();
    if (handle == nullptr) {
        return api;
    }
    api.Init = (aclprofInit_t)dlsym(handle, "aclprofInit");
    api.Finalize = (aclprofFinalize_t)dlsym(handle, "aclprofFinalize");
    api.CreateConfig =
        (aclprofCreateConfig_t)dlsym(handle, "aclprofCreateConfig");
    api.DestroyConfig =
        (aclprofDestroyConfig_t)dlsym(handle, "aclprofDestroyConfig");
    api.Start = (aclprofStart_t)dlsym(handle, "aclprofStart");
    api.Stop = (aclprofStop_t)dlsym(handle, "aclprofStop");
    api.available = api.Init != nullptr && api.Finalize != nullptr &&
        api.CreateConfig != nullptr && api.DestroyConfig != nullptr &&
        api.Start != nullptr && api.Stop != nullptr;
    return api;
}

// One instance per extension module: `inline` gives a single shared object
// across translation units while keeping the header-only style.
inline Api& api() {
    static Api instance = load_api();
    return instance;
}

inline State& state() {
    static State instance;
    return instance;
}

// ---------------------------------------------------------------------------
// Environment helpers
// ---------------------------------------------------------------------------
inline bool enabled() {
    const char* value = getenv("CUPY_ASCEND_PROFILER");
    if (value == nullptr || value[0] == '\0') {
        return true;
    }
    return !(value[0] == '0' || value[0] == 'n' || value[0] == 'N' ||
             value[0] == 'f' || value[0] == 'F');
}

inline std::string result_path() {
    const char* value = getenv("CUPY_ASCEND_PROFILER_PATH");
    if (value != nullptr && value[0] != '\0') {
        return std::string(value);
    }
    return std::string("cupy_ascend_prof");
}

inline uint64_t data_type_config() {
    const char* value = getenv("CUPY_ASCEND_PROFILER_DATA_TYPE");
    if (value != nullptr && value[0] != '\0') {
        return (uint64_t)strtoull(value, nullptr, 0);
    }
    return ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS |
        ACL_PROF_RUNTIME_API;
}

inline aclprofAicoreMetrics aicore_metrics() {
    const char* value = getenv("CUPY_ASCEND_PROFILER_AICORE_METRICS");
    if (value != nullptr && value[0] != '\0') {
        const int metrics = atoi(value);
        if ((metrics >= 0 && metrics <= (int)ACL_AICORE_MEMORY_ACCESS) ||
            metrics == (int)ACL_AICORE_NONE) {
            return (aclprofAicoreMetrics)metrics;
        }
    }
    return ACL_AICORE_ARITHMETIC_UTILIZATION;
}

inline void warn_once(const char* message) {
    static bool warned = false;
    if (!warned) {
        warned = true;
        fprintf(stderr, "[cupy ascend profiler] %s\n", message);
    }
}

// ---------------------------------------------------------------------------
// Session handling
// ---------------------------------------------------------------------------
inline void close_session(Api& api, State& state) {
    if (state.config != nullptr) {
        api.DestroyConfig(state.config);
        state.config = nullptr;
    }
    if (state.inited) {
        api.Finalize();  // flush whatever was collected to the result path
        state.inited = false;
    }
    state.started = false;
}

inline cudaError_t profiler_start() {
    Api& api_ = api();
    if (!api_.available || !enabled()) {
        // Nothing to profile with -> behave like CUDA without a profiler.
        return ACL_SUCCESS;
    }
    State& state_ = state();
    if (state_.started || state_.disabled) {
        return ACL_SUCCESS;  // idempotent, like cudaProfilerStart()
    }

    if (!state_.inited) {
        const std::string path = result_path();
        const aclError ret = api_.Init(path.c_str(), path.size());
        if (ret != ACL_SUCCESS) {
            warn_once("aclprofInit failed, profiling disabled");
            return ret;
        }
        state_.inited = true;
    }

    if (state_.config == nullptr) {
        int device = 0;
        if (aclrtGetDevice(&device) != ACL_SUCCESS) {
            device = 0;  // not yet set up: fall back to the default device id
        }
        uint32_t device_ids[1] = {(uint32_t)device};
        state_.config = api_.CreateConfig(
            device_ids, 1, aicore_metrics(), nullptr, data_type_config());
        if (state_.config == nullptr) {
            // Typically "no usable device"; not an error the caller can fix.
            warn_once("aclprofCreateConfig failed (no usable device?), "
                      "profiling disabled");
            state_.disabled = true;
            close_session(api_, state_);
            return ACL_SUCCESS;
        }
    }

    const aclError ret = api_.Start(state_.config);
    if (ret != ACL_SUCCESS) {
        close_session(api_, state_);
        return ret;
    }
    state_.started = true;
    return ACL_SUCCESS;
}

inline cudaError_t profiler_stop() {
    Api& api_ = api();
    State& state_ = state();
    if (!api_.available || !state_.started) {
        return ACL_SUCCESS;  // idempotent, like cudaProfilerStop()
    }
    aclError ret = api_.Stop(state_.config);
    const aclError finalize_ret = state_.inited ? api_.Finalize() : ACL_SUCCESS;
    if (state_.config != nullptr) {
        api_.DestroyConfig(state_.config);
        state_.config = nullptr;
    }
    state_.inited = false;
    state_.started = false;
    return ret != ACL_SUCCESS ? ret : finalize_ret;
}

}  // namespace cupy_ascend_profiler

extern "C" {

cudaError_t cudaProfilerStart() {
    return cupy_ascend_profiler::profiler_start();
}

cudaError_t cudaProfilerStop() {
    return cupy_ascend_profiler::profiler_stop();
}

}  // extern "C"

#endif  // #ifndef INCLUDE_GUARD_ASCEND_CUPY_PROFILER_H
