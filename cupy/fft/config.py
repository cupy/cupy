from cupy import _util
from cupy.cuda import cufft  # TODO: move CallbackManager here

# expose cache handles to this module
from cupy.fft._cache import get_plan_cache  # NOQA
from cupy.fft._cache import clear_plan_cache  # NOQA
from cupy.fft._cache import get_plan_cache_size  # NOQA
from cupy.fft._cache import set_plan_cache_size  # NOQA
from cupy.fft._cache import get_plan_cache_max_memsize  # NOQA
from cupy.fft._cache import set_plan_cache_max_memsize  # NOQA
from cupy.fft._cache import show_plan_cache_info  # NOQA


enable_nd_planning = True
use_multi_gpus = False
_devices = None


def set_cufft_gpus(gpus):
    '''Set the GPUs to be used in multi-GPU FFT.

    Args:
        gpus (int or list of int): The number of GPUs or a list of GPUs
            to be used. For the former case, the first ``gpus`` GPUs
            will be used.

    .. warning::
        This API is currently experimental and may be changed in the future
        version.

    .. seealso:: `Multiple GPU cuFFT Transforms`_

    .. _Multiple GPU cuFFT Transforms:
        https://docs.nvidia.com/cuda/cufft/index.html#multiple-GPU-cufft-transforms
    '''
    _util.experimental('cupy.fft.config.set_cufft_gpus')
    global _devices

    if isinstance(gpus, int):
        devs = [i for i in range(gpus)]
    elif isinstance(gpus, list):
        devs = gpus
    else:
        raise ValueError("gpus must be an int or a list of int.")
    if len(devs) <= 1:
        raise ValueError("Must use at least 2 GPUs.")

    # make it hashable
    _devices = tuple(devs)


_callback_load = ''
_callback_store = ''
_callback_mgr = []


# TODO(leofang): turn this into a context manager?
def set_callbacks(cb_load='', cb_store=''):
    """Set load and/or store callbacks.

    Args:
        cb_load (str): A string contains the device kernel for the load
            callback. It must define ``d_loadCallbackPtr``.
        cb_store (str): A string contains the device kernel for the store
            callback. It must define ``d_storeCallbackPtr``.

    .. note::
        An example for a load callback is shown below:

        .. code-block:: python

            code = r'''
            __device__ cufftComplex CB_ConvertInputC(
                void *dataIn,
                size_t offset,
                void *callerInfo,
                void *sharedPtr) {
              // implementation
            }

            __device__ cufftCallbackLoadC d_loadCallbackPtr = CB_ConvertInputC;
            '''
            cp.fft.config.set_callbacks(cb_load=code)

    .. warning::
        Using cuFFT callbacks requires compiling and loading an FFT module as
        well as static linking for each distinct transform or callback, so the
        first invocation would be slow. This is a limitation of cuFFT, so use
        this feature only when the callback-enabled transform can be reused
        to amortize the cost.

    .. warning::
        When a statically-linked callback-enabled plan is generated (likely
        cached), the callbacks set by this function will be reset. To enable
        callbacks for the next transform, call this function again.

    """
    if cb_load:
        global _callback_load
        _callback_load = cb_load
    if cb_store:
        global _callback_store
        _callback_store = cb_store


def get_static_plan(plan_type, fft_type, keys):
    # TODO: move CallbackManager to this module
    global _callback_load, _callback_store, _callback_mgr

    mgr = cufft.CallbackManager(
        (plan_type, keys), cb_load=_callback_load, cb_store=_callback_store)
    if fft_type == cufft.CUFFT_C2C:
        cb_load_type = cufft.CUFFT_CB_LD_COMPLEX if _callback_load else -1
        cb_store_type = cufft.CUFFT_CB_ST_COMPLEX if _callback_store else -1
    elif fft_type == cufft.CUFFT_R2C:
        cb_load_type = cufft.CUFFT_CB_LD_REAL if _callback_load else -1
        cb_store_type = cufft.CUFFT_CB_ST_COMPLEX if _callback_store else -1
    elif fft_type == cufft.CUFFT_C2R:
        cb_load_type = cufft.CUFFT_CB_LD_COMPLEX if _callback_load else -1
        cb_store_type = cufft.CUFFT_CB_ST_REAL if _callback_store else -1
    elif fft_type == cufft.CUFFT_Z2Z:
        cb_load_type = cufft.CUFFT_CB_LD_COMPLEX_DOUBLE if _callback_load else -1
        cb_store_type = cufft.CUFFT_CB_ST_COMPLEX_DOUBLE if _callback_store else -1
    elif fft_type == cufft.CUFFT_D2Z:
        cb_load_type = cufft.CUFFT_CB_LD_REAL_DOUBLE if _callback_load else -1
        cb_store_type = cufft.CUFFT_CB_ST_COMPLEX_DOUBLE if _callback_store else -1
    elif fft_type == cufft.CUFFT_Z2D:
        cb_load_type = cufft.CUFFT_CB_LD_COMPLEX_DOUBLE if _callback_load else -1
        cb_store_type = cufft.CUFFT_CB_ST_REAL_DOUBLE if _callback_store else -1
    else:
        raise ValueError
    mgr.set_callback(cb_load_type, cb_store_type)

    _callback_load = ''
    _callback_store = ''
    _callback_mgr.append(mgr)  # keep the manager alive
    return mgr.plan
