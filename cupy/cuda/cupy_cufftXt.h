#include <cufftXt.h>
#include <cstdint>
#include <stdexcept>


extern "C" {

cufftResult set_callback(cufftHandle plan, cufftXtCallbackType cbType, bool cb_load, void** callerInfo=NULL);

}
