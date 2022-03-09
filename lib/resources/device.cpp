#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file device.cpp

#include <cstdarg>

#define DEFINE_HIDDEN_VARIABLES_ACCESSORS
# include <resources/deviceHiddenVariablesDeclarations.hpp>
#undef DEFINE_HIDDEN_VARIABLES_ACCESSORS

#include <resources/device.hpp>

namespace esnort::device
{
#if ENABLE_DEVICE_CODE
  
  void memcpy(void* dst,const void* src,size_t count,cudaMemcpyKind kind)
  {
    logger()<<"Cuda memcpy: "<<kind;
    decryptError(cudaMemcpy(dst,src,count,kind),"calling cudaMemcpy");
  }
  
  void synchronize()
  {
    SCOPE_INDENT();
    
    logger()<<"Synchronizing gpu";
    decryptError(cudaDeviceSynchronize(),"Synchronizing");
  }
  
  void free(void* ptr)
  {
    SCOPE_INDENT();
    
    logger()<<"Freeing on gpu: "<<ptr;
    decryptError(cudaFree(ptr),"");
  }
#endif
  
#if ENABLE_DEVICE_CODE
  void decryptError(cudaError_t rc,const char *templ,...)
  {
    if(rc!=cudaSuccess)
      {
	va_list ap;
	va_start(ap,templ);
	char mess[1024];
	vsnprintf(mess,1024,templ,ap);
	va_end(ap);
	
	CRASH<<mess<<", cuda raised error: "<<cudaGetErrorString(rc);
      }
  }
#endif
  
  void initialize(const int& iDevice)
  {
#if ENABLE_DEVICE_CODE
    decryptError(cudaGetDeviceCount(&_nDevices),"Counting nDevices");
    
    for(int i=0;i<nDevices;i++)
      {
	cudaDeviceProp deviceProp;
	decryptError(cudaGetDeviceProperties(&deviceProp,i),"Getting properties for device");
	logger()<<" CUDA Enabled device "<<i<<"/"<<nDevices<<": "<<deviceProp.major<<"."<<deviceProp.minor;
      }
    
    decryptError(cudaSetDevice(iDevice),"Unable to set the device");
    synchronize();
    
    Duration initDur;
    
    logger()<<"cuda initialized in "<<durationInSec(initDur)<<" s, nDevices: "<<nDevices;
#endif
  }
}
