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
    VERBOSE_LOGGER(3)<<"Cuda memcpy: "<<kind;
    DEVICE_CRASH_ON_ERROR(cudaMemcpy(dst,src,count,kind),"calling cudaMemcpy");
  }
  
  void synchronize()
  {
    SCOPE_INDENT();
    
    VERBOSE_LOGGER(3)<<"Synchronizing gpu";
    DEVICE_CRASH_ON_ERROR(cudaDeviceSynchronize(),"Synchronizing");
  }
  
  void free(void* ptr)
  {
    SCOPE_INDENT();
    
    VERBOSE_LOGGER(3)<<"Freeing on gpu: "<<ptr;
    DEVICE_CRASH_ON_ERROR(cudaFree(ptr),"");
  }
#endif
  
#if ENABLE_DEVICE_CODE
  void crashOnError(const int line,const char* file,const char* function,const cudaError_t rc,const char *err)
  {
    if(rc!=cudaSuccess)
      {
	minimalCrash(file,line,function,"%s, cuda raised error %d, err: %s",cudaGetErrorString(rc),rc,err);
      }
  }
#endif
  
  void initialize(const int& iDevice)
  {
#if ENABLE_DEVICE_CODE
    
    LOGGER;
    DEVICE_CRASH_ON_ERROR(cudaGetDeviceCount(&_nDevices),"Counting nDevices");
    LOGGER<<"Found "<<nDevices<<" CUDA Enabled devices";
    
    for(int i=0;i<nDevices;i++)
      {
	cudaDeviceProp deviceProp;
	DEVICE_CRASH_ON_ERROR(cudaGetDeviceProperties(&deviceProp,i),"Getting properties for device");
	LOGGER<<" CUDA Enabled device "<<i<<"/"<<nDevices<<": "<<deviceProp.major<<"."<<deviceProp.minor;
      }
    
    DEVICE_CRASH_ON_ERROR(cudaSetDevice(iDevice),"Unable to set the device");
    synchronize();
    
    Duration initDur;
    
    LOGGER<<"CUDA initialized in "<<durationInSec(initDur)<<" s, nDevices: "<<nDevices;
#endif
  }
  
  void finalize()
  {
#if ENABLE_DEVICE_CODE
    
    LOGGER;
    
    synchronize();
    DEVICE_CRASH_ON_ERROR(cudaDeviceReset(),"Unable to reset the device");
    LOGGER<<"CUDA finalized";
#endif
  }
}
