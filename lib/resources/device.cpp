#ifdef HAVE_CONFIG_H
# define DEFINE_HIDDEN_VARIABLES_ACCESSORS
# include <config.hpp>
#endif

/// \file device.cpp

#include <resources/deviceGlobalVariablesDeclarations.hpp>

namespace esnort
{
  namespace device
  {
#if ENABLE_DEVICE_CODE
    
    void memcpy(void* dst,const void* src,size_t count,cudaMemcpyKind kind)
    {
      logger()<<"Cuda memcpy: "<<kind;
      decryptError(cudaMemcpy(dst,src,count,kind),"calling cudaMemcpy");
    }
    
    void synchronize();
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
    
    void initialize(const int& iDevice)
    {
#if ENABLE_DEVICE_CODE
      decryptCudaError(cudaGetDeviceCount(&_nDevices),"Counting nDevices");
      
      for(int i=0;i<nDevices();i++)
	{
	  cudaDeviceProp deviceProp;
	  decryptError(cudaGetDeviceProperties(&deviceProp,i),"Getting properties for device");
	  logger()<<" CUDA Enabled device "<<i<<"/"<<nDevices()<<": "<<deviceProp.major<<"."<<deviceProp.minor;
	}
      
      decryptError(cudaSetDevice(iDevice),"Unable to set the device");
      synchronize();
      
      Duration initDur;
      
      logger()<<"cuda initialized in "<<durationInSec(initDur)<<" s, nDevices: "<<nDevices();
#endif
    }
  }
}
