#ifndef _CUDA_HPP
#define _CUDA_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file cuda.hpp

#include <cstddef>

#include <debug/crash.hpp>
#include <ios/logger.hpp>
#include <metaprogramming/singleInstance.hpp>

namespace esnort
{
  template <typename IMin,
	    typename IMax,
	    typename F>
  CUDA_GLOBAL
  static void cudaGenericKernel(const IMin min,
			    const IMax max,
			    F f)
  {
#if ENABLE_DEVICE_CODE
    const auto i=min+blockIdx.x*blockDim.x+threadIdx.x;
    if(i<max)
      f(i);
#endif
  }
  
  struct Cuda :
    SingleInstance<Cuda>
  {
    
#if ENABLE_DEVICE_CODE
    static void decryptError(cudaError_t rc,const char *templ,...)
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
    
    static void free(void* ptr)
    {
#if ENABLE_DEVICE_CODE
      printf("freeing on gpu: %p\n",ptr);
      decryptError(cudaFree(ptr),"");
#endif
    }
    
    template <typename  T>
    static void malloc(T& ptr,const size_t& sizeInUnit)
    {
#if ENABLE_DEVICE_CODE
      decryptError(cudaMalloc(&ptr,sizeInUnit*sizeof(T)),"");
      runLog()<<"Allocating on gpu: "<<ptr;
#endif
    }
    
    /// Get the number of devices
    int getNDevices()
      const
    {
      
#if ENABLE_DEVICE_CODE
      
      int nDevices;
      if(cudaGetDeviceCount(&nDevices)!=cudaSuccess)
	CRASH<<"no CUDA enabled device";
      
      return
	nDevices;
      
#else
      
      return
	0;
      
#endif
    }
    
    /// Cached value of number of devices
    int nDevices()
      const
    {
      /// Stored value
      static int _nDevices=
	getNDevices();
      
      return
	_nDevices;
    }
    
    void init(const int& iDevice)
    {
#if ENABLE_DEVICE_CODE
      
      for(int i=0;i<nDevices();i++)
	{
	  cudaDeviceProp deviceProp;
	  cudaGetDeviceProperties(&deviceProp,i);
	  runLog()<<" CUDA Enabled device "<<i<<"/"<<nDevices()<<": "<<deviceProp.major<<"."<<deviceProp.minor;
	}
      
      decryptError(cudaSetDevice(iDevice),"Unable to set the device");
      decryptError(cudaDeviceSynchronize(),"Unable to synchronize the device");
#endif
    }
    
    /// Initialize Cuda
    Cuda()
    {
#if ENABLE_DEVICE_CODE
      
      /// Takes the time
      Duration initDur;
      
      init(0);
      
      runLog()<<"cuda initialized in "<<durationInSec(initDur)<<" s, nDevices: "<<nDevices();
      
#endif
    }
    
  };
}

#endif
