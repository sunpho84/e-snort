#ifndef _DEVICE_HPP
#define _DEVICE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file device.hpp

#include <cstddef>

#include <debug/crash.hpp>
#include <ios/scopeFormatter.hpp>
#include <ios/logger.hpp>
#include <metaprogramming/singleInstance.hpp>

namespace esnort
{
#if ENABLE_DEVICE_CODE
  template <typename F,
	    typename IMin,
	    typename IMax>
  __global__
  static void cudaGenericKernel(F f,
				const IMin& min,
				const IMax& max)
  {
    const auto i=
      min+blockIdx.x*blockDim.x+threadIdx.x;
    
    if(i<max)
      f(i);
  }
#endif
  
  struct Device :
    SingleInstance<Device>
  {
#if ENABLE_DEVICE_CODE
    static INLINE_FUNCTION
    void memcpy(void* dst,const void* src,size_t count,cudaMemcpyKind kind)
    {
      logger()<<"Cuda memcpy: "<<kind;
      decryptError(cudaMemcpy(dst,src,count,kind),"calling cudaMemcpy");
    }
    
    /// Launch the kernel over the passed range
    template <typename IMin,
	      typename IMax,
	      typename F>
    static INLINE_FUNCTION
    void launchKernel(const int line,
		      const char *file,
		      const IMin min,
		      const IMax max,
		      F&& f)
    {
     /// Length of the loop
      const auto length=
        max-min;
      
      const int nCudaThreads=128;
      
      /// Dimension of the block
      const dim3 blockDimension(nCudaThreads);
      
      /// Dimension of the grid
      const dim3 gridDimension((length+blockDimension.x-1)/blockDimension.x);
      
#ifdef __NVCC__
      static_assert(__nv_is_extended_device_lambda_closure_type(std::remove_reference_t<F>),"We need an extended lambda closure");
#endif
      
      logger()<<"at line "<<line<<" of file "<<file<<" launching kernel on loop ["<<min<<","<<max<<") using blocks of size "<<blockDimension.x<<" and grid of size "<<gridDimension.x;
      
      cudaGenericKernel<<<gridDimension,blockDimension>>>(std::forward<F>(f),min,max);
      
      synchronize();
    }
    
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
    
    static void synchronize()
    {
      SCOPE_INDENT();
      
      logger()<<"Synchronizing gpu";
      decryptError(cudaDeviceSynchronize(),"Synchronizing");
    }
    
    static void free(void* ptr)
    {
      SCOPE_INDENT();
      
      logger()<<"Freeing on gpu: "<<ptr;
      decryptError(cudaFree(ptr),"");
    }
    
    template <typename  T>
    static void malloc(T& ptr,const size_t& sizeInUnit)
    {
      SCOPE_INDENT();
      
      decryptError(cudaMalloc(&ptr,sizeInUnit*sizeof(T)),"");
      logger()<<"Allocated on gpu: "<<ptr;
    }
#endif
    
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
	  decryptError(cudaGetDeviceProperties(&deviceProp,i),"Getting properties for device");
	  logger()<<" CUDA Enabled device "<<i<<"/"<<nDevices()<<": "<<deviceProp.major<<"."<<deviceProp.minor;
	}
      
      decryptError(cudaSetDevice(iDevice),"Unable to set the device");
      decryptError(cudaDeviceSynchronize(),"Unable to synchronize the device");
#endif
    }
    
    /// Initialize Cuda
    Device()
    {
#if ENABLE_DEVICE_CODE
      
      /// Takes the time
      Duration initDur;
      
      init(0);
      
      logger()<<"cuda initialized in "<<durationInSec(initDur)<<" s, nDevices: "<<nDevices();
      
#endif
    }
  };
  
#define DEVICE_LOOP(INDEX,EXT_START,EXT_END,BODY...) Device::launchKernel(__LINE__,__FILE__,EXT_START,EXT_END,[=] DEVICE_ATTRIB (const std::common_type_t<decltype((EXT_END)),decltype((EXT_START))>& INDEX) mutable {BODY})
}

#endif
