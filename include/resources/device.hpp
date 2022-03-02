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
#include <resources/deviceHiddenVariablesDeclarations.hpp>

namespace esnort::device
{
#if ENABLE_DEVICE_CODE
  template <typename F,
	    typename IMin,
	    typename IMax>
  __global__
  void cudaGenericKernel(F f,
				const IMin& min,
				const IMax& max)
  {
    const auto i=
      min+blockIdx.x*blockDim.x+threadIdx.x;
    
    if(i<max)
      f(i);
  }
  
  void memcpy(void* dst,const void* src,size_t count,cudaMemcpyKind kind);
  
  void synchronize();
  
  void free(void* ptr);
  
  /// Launch the kernel over the passed range
  template <typename IMin,
	    typename IMax,
	    typename F>
  INLINE_FUNCTION
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
  
  void decryptError(cudaError_t rc,const char *templ,...);
  
  template <typename  T>
  void malloc(T& ptr,const size_t& sizeInUnit)
  {
    SCOPE_INDENT();
    
    decryptError(cudaMalloc(&ptr,sizeInUnit*sizeof(T)),"");
    logger()<<"Allocated on gpu: "<<ptr;
  }
#endif
  
  void initialize(const int& iDevice);
  
#define DEVICE_LOOP(INDEX,EXT_START,EXT_END,BODY...)			\
  device::launchKernel(__LINE__,__FILE__,EXT_START,EXT_END,[=] DEVICE_ATTRIB (const std::common_type_t<decltype((EXT_END)),decltype((EXT_START))>& INDEX) mutable {BODY})
}

#endif
