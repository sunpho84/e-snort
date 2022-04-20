#ifndef _DEVICE_HPP
#define _DEVICE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file device.hpp

#include <cstddef>
#include <cstring>

#include <debug/crash.hpp>
#include <expr/assign/executionSpace.hpp>
#include <ios/scopeFormatter.hpp>
#include <ios/logger.hpp>
#include <metaprogramming/singleInstance.hpp>
#include <resources/deviceHiddenVariablesDeclarations.hpp>

#if ENABLE_DEVICE_CODE
  /// Crash on device error, providing a meaningful error
# define DEVICE_CRASH_ON_ERROR(ARGS...)					\
    device::crashOnError(__LINE__,__FILE__,__PRETTY_FUNCTION__,ARGS)
#endif

namespace grill::device
{
#if ENABLE_DEVICE_CODE
  template <typename F,
	    typename IMin,
	    typename IMax>
  __global__
  void cudaGenericKernel(F f,
			 const IMin min,
			 const IMax max)
  {
    const auto i=
      min+blockIdx.x*blockDim.x+threadIdx.x;
    
    if(i<max)
      f(i);
  }
  
  void crashOnError(const int line,const char* file,const char* function,const cudaError_t rc,const char *format,...);
  
  void memcpy(void* dst,const void* src,size_t count,cudaMemcpyKind kind);
  
#define PROVIDE_MEMCPY(FROM_TO_TO)					\
  INLINE_FUNCTION							\
  void memcpy ## FROM_TO_TO(void* dst,const void* src,size_t count)	\
  {									\
    VERBOSE_LOGGER(3)<<"copying "<<count<<" bytes " #FROM_TO_TO;	\
									\
    memcpy(dst,src,count,cudaMemcpy ## FROM_TO_TO);			\
  }
  
  PROVIDE_MEMCPY(DeviceToDevice);
  PROVIDE_MEMCPY(DeviceToHost);
  PROVIDE_MEMCPY(HostToDevice);
  PROVIDE_MEMCPY(HostToHost);
  
#undef PROVIDE_MEMCPY
  
  void synchronize();
  
  void free(void* ptr);
#endif
  
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
#if ENABLE_DEVICE_CODE
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
    
    VERBOSE_LOGGER(3)<<"at line "<<line<<" of file "<<file<<" launching kernel on loop ["<<min<<","<<max<<") using blocks of size "<<blockDimension.x<<" and grid of size "<<gridDimension.x;
    
    cudaGenericKernel<<<gridDimension,blockDimension>>>(std::forward<F>(f),min,max);
    DEVICE_CRASH_ON_ERROR(cudaPeekAtLastError(),"Spawning the generic kernel");
    
    synchronize();
#endif
  }
  
  template <typename T>
  void malloc(T& ptr,const size_t& sizeInUnit)
  {
#if ENABLE_DEVICE_CODE
    SCOPE_INDENT();
    
    DEVICE_CRASH_ON_ERROR(cudaMalloc(&ptr,sizeInUnit*sizeof(T)),"allocating on device");
    VERBOSE_LOGGER(3)<<"Allocated on device: "<<ptr;
#else
    CRASH<<"Not computed for device";
#endif
  }
  
  void initialize(const int& iDevice);
  
  void finalize();
  
#define DEVICE_LOOP(INDEX,EXT_START,EXT_END,BODY...)			\
  device::launchKernel(__LINE__,__FILE__,EXT_START,EXT_END,[=] DEVICE_ATTRIB (const std::common_type_t<decltype((EXT_END)),decltype((EXT_START))>& INDEX) mutable {BODY})
  
  template <ExecSpace Dest,
	    ExecSpace Src>
  void memcpy(void* dst,const void* src,size_t count)
  {
    static_assert(Dest!=ExecSpace::UNDEFINED,"Cannot copy to undefined execution space");
    static_assert(Src!=ExecSpace::UNDEFINED,"Cannot copy from undefined execution space");
    
#if ENABLE_DEVICE_CODE
    if constexpr(Dest==ExecSpace::DEVICE)
      {
	if constexpr(Src==ExecSpace::DEVICE)
	  device::memcpyDeviceToDevice(dst,src,count);
	else
	  device::memcpyHostToDevice(dst,src,count);
      }
    else
      {
	if constexpr(Src==ExecSpace::DEVICE)
	  device::memcpyDeviceToHost(dst,src,count);
	else
	  device::memcpyHostToHost(dst,src,count);
      }
#else
    ::memcpy(dst,src,count);
#endif
  }
}

#endif
