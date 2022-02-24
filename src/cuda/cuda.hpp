#ifndef _CUDA_HPP
#define _CUDA_HPP

#include <cstddef>
#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

namespace esnort
{
  template <typename IMin,
	    typename IMax,
	    typename F>
  CUDA_GLOBAL
  void cuda_generic_kernel(const IMin min,
			   const IMax max,
			   F f)
  {
#if ENABLE_DEVICE_CODE
    const auto i=min+blockIdx.x*blockDim.x+threadIdx.x;
    if(i<max)
      f(i);
#endif
  }
  
#if ENABLE_DEVICE_CODE
  inline void decrypt_cuda_error(cudaError_t rc,const char *templ,...)
  {
    if(rc!=cudaSuccess)
      {
	va_list ap;
	va_start(ap,templ);
	char mess[1024];
	vsnprintf(mess,1024,templ,ap);
	va_end(ap);
	
	fprintf(stderr,"%s, cuda raised error: %s\n",mess,cudaGetErrorString(rc));
	exit(1);
      }
  }
#endif
  
  inline void freeCuda(void* ptr)
  {
#if ENABLE_DEVICE_CODE
    printf("freeing on gpu: %p\n",ptr);
    decrypt_cuda_error(cudaFree(ptr),"");
#endif
  }

  template <typename  T>
  inline void mallocCuda(T& ptr,const size_t& sizeInUnit)
  {
#if ENABLE_DEVICE_CODE
    decrypt_cuda_error(cudaMalloc(&ptr,sizeInUnit*sizeof(T)),"");
    printf("Allocating on gpu: %p\n",ptr);
#endif
  }
  
  inline void cuda_init(const int& iDevice)
  {
#if ENABLE_DEVICE_CODE
    int nDevices;
    if(cudaGetDeviceCount(&nDevices)!=cudaSuccess)
      {
	fprintf(stderr,"no CUDA enabled device\n");
	exit(0);
      }
    
    for(int i=0;i<nDevices;i++)
      {
	cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp,i);
        printf(" CUDA Enabled device %d/%d: %d.%d\n",i,nDevices,deviceProp.major,deviceProp.minor);
      }
    
    decrypt_cuda_error(cudaSetDevice(iDevice),"Unable to set the device");
    decrypt_cuda_error(cudaDeviceSynchronize(),"Unable to synchronize the device");
#endif
  }
}

#endif
