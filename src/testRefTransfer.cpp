#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <cstdarg>
#include <cstdio>
#include <type_traits>

int host;

#if ENABLE_DEVICE_CODE
template <typename F>
__global__
void cuda_generic_kernel(F f)
{
  f();
}

inline void thread_barrier_internal()
{
  cudaDeviceSynchronize();
}

void decript_cuda_error(cudaError_t rc,const char *templ,...)
{
  if(rc!=cudaSuccess)
    {
      va_list ap;
      va_start(ap,templ);
      char mess[1024];
      vsprintf(mess,templ,ap);
      va_end(ap);
      
      fprintf(stderr,"%s, cuda raised error: %s\n",mess,cudaGetErrorString(rc));
      exit(1);
    }
}

void init_cuda()
{
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
  
  decript_cuda_error(cudaSetDevice(0),"Unable to set the device");
}
#endif

struct IncapsInt
{
  int value;
};

struct RefToIncapsulatedInt
{
  const IncapsInt& value;
  
  RefToIncapsulatedInt(const IncapsInt& value) :
    value(value)
  {
  }
};

int main()
{
  IncapsInt value{2354};
  RefToIncapsulatedInt ref(value);
  
#if ENABLE_DEVICE_CODE
  const dim3 block_dimension(128);
  const dim3 grid_dimension(128);
  
  init_cuda();
  
  int* dev;
  cudaMalloc(&dev,sizeof(int));
  cuda_generic_kernel<<<grid_dimension,block_dimension>>>([=] CUDA_DEVICE CUDA_HOST ()
  {
    (*dev)=ref.value.value;
  });
  cudaDeviceSynchronize();
  
  decript_cuda_error(cudaMemcpy(&host,dev,sizeof(int),cudaMemcpyDeviceToHost),"Unable to copy");
  
  printf("%d\n",host);
  
  cudaFree(dev);
#endif
  return 0;
}

struct A
{
#ifdef __NVCC__
  #ifndef __CUDA_ARCH__
CUDA_HOST static constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
  #else
CUDA_DEVICE static constexpr std::integral_constant<bool,true> isOnDevice() { return {}; }
  #endif
#else
# if ENABLE_DEVICE_CODE
  CUDA_HOST static constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
  CUDA_DEVICE static constexpr std::integral_constant<bool,true> isOnDevice() { return {}; }
# else
  static constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
# endif
#endif
};

void testHost()
{
#ifndef __CUDA_ARCH__
  static_assert(not A::isOnDevice(),"We are issuing A on the host");
#endif
}

CUDA_DEVICE void testDevice()
{
#ifdef __CUDA_ARCH__
  static_assert(A::isOnDevice(),"We are issuing A on the device");
#endif
}
