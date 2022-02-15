#include <cstdio>

__device__ int dev;

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

template <typename F>
__global__
void cuda_generic_kernel(F f)
{
  dev=234235;
  //  f();
}

inline void thread_barrier_internal()
{
#ifdef COMPILING_FOR_DEVICE
  cudaDeviceSynchronize();
#endif
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
  
  cudaSetDevice(0);
}

int main()
{
  IncapsInt value{2354};
  RefToIncapsulatedInt ref(value);
  
  const dim3 block_dimension(128);
  const dim3 grid_dimension(1);
  
  init_cuda();
  
  cuda_generic_kernel<<<grid_dimension,block_dimension>>>([=] __device__ ()
  {
    dev=2343;
  });
  
  int host;
  
  cudaMemcpy(&host,&dev,sizeof(int),cudaMemcpyDeviceToHost);
  
  printf("%d\n",host);
  
  return 0;
}
