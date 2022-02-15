#include <cstdarg>
#include <cstdio>

int host;

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
  f();
}

inline void thread_barrier_internal()
{
#ifdef COMPILING_FOR_DEVICE
  cudaDeviceSynchronize();
#endif
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

int main()
{
  IncapsInt value{2354};
  RefToIncapsulatedInt ref(value);
  
  const dim3 block_dimension(128);
  const dim3 grid_dimension(128);
  
  init_cuda();
  
  int* dev;
  cudaMalloc(&dev,sizeof(int));
  cuda_generic_kernel<<<grid_dimension,block_dimension>>>([=] __device__ __host__ ()
  {
    (*dev)=value.value;
  });
  cudaDeviceSynchronize();
  
  decript_cuda_error(cudaMemcpy(&host,dev,sizeof(int),cudaMemcpyDeviceToHost),"Unable to copy");
  
  printf("%d\n",host);
  
  cudaFree(dev);
  
  return 0;
}
