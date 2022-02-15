# e-snort

We need to fresh up our mind.

MemoryManager
---
We put two memory manager, the correct one is automatically chosen

Stacked tensor
------------
Is automatically captured? yes

Dynamic sized tensor
--------------------
Is located on host or device, default storage is device but we can manually take host.
Can we add a method or a wrapper:

```
onHost()
```

```
onDevice()
```

Mirrored data
-------------
We use like lookup table, with proxy to avoid consolidate.
Mirrored data cannot be written directly but an accessor can be asked.

```
getWriteableAcces([](auto& instance)
{

})

getReadableAccess(const auto& instance)
```

Metaprogramming
---
Avoid SFINAE, use tag dispatch

Parallelization
---
Parallelization is issued when assigning or reducing

Execution space
---
Each expression must have an execution space associated.
Question: A stacked tensor understand the execution space? Partially:


```
struct A
{
#ifdef __NVCC__
  #ifndef __CUDA_ARCH__
__host__ static constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
  #else
__device__ static constexpr std::integral_constant<bool,true> isOnDevice() { return {}; }
  #endif
#else
__host__ static constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
__device__ static constexpr std::integral_constant<bool,true> isOnDevice() { return {}; }
#endif
};

void testHost()
{
#ifndef __CUDA_ARCH__
  static_assert(not A::isOnDevice(),"We are issuing A on the host");
#endif
}

__device__ void testDevice()
{
#ifdef __CUDA_ARCH__
  static_assert(A::isOnDevice(),"We are issuing A on the device");
#endif
}
```

Reduction
---
We fill a buffer and iterate reduction. 
Accumulated precision must be adjustable.
The reduction must output a stacked tensor.

Nodes
---
If we take a reference inside a node on the host, then we send it to a
device function, is the reference copied, or do we refer to the
original quantity? 

```
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
    (*dev)=ref.value.value;
  });
  cudaDeviceSynchronize();
  
  decript_cuda_error(cudaMemcpy(&host,dev,sizeof(int),cudaMemcpyDeviceToHost),"Unable to copy");
  
  printf("%d\n",host);
  
  cudaFree(dev);
  
  return 0;
}
```

ARGH! No! `ref.value.value` is not correctly understood, because
ref.value points to the host copy.

No, we still refer to the host quantity.  This means that before
executing an expression tree, we need to prepare it for the correct
execution space
