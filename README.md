# grill

A grid with more taste

We need to fresh up our mind.

MemoryManager
---
We put two memory manager, the correct one is automatically chosen

Stacked tensor
------------
Is automatically captured? Well that's tricky, really tricky. If we were to capture it directly, no problem:

```
StackedVariable<int> a;

cuda_generic_kernel<<<grid_dimension,block_dimension>>>(0,1,[a] __device__ (const int&){});

```

this would work, because the copy is taken directly. But suppose we take a refernce to `a`, that would *not* work, because the reference would point to the host object. We need to track this across the expression tree. In other words we need to *compile* the tree before launching the kernel, remapping the stacked variable to a dynamic one on the gpu, see below for next step.


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

To avoid copying when passing by value we need to create a reference. Is this actually needed? Yes because going out of scope, a direct copy of a dynamic tensor would delete the intenral array.


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

Assigning across different execution spaces
---
The dominant execution space is that of lhs since the ultimate goal is
to modify it. The lhs must have a clearly defined execution space.

The rhs might not have a well defined execution space, e.g host or
device, what we care is that
* we can check if a certain execution space can be enforced,
* we can quantify the cost of the change of execution space,
* we can change the execution space to a specific one.

Four scenarios are possible

in two cases, lhs is on the same execution space than rhs.

Opportunity to change the lhs or rhs execution space
---
To decide whether the lhs or rhs should change the execution space, we
compare some estimates at compile time. If the two costs are the same,
we will change the execution space of rhs, since ultimately the lhs
must be stored on its execution space.

Correct execution space
---
The excution space for a stacked tensor can be changed from host to
device by allocating a dynamic tensor on the device and memcopying to
it.

The execution space for a lhs stacked tensor is definitevely the CPU

Availability of other execution spaces
---
How should we treat the device code when not compiling for device?
* We should directly avoid defining the DEVICE execution space?
* Or should we leave all the functionality defined, but shortcircuit
the action of all the device code to alias the non-device versions?

The first case would forbid usign the DEVICE execution space, would
this cause problem in writing generic code?

I believe it's easier to write generic code and short-circuit when
needed.

Expr
---
An expression is static polymorphic for a node in the AST
Fund and Comps are written into the node

When it will be the time, we can generalize sum into algebricCombine

Grill
---
We define grill!

DONE
---
- Move external variabiles to a specific file
- Convert device to namespace
- Enable threads
- Gdb Attacher
- Move Mpi to namespace
- Options from maze
- Enable memory
- Added unroller for
- Tuple
- Components
- Tensor from nissa
- Tensor as reference
- Assign tensor to tensor
- Added BaseTens to allow different kind of tensor copy
- Added ref inside BaseTens
- Comp is able to convert to index automatically
- Remove aliver as it is not starting correctly the devices... why!?
- Enable vectorization
- Put order into the assignment mess: which assigner is acutally called?!
- Bypass c+17 test
- Bypass auto-determination of simd feature
- Fix assignment across different execution spaces
- Move constructor for dynamicTens
- Easier comps
- Binder
- A reference copied to a gpu points inside a binder must get copied by value
- Compl, conj, transp
- Products
- Sums
- Fma
- Trace from nissa
- Algebric sum in place of sum
- Scalar simdify
- Grill
- Fields
- More rigorous lhs vs rhs assignment comps requirement
- Creates a SubLattice class
- Halos
- Neighbors
- Rationalize the sites filling to split parity (oridir)
- Comps now cannot mix anymore, strict cast needed
- Simdified halo
- Shifter added

TODO
---
- closing should infer via sticky mechanism if to store as stacktens, dynamictens, or field
- halo state tracking
- unify simdified and non simdified halo communications
- put together orientation and dir for filling halo, to simplify
- when simdifying should not we reduce nelements? bindcomps is not doing it
- how to pass a generic site comps?
- Putting 1 and I complex
- Reducer from nissa
- MPI io

- Random generator
- Spawning expr assignment in the different case, how will we deal with \*=, -= etc? Do they have any usage, or should we divert to the =a*b case?
- Autotuner
