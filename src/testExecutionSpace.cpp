#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <cstdarg>
#include <cstdio>
#include <unistd.h>

#include <cuda/cuda.hpp>

#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
#include <tensor/stackedVariable.hpp>

using namespace esnort;

#include <cstdio>

template <typename Function>
__global__ void kernel(const int min,const int max,Function f)
{
    const auto i=min+blockIdx.x*blockDim.x+threadIdx.x;
  printf("value = %d\n", f(i));
}

int main()
{
  cuda_init();

#if !COMPILING_FOR_DEVICE
  static_assert(StackedVariable<int>::execSpace()==esnort::EXEC_HOST,"We are issuing A on the host");
#endif
  
  StackedVariable<int> a;
  a()=1;
  DynamicVariable<int,EXEC_DEVICE> c;
  
  const auto devA=a.changeExecSpaceTo<EXEC_DEVICE>();
  auto rhs=devA.getRef();

  auto lhs=c.getRef();

  auto lam1 = [=] __device__ (const int& i) mutable{ return lhs()=rhs(); };
  cuda_generic_kernel<<<1,1>>>(0,2,lam1);
  cudaDeviceSynchronize();
  
  // return 0;

  
  /////////////////////////////////////////////////////////////////
  //c=a;
  StackedVariable<int> b;
  auto lhsc=c.changeExecSpaceTo<EXEC_HOST>();
  b=lhsc;
  
  // StackedVariable<int> b;
  // b=c;
  // auto d=c.changeExecSpaceTo<EXEC_HOST>();
  // c.changeExecSpaceTo<EXEC_HOST>();
  
  return 0;
}
