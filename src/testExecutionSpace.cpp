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

int main(int narg,char** arg)
{
  int iDevice;
  if(narg==1)
    iDevice=0;
  else
    sscanf(arg[1],"%d",&iDevice);
  
  printf("Using device: %d\n",iDevice);
  
  cuda_init(iDevice);

#if !COMPILING_FOR_DEVICE
  static_assert(StackedVariable<int>::execSpace()==esnort::ExecutionSpace::HOST,"We are issuing A on the host");
#endif
  
  StackedVariable<int> a;
  a()=1;
  
  printf("going to issue the assignment\n");
  DynamicVariable<int,ExecutionSpace::DEVICE> b;
  b=a;
  
  DynamicVariable<int,ExecutionSpace::DEVICE> c;
  c=b;
  
  StackedVariable<int> d;
  d=c;
  printf("Result: %d -> %d\n",a(),d());
  
  return 0;
}
