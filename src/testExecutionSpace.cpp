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
  static_assert(StackedVariable<int>::execSpace()==esnort::EXEC_HOST,"We are issuing A on the host");
#endif
  
  StackedVariable<int> a;
  a()=1;
  DynamicVariable<int,EXEC_DEVICE> c;
  
#if 0
  const auto devA=a.changeExecSpaceTo<EXEC_DEVICE>();
  auto rhs=devA.getRef();

  auto lhs=c.getRef();

  auto lam1 = [=] __device__ (const int& i) mutable{ return lhs()=rhs(); };
  cuda_generic_kernel<<<1,1>>>(0,2,lam1);
  cudaDeviceSynchronize();
  
  return 0;
#endif
  
  /////////////////////////////////////////////////////////////////
  c=a;
        //      const auto deviceA=
	// a.template changeExecSpaceTo<EXEC_DEVICE>();
      
	     // Assign<EXEC_DEVICE,EXEC_DEVICE,CHANGE_EXEC_SPACE_RHS_SIDE>::exec(c,deviceA);
#if 0
	     const dim3 block_dimension(1);
      const dim3 grid_dimension(1);
      
      auto devLhs=c.getRef();
      const auto devRhs=deviceA.getRef();
      
      auto f=[=] CUDA_DEVICE (const int& i) mutable
      {
	return devLhs()=devRhs();
      };

      cuda_generic_kernel<<<grid_dimension,block_dimension>>>(0,1,f);
      
      cudaDeviceSynchronize();
#endif
  StackedVariable<int> b;
  // auto lhsc=c.changeExecSpaceTo<EXEC_HOST>();
  // b=lhsc;
  b=c;
  printf("Result: %d -> %d\n",a(),b());
  
  // StackedVariable<int> b;
  // b=c;
  // auto d=c.changeExecSpaceTo<EXEC_HOST>();
  // c.changeExecSpaceTo<EXEC_HOST>();
  
  return 0;
}
