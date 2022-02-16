#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <cstdarg>
#include <cstdio>
#include <unistd.h>

#if ENABLE_CUDA_CODE
# include <cuda/cuda.hpp>
#endif

#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
#include <tensor/stackedVariable.hpp>

using namespace esnort;

#include <cstdio>

template <typename Function>
__global__ void kernel(Function f)
{
  printf("value = %d\n", f());
}

struct Wrapper
{
  int x;
  Wrapper() : x(10)
  {
  };
  
  void doWork()
  {
    // ‘*this’ capture mode tells compiler to make a copy
    // of the object
  };
};

void emin()
{
  Wrapper w1;
  
  w1.doWork();
}

int main()
{
#if ENABLE_CUDA_CODE
  cuda_init();
#endif

  // int x=10;
  // auto lam1 = [=] __device__ { return x+1; };
  // kernel<<<1,1>>>(lam1);
  // cudaDeviceSynchronize();
  // emin();
  
  // return 0;
  
  StackedVariable<int> a;
  a()=1;
  
#if !COMPILING_FOR_DEVICE
  static_assert(StackedVariable<int>::execSpace()==esnort::EXEC_HOST,"We are issuing A on the host");
#endif
  
#if !COMPILING_FOR_DEVICE
      const dim3 block_dimension(1);
      const dim3 grid_dimension(1);
      
      auto f=[// lhs=lhs.getRef(),rhs=rhs.getRef()
	      ] CUDA_DEVICE (const int&)
      {
	return 1;
	//lhs()=rhs();
      };
      
      static_assert(__nv_is_extended_device_lambda_closure_type(decltype(f)),"");
      
      cuda_generic_kernel<<<grid_dimension,block_dimension>>>( 0,1,f);
      
      cudaDeviceSynchronize();
  
  DynamicVariable<int,EXEC_DEVICE> c;
  
  //c=a;
#endif
  
  // StackedVariable<int> b;
  // b=c;
  // auto d=c.changeExecSpaceTo<EXEC_HOST>();
  // c.changeExecSpaceTo<EXEC_HOST>();
  
  return 0;
}
