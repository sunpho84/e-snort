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

int main()
{
  cuda_init();

#if !COMPILING_FOR_DEVICE
  static_assert(StackedVariable<int>::execSpace()==esnort::EXEC_HOST,"We are issuing A on the host");
#endif
  
  
  DynamicVariable<int,EXEC_DEVICE> c;
  
  StackedVariable<int> a;
  a()=1;
  
  c=a;
  
  // StackedVariable<int> b;
  // b=c;
  // auto d=c.changeExecSpaceTo<EXEC_HOST>();
  // c.changeExecSpaceTo<EXEC_HOST>();
  
  return 0;
}
