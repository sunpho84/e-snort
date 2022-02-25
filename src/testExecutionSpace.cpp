#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <algorithm>

#include <cstdarg>
#include <cstdio>
#include <unistd.h>
#include <iostream>

#include <cuda/cuda.hpp>

#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
#include <resources/valueWithExtreme.hpp>
#include <tensor/stackedVariable.hpp>

#include <esnort.hpp>

using namespace esnort;

int main(int narg,char** arg)
{
  int iDevice;
  if(narg==1)
    iDevice=0;
  else
    sscanf(arg[1],"%d",&iDevice);
  
  runLog()<<"Using device: "<<iDevice;
  
  cuda_init(iDevice);

#if !COMPILING_FOR_DEVICE
  static_assert(StackedVariable<int>::execSpace()==esnort::ExecutionSpace::HOST,"We are issuing A on the host");
#endif
  
  ASM_BOOKMARK_BEGIN("TEST_ASSIGN");
  
  StackedVariable<int> a;
  a()=1;
  
  runLog()<<"going to issue the assignment";
  DynamicVariable<int,ExecutionSpace::DEVICE> b;
  b=a;
  
  DynamicVariable<int,ExecutionSpace::DEVICE> c;
  c=b;
  
  StackedVariable<int> d;
  d=c;
  runLog()<<"Result: "<<a()<<" -> "<<d();
  
  ASM_BOOKMARK_END("TEST_ASSIGN");
  
  return 0;
}
