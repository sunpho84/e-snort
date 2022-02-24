#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <algorithm>

#include <cstdarg>
#include <cstdio>
#include <unistd.h>

#include <cuda/cuda.hpp>

#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
#include <resources/valueWithExtreme.hpp>
#include <tensor/stackedVariable.hpp>

#include <esnort.hpp>

#include <iostream>

using namespace esnort;

void testBacktrace()
{
  using namespace std;
  for(auto& b : getBackTraceList())
    cout<<b<<endl;
}

int main(int narg,char** arg)
{
  testBacktrace();
  return 0;
  
  // ValWithMax<int> e(0);
  // e+=12;
  // e=1;
  // // int val=1;
  // // int max=0;
  
  // // {
  // //   RefProxy ref([&val]()->int&{return val;},[&max,&val](){max=std::max(max,val);});

  // //   ref()=12;
  // // }
  // printf("1? : %d, 12? %d\n",(int)e,e.extreme());
  
  // return 0;
  
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
  
  ASM_BOOKMARK_BEGIN("TEST_ASSIGN");
  
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
  
  ASM_BOOKMARK_END("TEST_ASSIGN");

  return 0;
}
