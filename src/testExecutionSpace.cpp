#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <algorithm>

#include <cstdarg>
#include <cstdio>
#include <unistd.h>
#include <iostream>

#include <resources/device.hpp>

#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
#include <resources/valueWithExtreme.hpp>
#include <tensor/stackedVariable.hpp>

#include <esnort.hpp>

using namespace esnort;

int main(int narg,char** arg)
{
  SET_FOR_CURRENT_SCOPE(Logger::verbosityLv,1);
  
#if not COMPILING_FOR_DEVICE
  static_assert(StackedVariable<int>::execSpace()==esnort::ExecutionSpace::HOST,"We are issuing A on the host");
#endif
  
  ASM_BOOKMARK_BEGIN("TEST_ASSIGN");
  
  StackedVariable<int> a;
  a()=1;
  
  DynamicVariable<int,ExecutionSpace::DEVICE> b;
  b=a;
  
  DynamicVariable<int,ExecutionSpace::DEVICE> c;
  c=b;
  
  StackedVariable<int> d;
  d=c;
  logger()<<"Result: "<<a()<<" -> "<<d();
  
  ASM_BOOKMARK_END("TEST_ASSIGN");
  
  return 0;
}
