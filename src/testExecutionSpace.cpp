#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <cstdarg>
#include <cstdio>
#include <unistd.h>

#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>

#include <tensor/stackedVariable.hpp>

using namespace esnort;

int main()
{
  StackedVariable<int> a;
  a()=1;
  
#if !COMPILING_FOR_DEVICE
  static_assert(StackedVariable<int>::execSpace()==esnort::EXEC_HOST,"We are issuing A on the host");
#endif
  
#if !COMPILING_FOR_DEVICE
  DynamicVariable<int,EXEC_DEVICE> c;
  c=a;
#endif
  
  // StackedVariable<int> b;
  // b=c;
  // auto d=c.changeExecSpaceTo<EXEC_HOST>();
  // c.changeExecSpaceTo<EXEC_HOST>();
  
  return 0;
}
