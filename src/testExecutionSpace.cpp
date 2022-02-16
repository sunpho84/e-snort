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
  
  DynamicVariable<int,EXEC_DEVICE> c;
  auto aGpu=a.changeExecSpaceTo<EXEC_DEVICE>();
  c.getRef()=aGpu.getRef();
  // StackedVariable<int> b;
  // b=c;
  // auto d=c.changeExecSpaceTo<EXEC_HOST>();
  // c.changeExecSpaceTo<EXEC_HOST>();
  
  return 0;
}
