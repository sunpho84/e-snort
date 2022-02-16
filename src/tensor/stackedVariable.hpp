#ifndef _STACKEDVARIABLE_HPP
#define _STACKEDVARIABLE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <tensor/dynamicVariable.hpp>

namespace esnort
{
  template <typename T>
  struct StackedVariable :
    Expr<StackedVariable<T>>
  {
    using Expr<StackedVariable<T>>::operator=;
    
    static constexpr ExecutionSpace execSpace()
    {
      return currentExecSpace();
    }
    
    static constexpr ExecutionSpaceChangeCost execSpaceChangeCost()
    {
      return EXEC_SPACE_CHANGE_COSTS_LITTLE;
    }
    
    template <ExecutionSpace OthExecSpace>
    decltype(auto) changeExecSpaceTo() const
    {
#ifdef ENABLE_CUDA_CODE
      if constexpr(OthExecSpace!=execSpace())
	{
	  DynamicVariable<T,OthExecSpace> res;
	  cudaMemcpy(res.ptr,
		     &value,
		     sizeof(T),
		     OthExecSpace==EXEC_DEVICE?
		     cudaMemcpyHostToDevice:
		     cudaMemcpyDeviceToHost);
	  return res;
	}
      else
#endif
	return *this;
    }
    
    T value;
    
    const T& operator()() const CUDA_HOST CUDA_DEVICE
    {
      return value;
    }
    
    T& operator()() CUDA_HOST CUDA_DEVICE
    {
      return value;
    }
  };
  
}

#endif
