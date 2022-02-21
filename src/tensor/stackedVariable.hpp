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
      return ExecutionSpace::HOST;
    }
    
    static constexpr ExecutionSpaceChangeCost execSpaceChangeCost()
    {
      return ExecutionSpaceChangeCost::LITTLE;
    }
    
    template <bool B>
    using ExecSpaceChangeDiscriminer=
      std::integral_constant<bool,B>;
    
    template <ExecutionSpace OthExecSpace>
    decltype(auto) changeExecSpaceTo() const
    {
#if ENABLE_DEVICE_CODE
      if constexpr(OthExecSpace!=execSpace())
	{
	  const DynamicVariable<T,OthExecSpace> res;
	  cudaMemcpy(res.ptr,
		     &value,
		     sizeof(T),
		     (OthExecSpace==ExecutionSpace::DEVICE)?
		     cudaMemcpyHostToDevice:
		     cudaMemcpyDeviceToHost);
	  return res;
	}
      else
#endif
	return
	  TensorRef<T,OthExecSpace,true>{&value};
    }
    
    template <ExecutionSpace OthExecSpace>
    decltype(auto) changeExecSpaceTo()
    {
#if ENABLE_DEVICE_CODE
      if constexpr(OthExecSpace!=execSpace())
	{
	  DynamicVariable<T,OthExecSpace> res;
	  cudaMemcpy(res.ptr,
		     &value,
		     sizeof(T),
		     (OthExecSpace==ExecutionSpace::DEVICE)?
		     cudaMemcpyHostToDevice:
		     cudaMemcpyDeviceToHost);
	  return res;
	}
      else
#endif
	return
	  TensorRef<T,OthExecSpace,false>{&value};
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
    
    TensorRef<T,ExecutionSpace::HOST,true> getRef() const
    {
      return &value;
    }
    
    TensorRef<T,ExecutionSpace::HOST,false> getRef()
    {
      return &value;
    }
  };
}

#endif
