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
      return EXEC_HOST;//currentExecSpace();
    }
    
    static constexpr ExecutionSpaceChangeCost execSpaceChangeCost()
    {
      return EXEC_SPACE_CHANGE_COSTS_LITTLE;
    }
    
    template <bool B>
    using ExecSpaceChangeDiscriminer=
      std::integral_constant<bool,B>;
    
#if ENABLE_CUDA_CODE
    template <ExecutionSpace OthExecSpace>
    StackedVariable _changeExecSpaceTo(ExecSpaceChangeDiscriminer<false>) const
    {
      return *this;
    }
    
    template <ExecutionSpace OthExecSpace>
    DynamicVariable<T,OthExecSpace> _changeExecSpaceTo(ExecSpaceChangeDiscriminer<true>) const
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
#endif
    
    template <ExecutionSpace OthExecSpace>
    decltype(auto) changeExecSpaceTo() const
    {
      [[ maybe_unused ]]
      static constexpr bool hasToChange=
	OthExecSpace!=execSpace();
      
      return
#if ENABLE_CUDA_CODE
      _changeExecSpaceTo<OthExecSpace>(ExecSpaceChangeDiscriminer<hasToChange>{})
#else
	*this
#endif
	;
      
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
    
    TensorRef<T,execSpace(),true> getRef() const
    {
      return &value;
    }
    
    TensorRef<T,execSpace(),false> getRef()
    {
      return &value;
    }
  };
}

#endif
