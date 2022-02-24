#ifndef _STACKEDVARIABLE_HPP
#define _STACKEDVARIABLE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file stackedVariable.hpp

#include <tensor/dynamicVariable.hpp>

namespace esnort
{
#define THIS					\
    StackedVariable<T>
  
  /// Holds a variable on Host stack
  template <typename T>
  struct StackedVariable :
    Expr<THIS>,
    VariableExecSpaceChanger<THIS,T,ExecutionSpace::HOST>
  {
    using Expr<THIS>::operator=;
    
#undef THIS
    
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
    
    T value;
    
    const T* getPtr() const
    {
      return &value;
    }
    
    T* getPtr()
    {
      return &value;
    }
    
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
