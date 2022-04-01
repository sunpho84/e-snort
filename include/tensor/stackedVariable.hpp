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
    Node<THIS>,
    VariableExecSpaceChanger<THIS,T,ExecSpace::HOST>
  {
    using Node<THIS>::operator=;
    
#undef THIS
    
    static constexpr ExecSpace execSpace=
      ExecSpace::HOST;
    
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
    
    const T& operator()() const HOST_DEVICE_ATTRIB
    {
      return value;
    }
    
    T& operator()() HOST_DEVICE_ATTRIB
    {
      return value;
    }
    
    TensorRef<T,ExecSpace::HOST,true> getRef() const
    {
      return &value;
    }
    
    TensorRef<T,ExecSpace::HOST,false> getRef()
    {
      return &value;
    }
  };
}

#endif
