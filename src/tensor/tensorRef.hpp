#ifndef _TENSORREF_HPP
#define _TENSORREF_HPP

#include "expr/expr.hpp"
namespace esnort
{
  template <typename T,
	    ExecutionSpace ExecSpace>
  struct TensorRef :
    Expr<TensorRef<T,ExecSpace>>
  {
    using Expr<TensorRef<T,ExecSpace>>::operator=;
    
    static constexpr ExecutionSpace execSpace()
    {
      return ExecSpace;
    }
    
    static constexpr ExecutionSpaceChangeCost execSpaceChangeCost()
    {
      return EXEC_SPACE_CHANGE_COSTS_LITTLE;
    }
    
    T* ptr;
    
    const T& operator()() const CUDA_HOST CUDA_DEVICE
    {
      return *ptr;
    }
    
    T& operator()() CUDA_HOST CUDA_DEVICE
    {
      return *ptr;
    }
    
    TensorRef(T* ptr) :
      ptr(ptr)
    {
    }
    
    TensorRef(const TensorRef& oth) :
      ptr(oth.ptr)
    {
    }
    
    const TensorRef& getRef() const
    {
      return *this;
    }
    
    TensorRef& getRef()
    {
      return *this;
    }
  };
}

#endif
