#ifndef _TENSORREF_HPP
#define _TENSORREF_HPP

#include "expr/expr.hpp"
namespace esnort
{
  template <typename T,
	    ExecutionSpace ExecSpace,
	    bool IsConst>
  struct TensorRef :
    Expr<TensorRef<T,ExecSpace,IsConst>>
  {
    using Expr<TensorRef<T,ExecSpace,IsConst>>::operator=;
    
    static constexpr ExecutionSpace execSpace()
    {
      return ExecSpace;
    }
    
    static constexpr ExecutionSpaceChangeCost execSpaceChangeCost()
    {
      return EXEC_SPACE_CHANGE_COSTS_LITTLE;
    }
    
    std::conditional_t<IsConst,const T*,T*> ptr;
    
    const T& operator()() const CUDA_HOST CUDA_DEVICE
    {
      return *ptr;
    }
    
    std::conditional_t<IsConst,const T&,T&> operator()() CUDA_HOST CUDA_DEVICE
    {
      return *ptr;
    }
    
    TensorRef(T* ptr) CUDA_HOST CUDA_DEVICE:
      ptr(ptr)
    {
    }
    
    // TensorRef(const T* ptr) CUDA_HOST CUDA_DEVICE:
    //   ptr(ptr)
    // {
    // }
    
    TensorRef(const TensorRef& oth) CUDA_HOST CUDA_DEVICE :
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
