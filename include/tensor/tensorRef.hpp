#ifndef _TENSORREF_HPP
#define _TENSORREF_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file tensorRef.hpp

#include <expr/expr.hpp>
#include <ios/logger.hpp>

namespace esnort
{
  template <typename T,
	    ExecutionSpace ExecSpace,
	    bool IsConst>
  struct TensorRef  :
    Expr<TensorRef<T,ExecSpace,IsConst>>
  {
    using Expr<TensorRef<T,ExecSpace,IsConst>>::operator=;
    std::conditional_t<IsConst,const T*,T*> ptr;
    
    const T* getPtr() const
    {
      return ptr;
    }
    
    T* getPtr()
    {
      return ptr;
    }
    
    static constexpr ExecutionSpace execSpace()
    {
      return ExecSpace;
    }
    
    static constexpr ExecutionSpaceChangeCost execSpaceChangeCost()
    {
      return ExecutionSpaceChangeCost::LITTLE;
    }
    
    const T& operator()() const CUDA_HOST CUDA_DEVICE
    {
      return *ptr;
    }
    
    std::conditional_t<IsConst,const T&,T&> operator()() CUDA_HOST CUDA_DEVICE
    {
      return *ptr;
    }
    
    TensorRef(std::conditional_t<IsConst,const T*,T*> ptr) CUDA_HOST CUDA_DEVICE:
      ptr(ptr)
    {
      SCOPE_INDENT(runLog);
      
      runLog()<<ptr<<" associated to ref";
    }
    
    // TensorRef(const T* ptr) CUDA_HOST CUDA_DEVICE:
    //   ptr(ptr)
    // {
    // }
    
    // TensorRef(const TensorRef& oth) CUDA_HOST CUDA_DEVICE :
    //   ptr(oth.ptr)
    // {
    // }
    
    const TensorRef& getRef() const
    {
      SCOPE_INDENT(runLog);
      
      runLog()<<"Returning the same const reference to "<<this;
      
      return *this;
    }
    
    TensorRef& getRef()
    {
      SCOPE_INDENT(runLog);
      
      runLog()<<"Returning the same reference to "<<this;
      
      return *this;
    }
  };
}

#endif
