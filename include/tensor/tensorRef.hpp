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
    
    static constexpr ExecutionSpace execSpace=
      ExecSpace;
    
    const T& operator()() const HOST_DEVICE_ATTRIB
    {
      return *ptr;
    }
    
    std::conditional_t<IsConst,const T&,T&> operator()() HOST_DEVICE_ATTRIB
    {
      return *ptr;
    }
    
    TensorRef(std::conditional_t<IsConst,const T*,T*> ptr) :
      ptr(ptr)
    {
      SCOPE_INDENT();
      
      logger()<<ptr<<" associated to ref";
    }
    
    // TensorRef(const T* ptr) HOST_DEVICE_ATTRIB:
    //   ptr(ptr)
    // {
    // }
    
    // TensorRef(const TensorRef& oth) HOST_DEVICE_ATTRIB :
    //   ptr(oth.ptr)
    // {
    // }
    
    const TensorRef& getRef() const
    {
      SCOPE_INDENT();
      
      logger()<<"Returning the same const reference to "<<this;
      
      return *this;
    }
    
    TensorRef& getRef()
    {
      SCOPE_INDENT();
      
      logger()<<"Returning the same reference to "<<this;
      
      return *this;
    }
  };
}

#endif
