#ifndef _DYNAMICVARIABLE_HPP
#define _DYNAMICVARIABLE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
#include <ios/logger.hpp>
#include <metaprogramming/inline.hpp>
#include <resources/memory.hpp>
#include <tensor/tensorRef.hpp>
#include <tensor/variableExecSpaceChanger.hpp>

namespace esnort
{
#define THIS					\
  DynamicVariable<T,ExecSpace>
  
  template <typename T,
	    ExecutionSpace ExecSpace>
  struct DynamicVariable :
    Expr<THIS>,
    VariableExecSpaceChanger<THIS,T,ExecSpace>
  {
    using Expr<THIS>::operator=;
    
#undef THIS
    
    DynamicVariable& operator=(const DynamicVariable& oth)
    {
      (*static_cast<Expr<DynamicVariable>*>(this))=
	(*static_cast<const Expr<DynamicVariable>*>(&oth));
      
      return *this;
    }
    
    static constexpr ExecutionSpace execSpace=
      ExecSpace;
    
    static constexpr ExecutionSpaceChangeCost execSpaceChangeCost=
      ExecutionSpaceChangeCost::LITTLE;
    
    T* ptr{nullptr};
    
    const T* getPtr() const
    {
      return ptr;
    }
    
    T* getPtr()
    {
      return ptr;
    }
    
    constexpr INLINE_FUNCTION
    const T& operator()() const HOST_DEVICE_ATTRIB
    {
      return *ptr;
    }
    
    constexpr INLINE_FUNCTION
    T& operator()() HOST_DEVICE_ATTRIB
    {
      return *ptr;
    }
    
    constexpr INLINE_FUNCTION
    DynamicVariable()
    {
      static_assert(std::is_trivially_constructible_v<T>,"not implemented for non-trivially constructible quantities");
      
      ptr=memory::manager<ExecSpace>.template provide<T>(1);
    }
    
    //DynamicVariable(const DynamicVariable&) =delete;
    
    constexpr INLINE_FUNCTION
    DynamicVariable(DynamicVariable&& oth)
    {
      ptr=oth.ptr;
      oth.ptr=nullptr;
    }
    
    constexpr INLINE_FUNCTION
    DynamicVariable(const DynamicVariable& oth) :
      DynamicVariable()
    {
#if ENABLE_DEVICE_CODE
      if(execSpace==ExecutionSpace::DEVICE)
	device::memcpy(ptr,
		     oth.getPtr(),
		     sizeof(T),
		     cudaMemcpyHostToHost);
      else
#endif
	memcpy(ptr,oth.ptr,sizeof(T));
    }
    
    INLINE_FUNCTION
    ~DynamicVariable() HOST_ATTRIB
    {
      if(ptr)
	memory::manager<ExecSpace>.release(ptr);
    }
    
    TensorRef<T,ExecSpace,true> getRef() const
    {
      SCOPE_INDENT();
      
      LOGGER<<"Forging a const ref to "<<ptr;
      
      return ptr;
    }
    
    TensorRef<T,ExecSpace,false> getRef()
    {
      SCOPE_INDENT();
      
      LOGGER<<"Forging a ref to "<<ptr;
      
      return ptr;
    }
  };
}

#endif
