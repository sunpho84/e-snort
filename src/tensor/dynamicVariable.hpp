#ifndef _DYNAMICVARIABLE_HPP
#define _DYNAMICVARIABLE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
#include <metaprogramming/inline.hpp>
#include <tensor/tensorRef.hpp>

namespace esnort
{
  template <typename T,
	    ExecutionSpace ExecSpace>
  struct DynamicVariable :
    Expr<DynamicVariable<T,ExecSpace>>
  {
    using Expr<DynamicVariable<T,ExecSpace>>::operator=;
    
    DynamicVariable& operator=(const DynamicVariable& oth)
    {
      (*static_cast<Expr<DynamicVariable>*>(this))=
	(*static_cast<const Expr<DynamicVariable>*>(&oth));
      
      return *this;
    }
    
    static constexpr ExecutionSpace execSpace()
    {
      return ExecSpace;
    }
    
    static constexpr ExecutionSpaceChangeCost execSpaceChangeCost()
    {
      return ExecutionSpaceChangeCost::LITTLE;
    }
    
    T* ptr{nullptr};
    
    constexpr INLINE_FUNCTION
    const T& operator()() const CUDA_HOST CUDA_DEVICE
    {
      return *ptr;
    }
    
    constexpr INLINE_FUNCTION
    T& operator()() CUDA_HOST CUDA_DEVICE
    {
      return *ptr;
    }
    
    constexpr INLINE_FUNCTION
    DynamicVariable()
    {
#if ENABLE_DEVICE_CODE
      if(execSpace()==ExecutionSpace::DEVICE)
	{
	  printf("Allocating on gpu!\n");
	  mallocCuda(ptr,1);
	}
      else
#endif
	ptr=new T;
	  printf("Allocated %p\n",ptr);
    }
    
    //DynamicVariable(const DynamicVariable&) =delete;
    
    constexpr INLINE_FUNCTION
    DynamicVariable(DynamicVariable&& oth)
    {
      printf("A move was made\n");
      ptr=oth.ptr;
      oth.ptr=nullptr;
    }
    
    INLINE_FUNCTION
    ~DynamicVariable() CUDA_HOST
    {
      if(ptr)
	{
#if ENABLE_DEVICE_CODE
	  if(execSpace()==ExecutionSpace::DEVICE)
	    freeCuda(ptr);
	  else
#endif
	    delete ptr;
	}
    }
    
    template <ExecutionSpace OthExecSpace>
    decltype(auto) changeExecSpaceTo() const
    {
#if ENABLE_DEVICE_CODE
      if constexpr(OthExecSpace!=execSpace())
	{
	  const DynamicVariable<T,OthExecSpace> res;
	  cudaMemcpy(res.ptr,
		     ptr,
		     sizeof(T),
		     (OthExecSpace==ExecutionSpace::DEVICE)?
		     cudaMemcpyHostToDevice:
		     cudaMemcpyDeviceToHost);
	  return res;
	}
      else
#endif
	return
	  TensorRef<T,OthExecSpace,true>{ptr};
    }
    
    template <ExecutionSpace OthExecSpace>
    decltype(auto) changeExecSpaceTo()
    {
#if ENABLE_DEVICE_CODE
      if constexpr(OthExecSpace!=execSpace())
	{
	  DynamicVariable<T,OthExecSpace> res;
	  cudaMemcpy(res.ptr,
		     ptr,
		     sizeof(T),
		     (OthExecSpace==ExecutionSpace::DEVICE)?
		     cudaMemcpyHostToDevice:
		     cudaMemcpyDeviceToHost);
	  return res;
	}
      else
#endif
	return
	  TensorRef<T,OthExecSpace,false>{ptr};
    }
    
    TensorRef<T,ExecSpace,true> getRef() const
    {
      printf("Forging a const ref to %p\n",ptr);
      return ptr;
    }
    
    TensorRef<T,ExecSpace,false> getRef()
    {
      printf("Forging a ref to %p\n",ptr);
      return ptr;
    }
  };
}

#endif
