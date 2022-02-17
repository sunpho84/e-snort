#ifndef _DYNAMICVARIABLE_HPP
#define _DYNAMICVARIABLE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
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
      Expr<DynamicVariable>(*this)=Expr<DynamicVariable>(oth);
      
      return *this;
    }
    
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
    
    DynamicVariable()
    {
#if ENABLE_CUDA_CODE
      if(execSpace()==EXEC_DEVICE)
	mallocCuda(ptr,1);
      else
#endif
	ptr=new T;
    }
    
    //DynamicVariable(const DynamicVariable&) =delete;
    
    DynamicVariable(DynamicVariable&& oth)
    {
      printf("A move was made\n");
      ptr=oth.ptr;
      oth.ptr=nullptr;
    }
    
    ~DynamicVariable() CUDA_HOST
    {
      if(ptr)
	{
#if ENABLE_CUDA_CODE
	  if(execSpace()==EXEC_DEVICE)
	    freeCuda(ptr);
	  else
#endif
	    delete ptr;
	}
    }
    
    template <ExecutionSpace OthExecSpace>
    decltype(auto) changeExecSpaceTo() const
    {
#if ENABLE_CUDA_CODE
      if constexpr(OthExecSpace!=execSpace())
	{
	  DynamicVariable<T,OthExecSpace> res;
	  cudaMemcpy(res.ptr,
		     ptr,
		     sizeof(T),
		     OthExecSpace==EXEC_DEVICE?
		     cudaMemcpyHostToDevice:
		     cudaMemcpyDeviceToHost);
	  return res;
	}
      else
#endif
	return *this;
    }
    
    TensorRef<T,ExecSpace,true> getRef() const
    {
      return ptr;
    }
    
    TensorRef<T,ExecSpace,false> getRef()
    {
      return ptr;
    }
  };
}

#endif
