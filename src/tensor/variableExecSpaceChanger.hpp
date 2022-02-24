#ifndef _VARIABLEEXECSPACECHANGER_HPP
#define _VARIABLEEXECSPACECHANGER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <metaprogramming/crtp.hpp>
#include <tensor/tensorRef.hpp>

namespace esnort
{
  template <typename T,
	    ExecutionSpace ExecSpace>
  struct DynamicVariable;
  
  DEFINE_CRTP_INHERITANCE_DISCRIMINER_FOR_TYPE(VariableExecSpaceChanger)
  
  template <typename T,
	    typename Fund,
	    ExecutionSpace ExecSpace>
  struct VariableExecSpaceChanger :
  Crtp<T,crtp::VariableExecSpaceChangerDiscriminer>
    {
#if ENABLE_DEVICE_CODE
      
# define PROVIDE_CHANGE_EXEC_SPACE_TO(CONST_ATTRIB,IS_CONST)		\
      template <ExecutionSpace OthExecSpace>				\
      decltype(auto) changeExecSpaceTo() CONST_ATTRIB			\
      {									\
	if constexpr(OthExecSpace!=ExecSpace)				\
	  {								\
	    printf("Allocating on device to store: %p\n",		\
		   this->crtp().getPtr());				\
									\
	    DynamicVariable<Fund,OthExecSpace> res;			\
									\
	    cudaMemcpy(res.getPtr(),					\
		       this->crtp().getPtr(),				\
		       sizeof(Fund),					\
		       (OthExecSpace==ExecutionSpace::DEVICE)?		\
		       cudaMemcpyHostToDevice:				\
		       cudaMemcpyDeviceToHost);				\
									\
	    return res;							\
	  }								\
	else								\
	  {								\
	    printf("No need to allocate, we just return a reference to %p\n",\
		   this->crtp().getPtr());				\
	    								\
	    return							\
	      TensorRef<Fund,OthExecSpace,IS_CONST>(this->crtp().getPtr()); \
	  }								\
      }
      
#else
      
# define PROVIDE_CHANGE_EXEC_SPACE_TO(CONST_ATTRIB,IS_CONST)		\
      									\
      template <ExecutionSpace OthExecSpace>				\
      TensorRef<Fund,OthExecSpace,IS_CONST> changeExecSpaceTo() CONST_ATTRIB \
      {									\
	return								\
	  this->crtp().getPtr();					\
      }
      
#endif
      
      PROVIDE_CHANGE_EXEC_SPACE_TO(const,true)
      PROVIDE_CHANGE_EXEC_SPACE_TO(/*non const */,false)
    
#undef PROVIDE_CHANGE_EXEC_SPACE_TO
  };
}

#endif
