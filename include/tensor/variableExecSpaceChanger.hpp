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
	    ExecSpace ES>
  struct DynamicVariable;
  
  template <typename T,
	    typename Fund,
	    ExecSpace ES>
  struct VariableExecSpaceChanger
    {
#if ENABLE_DEVICE_CODE
      
# define PROVIDE_CHANGE_EXEC_SPACE_TO(CONST_ATTRIB,IS_CONST)		\
      template <ExecSpace OthExecSpace>					\
      decltype(auto) changeExecSpaceTo() CONST_ATTRIB			\
      {									\
	if constexpr(OthExecSpace!=ES)					\
	  {								\
	    logger()<<"Allocating on device to store: ",		\
		   DE_CRTPFY(T,this).getPtr();				\
									\
	    DynamicVariable<Fund,OthExecSpace> res;			\
									\
	    device::memcpy(res.getPtr(),				\
			   DE_CRTPFY(T,this).getPtr(),			\
			   sizeof(Fund),				\
			   (OthExecSpace==ExecSpace::DEVICE)?		\
			   cudaMemcpyHostToDevice:			\
			   cudaMemcpyDeviceToHost);			\
									\
	    return res;							\
	  }								\
	else								\
	  {								\
	    logger()<<"No need to allocate, returning a reference to ", \
		   DE_CRTPFY(T,this).getPtr();				\
	    								\
	    return							\
	      TensorRef<Fund,OthExecSpace,IS_CONST>(DE_CRTPFY(T,this).getPtr()); \
	  }								\
      }
      
#else
      
# define PROVIDE_CHANGE_EXEC_SPACE_TO(CONST_ATTRIB,IS_CONST)		\
      									\
      template <ExecSpace OthExecSpace>					\
      TensorRef<Fund,OthExecSpace,IS_CONST> changeExecSpaceTo() CONST_ATTRIB \
      {									\
	return								\
	  DE_CRTPFY(T,this).getPtr();					\
      }
      
#endif
      
      PROVIDE_CHANGE_EXEC_SPACE_TO(const,true)
      PROVIDE_CHANGE_EXEC_SPACE_TO(/*non const */,false)
    
#undef PROVIDE_CHANGE_EXEC_SPACE_TO
  };
}

#endif
