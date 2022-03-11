#ifndef _EXECUTION_SPACE_HPP
#define _EXECUTION_SPACE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <debug/minimalCrash.hpp>
#include <metaprogramming/inline.hpp>

/// \file expr/executionSpace.hpp
///
/// \brief Declares the execution spaces and the connected properties

namespace esnort
{
  /// Execution space possibilities
  enum class ExecutionSpace{HOST,DEVICE,UNDEFINED};
  
  /// Check that we are accessing device vector only on device code
  template <ExecutionSpace ES>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION
  constexpr void assertCorrectEvaluationStorage()
    {
#if ENABLE_DEVICE_CODE
      
# ifdef __CUDA_ARCH__
      //static_assert(ES==ExecutionSpace::DEVICE,"Cannot exec on host");
      if constexpr(ES==ExecutionSpace::HOST)
       __trap();
# else
       //static_assert(ES==ExecutionSpace::HOST,"Cannot exec on device");
      if constexpr(ES==ExecutionSpace::DEVICE)
	MINIMAL_CRASH("Cannot access device memory from host");
# endif
      
#endif
    }
  
  namespace internal
  {
    /// Convert the execution space name into a string
    template <ExecutionSpace ES>
    constexpr const char* execSpaceName()
    {
      switch(ES)
	{
	case ExecutionSpace::HOST:
	  return "host";
	  break;
	case ExecutionSpace::DEVICE:
	  return "device";
	  break;
	default:
	  return "unspecified";
	}
    }
  }
  
  /// Convert the execution space name into a string
  template <ExecutionSpace ES>
  constexpr const char* execSpaceName=
    internal::execSpaceName<ES>();
}

#endif
