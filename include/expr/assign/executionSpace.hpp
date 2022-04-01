#ifndef _EXECUTION_SPACE_HPP
#define _EXECUTION_SPACE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/assign/executionSpace.hpp
///
/// \brief Declares the execution spaces and the connected properties

#include <debug/minimalCrash.hpp>
#include <metaprogramming/inline.hpp>

namespace esnort
{
  /// Execution space possibilities
  enum class ExecSpace{HOST,DEVICE,UNDEFINED};
  
  /// Check that we are accessing device vector only on device code
  template <ExecSpace ES>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION
  constexpr void assertCorrectEvaluationStorage()
    {
#if ENABLE_DEVICE_CODE
      
# ifdef __CUDA_ARCH__
      //static_assert(ES==ExecSpace::DEVICE,"Cannot exec on host");
      if constexpr(ES==ExecSpace::HOST)
       __trap();
# else
       //static_assert(ES==ExecSpace::HOST,"Cannot exec on device");
      if constexpr(ES==ExecSpace::DEVICE)
	MINIMAL_CRASH("Cannot access device memory from host");
# endif
      
#endif
    }
  
  namespace internal
  {
    /// Convert the execution space name into a string
    template <ExecSpace ES>
    constexpr const char* execSpaceName()
    {
      switch(ES)
	{
	case ExecSpace::HOST:
	  return "host";
	  break;
	case ExecSpace::DEVICE:
	  return "device";
	  break;
	default:
	  return "unspecified";
	}
    }
  }
  
  /// Convert the execution space name into a string
  template <ExecSpace ES>
  constexpr const char* execSpaceName=
    internal::execSpaceName<ES>();
  
  /////////////////////////////////////////////////////////////////
  
  namespace internal
  {
    /// Returns the other execution space
    template <ExecSpace ES>
    constexpr ExecSpace _otherExecSpace()
    {
      switch(ES)
	{
	case ExecSpace::HOST:
	  return ExecSpace::DEVICE;
	  break;
	case ExecSpace::DEVICE:
	  return ExecSpace::HOST;
	  break;
	case ExecSpace::UNDEFINED:
	  return ExecSpace::UNDEFINED;
	  break;
	}
    }
  }
  
  /// Returns the other execution space
  template <ExecSpace ES>
  constexpr ExecSpace otherExecSpace=
    internal::_otherExecSpace<ES>();
  
  /// Common execution space between passed ones
  template <ExecSpace...Es>
  constexpr ExecSpace commonExecSpace()
  {
    constexpr ExecSpace u=ExecSpace::UNDEFINED,h=ExecSpace::HOST,d=ExecSpace::DEVICE;
    constexpr bool atLeastOneHost=((Es==h) or...);
    constexpr bool atLeastOneDevice=((Es==d) or...);
    
    static_assert(not (atLeastOneDevice and atLeastOneHost),"Cannot define common execution spaces among a mixture of host and device execution spaces");
    
    if constexpr(atLeastOneHost)
      return h;
    else
      if constexpr(atLeastOneDevice)
	return d;
      else
	return u;
  }
}

#endif
