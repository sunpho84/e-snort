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

namespace grill
{
  /// Execution space possibilities
  enum class ExecSpace : int{HOST_DEVICE,HOST,DEVICE};
  
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
	case ExecSpace::HOST_DEVICE:
	  return "hostDevice";
	  break;
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
	case ExecSpace::HOST_DEVICE:
	  return ExecSpace::HOST_DEVICE;
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
    constexpr int res=((int)Es|...);
    
    static_assert(res!=3,"Cannot mix pure host and pure device exec spaces");
    
    switch(res)
      {
      case 0:
	return ExecSpace::HOST_DEVICE;
	break;
      case 1:
	return ExecSpace::HOST;
	break;
      case 2:
	return ExecSpace::DEVICE;
	break;
      }
  }
  
  /// Default execution space
  inline constexpr ExecSpace defaultExecSpace=
    ExecSpace::
#if ENABLE_DEVICE_CODE
    DEVICE
#else
    HOST
#endif
    ;
}

#endif
