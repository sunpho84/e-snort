#ifndef _EXECUTION_SPACE_HPP
#define _EXECUTION_SPACE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <metaprogramming/inline.hpp>

/// \file expr/executionSpace.hpp
///
/// \brief Declares the execution spaces and the connected properties

namespace esnort
{
  /// Execution space possibilities
  enum class ExecutionSpace{HOST,DEVICE,UNDEFINED};
  
  /// Estimates of the cost to change the execution space
  enum class ExecutionSpaceChangeCost{NOTHING,LITTLE,ALOT};
  
  /// Check that we are accessing device vector only on device code
  template <ExecutionSpace ES>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION
  constexpr void assertCorrectEvaluationStorage()
    {
#if COMPILING_FOR_DEVICE
      static_assert(ES==ExecutionSpace::DEVICE,"Cannot exec on host");
	// __trap();
#else
      static_assert(ES==ExecutionSpace::HOST,"Cannot exec on device");
      // if constexpr(ES==ExecutionSpace::DEVICE)
      // 	CRASH<<"Cannot access device memory from host";
#endif
    }
}

#endif
