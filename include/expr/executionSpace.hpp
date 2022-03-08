#ifndef _EXECUTION_SPACE_HPP
#define _EXECUTION_SPACE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/executionSpace.hpp
///
/// \brief Declares the execution spaces and the connected properties

namespace esnort
{
  /// Execution space possibilities
  enum class ExecutionSpace{HOST,DEVICE,UNDEFINED};
  
  /// Estimates of the cost to change the execution space
  enum class ExecutionSpaceChangeCost{NOTHING,LITTLE,ALOT};
}

#endif
