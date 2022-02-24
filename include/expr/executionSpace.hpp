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
  
  /// Estimates of the cost to change ghe execution space
  enum class ExecutionSpaceChangeCost{NOTHING,LITTLE,ALOT};
  
  /// Different possibilities arising when performing an assignement,
  /// comparing the two execution spaces
  enum class ExecutionSpaceAssignmentType{MATCHING_SPACE,MISMATCHING_SPACE};
  
  /// Type to hold the results of the decision on which side of an
  /// assignment will change the execution space
  enum class WhichSideToChange{LHS,RHS};
}

#endif
