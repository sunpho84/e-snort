#ifndef _EXECUTION_SPACE_HPP
#define _EXECUTION_SPACE_HPP

namespace esnort
{
  enum class ExecutionSpace{HOST,DEVICE,UNDEFINED};
  
  enum class ExecutionSpaceChangeCost{NOTHING,LITTLE,ALOT};
  
  enum class ExecutionSpaceAssignmentType{MATCHING_SPACE,MISMATCHING_SPACE};
  
  enum class WhichSideToChange{LHS,RHS};
}

#endif
