#ifndef _EXECUTION_SPACE_HPP
#define _EXECUTION_SPACE_HPP

namespace esnort
{
  enum ExecutionSpace{EXEC_HOST,EXEC_DEVICE,EXEC_WHERE_LIVING,EXEC_UNDEFINED};
  enum ExecutionSpaceChangeCost{EXEC_SPACE_CHANGE_COSTS_NOTHING,EXEC_SPACE_CHANGE_COSTS_LITTLE,EXEC_SPACE_CHANGE_COSTS_ALOT};
  
//   constexpr ExecutionSpace currentExecSpace()
//   {
//     return
// #ifdef __CUDA_ARCH__
//       EXEC_DEVICE
// #else
//       EXEC_HOST
// #endif
//       ;
//   }
  
  template <typename T>
  constexpr ExecutionSpace execSpace()
  {
    constexpr ExecutionSpace tmp=
      T::_execSpace();
    
    if constexpr(tmp==EXEC_WHERE_LIVING)
      return
#if COMPILING_FOR_DEVICE
      EXEC_DEVICE
#else
      EXEC_HOST
#endif
      ;
    
    return tmp;
  }
  
  enum ExecutionSpaceAssignmentType{ASSIGN_MATCHING_SPACE,ASSIGN_MISMATCHING_SPACE};
  
  enum WhichSideToChange{CHANGE_EXEC_SPACE_LHS_SIDE,CHANGE_EXEC_SPACE_RHS_SIDE};
}

#endif
