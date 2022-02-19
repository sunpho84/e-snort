#ifndef _ASSIGNHOSTTODEVICE_HPP
#define _ASSIGNHOSTTODEVICE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <assign/assignBase.hpp>

namespace esnort
{
  template <>
  struct Assign<ExecutionSpace::DEVICE,ExecutionSpace::HOST,WhichSideToChange::RHS>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs)
    {
      printf("Copying to device the rhs\n");
      
      const auto deviceRhs=
	rhs.template changeExecSpaceTo<ExecutionSpace::DEVICE>();
      
      lhs=deviceRhs;
    }
  };
}

#endif
