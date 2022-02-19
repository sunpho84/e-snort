#ifndef _ASSIGNDEVICETOHOST_HPP
#define _ASSIGNDEVICETOHOST_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <assign/assignBase.hpp>

namespace esnort
{
  template <>
  struct Assign<ExecutionSpace::HOST,ExecutionSpace::DEVICE,WhichSideToChange::RHS>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs)
    {
      printf("Copying to host the rhs\n");
      
      auto hostRhs=
	rhs.template changeExecSpaceTo<ExecutionSpace::HOST>();
      
      lhs()=hostRhs();
    }
  };
}

#endif
