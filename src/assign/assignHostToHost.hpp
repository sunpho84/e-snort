#ifndef _ASSIGNHOSTTOHOST_HPP
#define _ASSIGNHOSTTOHOST_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <assign/assignBase.hpp>

namespace esnort
{
  template <WhichSideToChange W>
  struct Assign<ExecutionSpace::HOST,ExecutionSpace::HOST,W>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs)
    {
      lhs()=rhs();
    }
  };
}

#endif
