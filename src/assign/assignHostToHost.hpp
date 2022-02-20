#ifndef _ASSIGNHOSTTOHOST_HPP
#define _ASSIGNHOSTTOHOST_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file assign/assignHostToHost.hpp
///
/// \brief Assign within the host itself

#include <assign/assignBase.hpp>

namespace esnort
{
  /// Structure to decide the correct path of assignement
  ///
  /// Host to host assignment. We don't mind which side is expected to
  /// change, since no change is actually involved
  template <WhichSideToChange WeDontMind>
  struct Assign<ExecutionSpace::HOST,ExecutionSpace::HOST,WeDontMind>
  {
    /// Calls the assignement
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
