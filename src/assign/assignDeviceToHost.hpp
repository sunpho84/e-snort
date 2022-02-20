#ifndef _ASSIGNDEVICETOHOST_HPP
#define _ASSIGNDEVICETOHOST_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file assign/assignDeviceToHost.hpp
///
/// \brief Assign from device to a host expression

#include <assign/assignBase.hpp>

namespace esnort
{
  /// Structure to decide the correct path of assignement
  ///
  /// Device to host assignment, case in which we decided to change the rhs
  /// This requires just to transfer the rhs to the host, than call again the assignment
  template <>
  struct Assign<ExecutionSpace::HOST,ExecutionSpace::DEVICE,WhichSideToChange::RHS>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs)
    {
#warning add some verbosity switch
      printf("Copying to host the rhs\n");
      
      /// Version of the rhs located on the host
      const auto hostRhs=
	rhs.template changeExecSpaceTo<ExecutionSpace::HOST>();
      
      lhs()=hostRhs();
    }
  };
}

#endif
