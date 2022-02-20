#ifndef _ASSIGNHOSTTODEVICE_HPP
#define _ASSIGNHOSTTODEVICE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file assign/assignHostToDevice.hpp
///
/// \brief Assign from host to a device expression

#include <assign/assignBase.hpp>

namespace esnort
{
  /// Structure to decide the correct path of assignement
  ///
  /// Host to device assignment, case in which we decided to change the rhs
  /// This requires just to transfer the rhs to the device, than call again the assignment
  template <>
  struct Assign<ExecutionSpace::DEVICE,ExecutionSpace::HOST,WhichSideToChange::RHS>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs)
    {
#warning add some verbosity switch
      printf("Copying to device the rhs\n");
      
      /// Version of the rhs located on the device
      const auto deviceRhs=
	rhs.template changeExecSpaceTo<ExecutionSpace::DEVICE>();
      
      lhs=deviceRhs;
    }
  };
}

#endif
