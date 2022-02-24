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
      printf("Copying to device the rhs, is ref: %d, is const: %d\n",std::is_lvalue_reference_v<Rhs>,std::is_const_v<std::remove_reference_t<Rhs>>);
      
      /// Version of the rhs located on the device
      const auto deviceRhs=
	rhs.template changeExecSpaceTo<ExecutionSpace::DEVICE>();
      
      printf("Copied from host to device: %p -> %p\n",rhs.getPtr(),deviceRhs.getPtr());
      
      lhs=deviceRhs;
    }
  };
}

#endif
