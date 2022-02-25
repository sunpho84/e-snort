#ifndef _ASSIGNHOSTTODEVICE_HPP
#define _ASSIGNHOSTTODEVICE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file assign/assignHostToDevice.hpp
///
/// \brief Assign from host to a device expression

#include <assign/assignBase.hpp>
#include <ios/logger.hpp>

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
      SCOPE_INDENT(runLog);

#warning add some verbosity switch
      runLog()<<"Copying to device the rhs, is ref: "<<std::is_lvalue_reference_v<Rhs><<", is const: "<<std::is_const_v<std::remove_reference_t<Rhs>>;
      
      /// Version of the rhs located on the device
      const auto deviceRhs=
	rhs.template changeExecSpaceTo<ExecutionSpace::DEVICE>();
      
      runLog()<<"Copied from host to device: "<<rhs.getPtr()<<" -> "<<deviceRhs.getPtr();
      
      lhs=deviceRhs;
    }
  };
}

#endif
