#ifndef _DEVICEGLOBALVARIABLESDECLARATIONS_HPP
#define _DEVICEGLOBALVARIABLESDECLARATIONS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file deviceglobalvariablesdeclarations.hpp

namespace esnort
{
  namespace device
  {
    DEFINE_OR_DECLARE_HIDDEN_VARIABLE_WITH_CONST_ACCESSOR(int,nDevices,(0));
  }
}

#endif
