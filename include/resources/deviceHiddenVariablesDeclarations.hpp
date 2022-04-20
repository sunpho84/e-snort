#ifndef _DEVICEHIDDENVARIABLESDECLARATIONS_HPP
#define _DEVICEHIDDENVARIABLESDECLARATIONS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file deviceHiddenVariablesDeclarations.hpp

#include <metaprogramming/hiddenVariableProvider.hpp>

namespace grill::device
{
  DEFINE_OR_DECLARE_HIDDEN_VARIABLE_WITH_CONST_ACCESSOR(int,nDevices,(0));
}

#include <metaprogramming/hiddenVariableProviderTail.hpp>

#endif
