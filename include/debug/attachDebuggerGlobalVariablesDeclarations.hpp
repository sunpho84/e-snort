#ifndef _ATTACH_DEBUGGER_GLOBAL_VARIABLES_DECLARATIONS_HPP
#define _ATTACH_DEBUGGER_GLOBAL_VARIABLES_DECLARATIONS_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file attachDebuggerGlobalVariablesDeclarations.hpp

#include <metaprogramming/globalVariableProvider.hpp>

namespace grill
{
  /// Wait to attach debugger
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(bool,waitToAttachDebugger,{true});
}

#endif
