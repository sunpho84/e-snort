#ifndef _GDB_ATTACH_GLOBAL_VARIABLES_DECLARATIONS_HPP
#define _GDB_ATTACH_GLOBAL_VARIABLES_DECLARATIONS_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file gdbAttachGlobalVariablesDeclarations.hpp

#include <metaprogramming/globalVariableProvider.hpp>

namespace esnort
{
  /// Wait to attach gdb
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(bool,waitToAttachDebugger,{true});
}

#endif
