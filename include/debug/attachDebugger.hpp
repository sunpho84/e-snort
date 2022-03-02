#ifndef _ATTACH_DEBUGGER_HPP
#define _ATTACH_DEBUGGER_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file attachDebugger.hpp

#include <debug/attachDebuggerGlobalVariablesDeclarations.hpp>

namespace esnort
{
  /// Implements the trap to debug
  void possiblyWaitToAttachDebugger();
}

#endif
