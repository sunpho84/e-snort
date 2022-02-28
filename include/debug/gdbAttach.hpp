#ifndef _GDB_ATTACH_HPP
#define _GDB_ATTACH_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file gdbAttach.hpp

#include <debug/gdbAttachExternalVariables.hpp>

namespace esnort
{
  /// Implements the trap to debug
  void possiblyWaitToAttachDebugger();
}

#endif
