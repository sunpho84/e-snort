#ifndef _GDB_ATTACH_EXTERNAL_VARIABLES_HPP
#define _GDB_ATTACH_EXTERNAL_VARIABLES_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file gdbAttachExternalVariables.hpp

namespace esnort
{
  namespace envFlags
  {
    /// Wait to attach gdb
    DEFINE_OR_DECLARE_EXTERNAL_VARIABLE(bool,waitToAttachDebugger,{true});
  }
}

#endif
