#ifndef _MEMORYMANAGERGLOBALVARIABLESDECLARATIONS_HPP
#define _MEMORYMANAGERGLOBALVARIABLESDECLARATIONS_HPP

/// \file memoryManagerGlobalVariablesDeclarations.hpp

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

#include <expr/executionSpace.hpp>
#include <metaprogramming/globalVariableProvider.hpp>
#include <resources/memoryManager.hpp>

namespace esnort::memory
{
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(MemoryManager<ExecSpace::HOST>,hostManager,("host"));
#if ENABLE_DEVICE_CODE
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(MemoryManager<ExecSpace::DEVICE>,deviceManager,("device"));
#endif
}

#endif
