#ifndef _MEMORYGLOBALVARIABLESDECLARATIONS_HPP
#define _MEMORYGLOBALVARIABLESDECLARATIONS_HPP

/// \file memoryGlobalVariablesDeclarations.hpp

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

#include <expr/executionSpace.hpp>
#include <metaprogramming/globalVariableProvider.hpp>
#include <resources/memoryManager.hpp>

namespace esnort::memory
{
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(MemoryManager<ExecutionSpace::HOST>,hostManager,("host"));
#if ENABLE_DEVICE_CODE
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(MemoryManager<ExecutionSpace::DEVICE>,deviceManager,("device"));
#endif
}

#endif
