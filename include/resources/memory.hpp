#ifndef _MEMORY_HPP
#define _MEMORY_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file memory.hpp

#include <resources/memoryGlobalVariablesDeclarations.hpp>

namespace esnort::memory
{
  namespace internal
  {
    /// Gets the appropriate memory manager
    template <ExecutionSpace ES>
    constexpr auto& _manager()
    {
#if ENABLE_DEVICE_CODE
      if constexpr(ES==ExecutionSpace::DEVICE)
	return deviceManager;
      else
#endif
	return hostManager;
    }
  }
  
  /// Gets the appropriate memory manager
  template <ExecutionSpace ES>
  constexpr auto& manager=
    internal::_manager<ES>();
}

#endif
