#ifndef _MEMORY_HPP
#define _MEMORY_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file memory.hpp

#include <resources/memoryGlobalVariablesDeclarations.hpp>

#include <resources/device.hpp>

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
  
  template <ExecutionSpace Dst,
	    ExecutionSpace Src,
	    typename F>
  INLINE_FUNCTION
  void memcpy(F* dst,const F* src,const size_t& n)
  {
    const size_t size=sizeof(F)*n;
    
#if ENABLE_DEVICE_CODE
    using namespace device;
    
    if constexpr (Dst==ExecutionSpace::DEVICE)
      {
	if constexpr (Src==ExecutionSpace::DEVICE)
	  memcpyDeviceToDevice(dst,src,size);
	else
	  memcpyHostToDevice(dst,src,size);
      }
    else
      if constexpr (Src==ExecutionSpace::DEVICE)
	memcpyDeviceToHost(dst,src,size);
      else
#endif
	::memcpy(dst,src,size);
  }
  
  /// Initializes the memory managers
  void initialize();
  
  /// Finalize the memory managers
  void finalize();
}

#endif
