#ifndef _MEMORY_HPP
#define _MEMORY_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file memory.hpp

#include <resources/memoryManagerGlobalVariablesDeclarations.hpp>

#include <resources/device.hpp>

namespace grill::memory
{
  namespace internal
  {
    /// Gets the appropriate memory manager
    template <ExecSpace ES>
    constexpr auto& _manager()
    {
#if ENABLE_DEVICE_CODE
      if constexpr(ES==ExecSpace::DEVICE)
	return deviceManager;
      else
#endif
	return hostManager;
    }
  }
  
  /// Gets the appropriate memory manager
  template <ExecSpace ES>
  constexpr auto& manager=
    internal::_manager<ES>();
  
  template <ExecSpace Dst,
	    ExecSpace Src,
	    typename F>
  INLINE_FUNCTION
  void memcpy(F* dst,const F* src,const size_t& n)
  {
    const size_t size=sizeof(F)*n;
    
#if ENABLE_DEVICE_CODE
    using namespace device;
    
    if constexpr (Dst==ExecSpace::DEVICE)
      {
	if constexpr (Src==ExecSpace::DEVICE)
	  memcpyDeviceToDevice(dst,src,size);
	else
	  memcpyHostToDevice(dst,src,size);
      }
    else
      if constexpr (Src==ExecSpace::DEVICE)
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
