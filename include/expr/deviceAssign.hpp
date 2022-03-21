#ifndef _DEVICEASSIGN_HPP
#define _DEVICEASSIGN_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/deviceAssign.hpp
///
/// \brief Assign two expressions on the device

#include <expr/compLoops.hpp>
#include <ios/logger.hpp>
#include <resources/device.hpp>

namespace esnort
{
  /// Assign two expressions using device
  template <typename Lhs,
	    typename Rhs>
  INLINE_FUNCTION
  void deviceAssign(Lhs& lhs,
		    const Rhs& rhs)
  {
    constexpr int nDynamicComps=Lhs::nDynamicComps;
    
    static_assert(nDynamicComps==1,"Needs exactly one dynamic comps to run on device");
    
    /// For the time being, we assume that there is a single
    /// dynamic component, and we loop with the gpu threads on
    /// it, then we loop internally on the others
    LOGGER<<"Using device kernel";
    
    const auto dynamicSizes=lhs.getDynamicSizes();
    
    const auto dynamicSize=std::get<0>(dynamicSizes);
    
    using DC=std::tuple_element_t<0,typename Lhs::DynamicComps>;
    
    auto lhsRef=lhs.getRef();
    const auto rhsRef=rhs.getRef();
    
    DEVICE_LOOP(dc,DC(0),dynamicSize,
		deviceLoopOnAllComps<typename Lhs::StaticComps>(dynamicSizes,
								[lhsRef,rhsRef] DEVICE_ATTRIB (const auto&...comps) MUTABLE_INLINE_ATTRIBUTE
								{
								  lhsRef(comps...)=rhsRef(comps...);
								},
								dc);
		);
  }
}

#endif
