#ifndef _DEVICEASSIGN_HPP
#define _DEVICEASSIGN_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/assign/deviceAssign.hpp
///
/// \brief Assign two expressions on the device

#include <expr/comps/compLoops.hpp>
#include <ios/logger.hpp>
#include <resources/device.hpp>

namespace grill
{
  namespace internal
  {
    template <typename C>
    struct MostSuitableCompForDeviceParallelAssign;
    
    template <typename...C>
    struct MostSuitableCompForDeviceParallelAssign<CompsList<C...>>
    {
      static constexpr INLINE_FUNCTION auto idLargerComp()
      {
	int s[]={~C::sizeAtCompileTime...};
	
	int i=0;
	
	for(int j=0;j<sizeof...(C);j++)
	  i=std::max(i,s[j]);
	
	return i;
      }
      
      template <typename Ds>
      constexpr INLINE_FUNCTION
      auto get(const Ds& dynamicSize) const
      {
	constexpr int nDynamicComps=((C::sizeAtCompileTime==0)+...);
	
	static_assert(nDynamicComps<=1,"Needs at most one dynamic comps to run on device, got too many");
	
	if constexpr(nDynamicComps==1)
	  return std::get<0>(dynamicSize);
	else
	  {
	    using Res=std::tuple_element_t<idLargerComp(),std::tuple<C...>>;
	    
	    return (Res)Res::sizeAtCompileTime;
	  }
      }
    };
  }
  
  /// Assign two expressions using device
  template <typename Lhs,
	    typename Rhs>
  INLINE_FUNCTION
  void deviceAssign(Lhs& lhs,
		    const Rhs& rhs)
  {
    const auto dynamicSizes=lhs.getDynamicSizes();
    
    const auto lc=internal::MostSuitableCompForDeviceParallelAssign<typename Lhs::Comps>().get(dynamicSizes);
    using Lc=std::decay_t<decltype(lc)>;
    
    auto lhsRef=lhs.getRef();
    const auto rhsRef=rhs.getRef();
    
    using OrthComps=TupleFilterAllTypes<typename Lhs::Comps,
					CompsList<Lc>>;
    
    DEVICE_LOOP(dc,Lc(0),lc,
		deviceLoopOnAllComps<OrthComps>(dynamicSizes,
								[lhsRef,rhsRef] DEVICE_ATTRIB (const auto&...comps) MUTABLE_INLINE_ATTRIBUTE
								{
								  lhsRef(comps...)=rhsRef(comps...);
								},
								dc);
		);
  }
}

#endif
