#ifndef _REDUCE_HPP
#define _REDUCE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/operations/reduce.hpp

#include <tuples/tupleHasType.hpp>
#include <expr/nodes/node.hpp>
#include <expr/nodes/tens.hpp>

namespace grill
{
  
  template <typename T,
	    typename _R,
	    typename E>
  void reduceOnComp(_R&& res,const E& e)
  {
    using R=typename std::decay_t<_R>;
    
    using ResComps=typename R::Comps;
    
    using BufComps=
      TupleCat<
#if not ENABLE_DEVICE_CODE
	std::tuple<T>,
#endif
      ResComps
#if ENABLE_DEVICE_CODE
      ,std::tuple<T>
#endif
      >;
    
    constexpr ExecSpace Es=E::execSpace;
    
    [[maybe_unused]]
    const auto dynamicSizes=e.getDynamicSizes();
    
    auto buf=getTens<BufComps,typename R::Fund,Es>(dynamicSizes);
    buf=e;
    
    loopOnAllComps<ResComps>(dynamicSizes,
			     [dynamicSizes,&buf](const auto&...cs) INLINE_ATTRIBUTE
			     {
			       (void)&dynamicSizes;
			       
			       T n=T::sizeAtCompileTime;
			       
			       if constexpr(T::sizeAtCompileTime==0)
				     n=std::get<T>(dynamicSizes);
			       
			       while(n>1)
				 {
				   const T stride=(n+1)/2;
				   const T nreductions=n/2;
				   
				   for(T first=0;first<nreductions;first++)
				     {
				       const T second=first+stride;
				       
				       buf(first,cs...)+=
					 // buf(first,cs...)+
					 buf(second,cs...);
				     }
				   
				   n=stride;
				 }
			     });
    
    res=buf(T(0));
  }
  
  template <typename T,
	    typename E>
  auto reduceOnComp(const Node<E>& _e)
  {
    const E& e=DE_CRTPFY(const E,&_e);
    
    using Comps=typename E::Comps;
    
    static_assert(tupleHasType<Comps,T>,"Cannot reduce on not present comps");
	
    using ResComps=TupleFilterAllTypes<Comps,CompsList<T>>;
    
    [[maybe_unused]]
    const auto dynamicSizes=e.getDynamicSizes();
    
    using Fund=typename E::Fund;
    
    constexpr ExecSpace Es=E::execSpace;
    
    auto res=getTens<ResComps,Fund,Es>(dynamicSizes);
    
    if constexpr(E::canSimdify and not std::is_same_v<typename E::SimdifyingComp,T>)
      {
	LOGGER<<"Simdifying!";
	reduceOnComp<T>(res.simdify(),e.simdify());
      }
    else
      reduceOnComp<T>(res,e);
    
    return res;
  }
}

#endif
