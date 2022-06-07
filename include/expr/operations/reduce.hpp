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
	    typename Buf,
	    typename Index,
	    typename...Args>
  INLINE_FUNCTION
  void hostReduce(Buf&& buf,const Index& nReductions,const Index& stride,Args&&...args)
  {
#if ENABLE_THREADS
# pragma omp parallel for
#endif
    for(Index first=0;first<nReductions;first++)
      {
	const Index second=first+stride;
	
	buf((T)first,std::forward<Args>(args)...)+=
	  // buf(first,cs...)+
	  buf((T)second,std::forward<Args>(args)...);
      }
  }
  
  template <typename T,
	    typename Buf,
	    typename Index,
	    typename...Args>
  INLINE_FUNCTION
  void deviceReduce(Buf&& buf,const Index& nReductions,const Index& stride,Args&&...args)
  {
    auto bufRef=buf.getRef();
    
    const auto avoidComplain=std::make_tuple(args...);
    
    DEVICE_LOOP(first,0,nReductions,
		const Index second=first+stride;
		
		// lambda function cannot capture a pack... boh!
		
		std::apply(bufRef((T)first),avoidComplain)+=
		std::apply(bufRef((T)second),avoidComplain);
		);
  }
  
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
			       
			       using Index=typename T::Index;
			       
			       Index n=~T::sizeAtCompileTime;
			       
			       if constexpr(T::sizeAtCompileTime==0)
				 n=~std::get<T>(dynamicSizes);
			       
			       while(n>1)
				 {
				   const Index stride=(n+1)/2;
				   const Index nReductions=n/2;
				   
				   if constexpr(Es==ExecSpace::HOST or (Es==ExecSpace::DEVICE and not ENABLE_DEVICE_CODE))
				     hostReduce<T>(buf,nReductions,stride,cs...);
				   else
#if ENABLE_DEVICE_CODE
				     if constexpr(Es==ExecSpace::DEVICE)
				       deviceReduce<T>(buf,nReductions,stride,cs...);
				     else
#endif
				       CRASH<<"Cannot reduce if the execution space is not clear";
				   
				   n=stride;
				 }
			     });
    
    if constexpr(Es==ExecSpace::DEVICE and std::decay_t<decltype(res)>::execSpace!=ExecSpace::DEVICE)
      {
	DynamicTens<ResComps,typename R::Fund,ExecSpace::DEVICE> tmp;
	tmp=buf(T(0));
	res=tmp;
      }
    else
      res=buf(T(0));
  }
  
  template <typename T,
	    typename E,
	    ExecSpace TargEs=ExecSpace::HOST>
  auto reduceOnComp(const Node<E>& _e)
  {
    const E& e=DE_CRTPFY(const E,&_e);
    
    using Comps=typename E::Comps;
    
    static_assert(tupleHasType<Comps,T>,"Cannot reduce on not present comps");
    
    using ResComps=TupleFilterAllTypes<Comps,CompsList<T>>;
    
    [[maybe_unused]]
    const auto dynamicSizes=e.getDynamicSizes();
    
    using Fund=typename E::Fund;
    
    // constexpr ExecSpace Es=E::execSpace;
    
    auto res=getTens<ResComps,Fund,TargEs>(dynamicSizes);
    
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
