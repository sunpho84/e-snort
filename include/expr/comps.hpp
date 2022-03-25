#ifndef _COMPS_HPP
#define _COMPS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file comps.hpp

#include <tuple>

#include <expr/comp.hpp>
#include <resources/SIMD.hpp>
#include <tuples/tupleReplaceType.hpp>

namespace esnort
{
  /// Collection of components
  template <typename...Tc>
  using CompsList=
    std::tuple<Tc...>;
  
  /// Alias to make it easier to understand tensor instantiation
  template <typename...Tc>
  using OfComps=
    CompsList<Tc...>;
  
  /////////////////////////////////////////////////////////////////
  
  namespace internal
  {
    template <typename Tp>
    struct _LastComp;
    
    template <typename...Tp>
    struct _LastComp<OfComps<Tp...>>
    {
      static constexpr int nComps=sizeof...(Tp);
      
      static constexpr auto _lastCompTypeProvider()
      {
	if constexpr(nComps>0)
	  return std::tuple_element_t<nComps-1,std::tuple<Tp...>>{};
      }
      
      using type=
	decltype(_lastCompTypeProvider());
    };
  }
  
  /// Provide last component of a tuple
  template <typename Tp>
  using LastComp=typename internal::_LastComp<Tp>::type;
  
  /////////////////////////////////////////////////////////////////
  
  template <typename Index,
	    int Size>
  struct NonSimdifiedComp :
    Comp<compFeat::IsTransposable::FALSE,
	 Index,
	 NonSimdifiedComp<Index,Size>>
  {
    using Base=Comp<compFeat::IsTransposable::FALSE,
		    Index,
		    NonSimdifiedComp<Index,Size>>;
    
    using Base::Base;
    
    /// Size at compile time
    static constexpr int sizeAtCompileTime=Size;
  };
  
  /// Returns whether the last component can simdify
  ///
  /// Forward declaration
  template <typename Tp,
	    typename F>
  struct CompsListSimdifiableTraits;
  
  /// Returns whether the last component can simdify
  template <typename...Tp,
	    typename F>
  struct CompsListSimdifiableTraits<CompsList<Tp...>,F>
  {
    static constexpr int nComps=sizeof...(Tp);
    
    using LastComp=esnort::LastComp<CompsList<Tp...>>;
    
    static constexpr int _lastCompSizeProvider()
    {
      if constexpr(nComps>0)
	return LastComp::sizeAtCompileTime;
      else
	return 0;
    }
    
    static constexpr int lastCompSize=
      _lastCompSizeProvider();
    
    using Traits=
      SimdOfTypeTraits<F,lastCompSize>;
    
    static constexpr bool canSimdify=
      Traits::canSimdify();
    
    using SimdFund=
      typename Traits::type;
    
    static constexpr int nNonSimdifiedElements=
      Traits::nNonSimdifiedElements();
    
    using Comps=
      TupleReplaceType<CompsList<Tp...>,LastComp,NonSimdifiedComp<typename LastComp::Index,nNonSimdifiedElements>>;
  };
}

#endif
