#ifndef _FIELDCOMPSPROVIDER_HPP
#define _FIELDCOMPSPROVIDER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file lattice/fieldCompsProvider.hpp

#include <lattice/fieldDeclaration.hpp>
#include <metaprogramming/constnessChanger.hpp>
#include <expr/comps/comps.hpp>

namespace grill
{
  /// Actual components to be exposed for a field
  ///
  /// Forward declaration of internal implementation
  template <typename TP,
	    typename F,
	    typename L,
	    LatticeCoverage LC,
	    FieldLayout FL>
  struct FieldCompsProvider;
  
#define PROVIDE_FIELD_COMPS_PROVIDER(COVERAGE,LAYOUT,TYPES...)		\
  									\
  template <typename...C,						\
	    typename F,							\
	    typename L>							\
  struct FieldCompsProvider<CompsList<C...>,F,L,			\
			    LatticeCoverage::COVERAGE,			\
			    FieldLayout::LAYOUT>			\
  {									\
    using Comps=CompsList<TYPES>;					\
									\
    using Fund=F;							\
  }
  
  PROVIDE_FIELD_COMPS_PROVIDER(EVEN_ODD,SERIAL,typename L::Parity,typename L::LocEoSite,C...);
  PROVIDE_FIELD_COMPS_PROVIDER(EVEN,SERIAL,typename L::LocEoSite,C...);
  PROVIDE_FIELD_COMPS_PROVIDER(ODD,SERIAL,typename L::LocEoSite,C...);
  
  PROVIDE_FIELD_COMPS_PROVIDER(EVEN_ODD,GPU,C...,typename L::Parity,typename L::LocEoSite);
  PROVIDE_FIELD_COMPS_PROVIDER(EVEN,GPU,C...,typename L::LocEoSite);
  PROVIDE_FIELD_COMPS_PROVIDER(ODD,GPU,C...,typename L::LocEoSite);
  
  PROVIDE_FIELD_COMPS_PROVIDER(EVEN_ODD,SIMDIFIABLE,typename L::Parity,typename L::SimdLocEoSite,C...,typename L::SimdRank);
  PROVIDE_FIELD_COMPS_PROVIDER(EVEN,SIMDIFIABLE,typename L::SimdLocEoSite,C...,typename L::SimdRank);
  PROVIDE_FIELD_COMPS_PROVIDER(ODD,SIMDIFIABLE,typename L::SimdLocEoSite,C...,typename L::SimdRank);
  
#undef PROVIDE_FIELD_COMPS_PROVIDER
  
  template <typename...C,
	    LatticeCoverage CO,
	    typename L,
	    typename F>
  struct FieldCompsProvider<CompsList<C...>,F,L,CO,FieldLayout::SIMDIFIED>
  {
    using NonSimdComps=
      typename FieldCompsProvider<CompsList<C...>,F,L,CO,FieldLayout::SIMDIFIABLE>::Comps;
    
    using Traits=
      CompsListSimdifiableTraits<NonSimdComps,std::decay_t<F>>;
    
    using Comps=
      typename Traits::Comps;
    
    using Fund=
      ConstIf<std::is_const_v<F>,typename Traits::SimdFund>;
  };
}

#endif
