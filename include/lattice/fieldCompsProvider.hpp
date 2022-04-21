#ifndef _FIELDCOMPSPROVIDER_HPP
#define _FIELDCOMPSPROVIDER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file lattice/fieldCompsProvider.hpp

#include <lattice/fieldDeclaration.hpp>
#include <expr/comps/comps.hpp>

namespace grill
{
  namespace internal
  {
    /// Actual components to be exposed for a field
    ///
    /// Forward declaration of internal implementation
    template <typename TP,
	      typename L,
	      LatticeCoverage LC,
	      FieldLayout FL>
    struct _FieldCompsProvider;
    
#define PROVIDE_FIELD_COMPS_PROVIDER(COVERAGE,LAYOUT,TYPES...)		\
    									\
    template <typename...C,						\
	      typename L>						\
    struct _FieldCompsProvider<CompsList<C...>,L,			\
			       LatticeCoverage::COVERAGE,		\
			       FieldLayout::LAYOUT>			\
    {									\
      using type=CompsList<TYPES>;					\
    }
    
    PROVIDE_FIELD_COMPS_PROVIDER(EVEN_ODD,SERIAL,typename L::Parity,typename L::LocEoSite,C...);
    PROVIDE_FIELD_COMPS_PROVIDER(EVEN,SERIAL,typename L::LocEoSite,C...);
    PROVIDE_FIELD_COMPS_PROVIDER(ODD,SERIAL,typename L::LocEoSite,C...);
    
    PROVIDE_FIELD_COMPS_PROVIDER(EVEN_ODD,SIMD,typename L::Parity,typename L::SimdLocEoSite,C...,typename L::SimdRank);
    PROVIDE_FIELD_COMPS_PROVIDER(EVEN,SIMD,typename L::LocSimdEoSite,C...,typename L::SimdRank);
    PROVIDE_FIELD_COMPS_PROVIDER(ODD,SIMD,typename L::LocSimdEoSite,C...,typename L::SimdRank);
    
#undef PROVIDE_FIELD_COMPS_PROVIDER
  }
  
  /// Actual components to be exposed for a field
  template <typename TP,
	    typename L,
	    LatticeCoverage LC,
	    FieldLayout FL>
  using FieldCompsProvider=typename internal::_FieldCompsProvider<TP,L,LC,FL>::type;
}

#endif
