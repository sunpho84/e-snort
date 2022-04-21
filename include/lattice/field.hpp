#ifndef _FIELD_HPP
#define _FIELD_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file grill/field.hpp

#include <expr/assign/executionSpace.hpp>
#include <expr/comps/comps.hpp>
#include <lattice/fieldCompsProvider.hpp>
#include <lattice/lattice.hpp>

namespace grill
{
  /// Defines a field
  template <typename...C,
	    typename Fund,
	    int NDims,
	    int...I,
	    LatticeCoverage LC,
	    FieldLayout FL,
	    ExecSpace ES>
  struct Field<CompsList<C...>,Fund,Lattice<Universe<NDims,std::integer_sequence<int,I...>>>,LC,FL,ES>
  {
    using L=Lattice<Universe<NDims,std::integer_sequence<int,I...>>>;
    
    const L* lattice;
    
    using Comps=FieldCompsProvider<CompsList<C...>,L,LC,FL>;
    
    DynamicTens<Comps,Fund,ES> data;
    
    Field(const L& lattice,
	  const bool& withBord=false) :
      lattice(&lattice)
    {
      if(withBord)
	CRASH<<"Not yet implemented";
      
      if constexpr(FL==FieldLayout::SIMD)
	data.allocate(std::make_tuple(lattice.simdLocEoVol));
      else
	data.allocate(std::make_tuple(lattice.locEoVol));
    }
  };
}

#endif
