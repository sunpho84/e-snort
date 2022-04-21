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
  PROVIDE_DETECTABLE_AS(Field);
  
#define THIS					\
  Field<CompsList<C...>,_Fund,Lattice<Universe<NDims,std::integer_sequence<int,I...>>>,LC,FL,ES>
  
#define BASE					\
    Node<THIS>
  
  /// Defines a field
  template <typename...C,
	    typename _Fund,
	    int NDims,
	    int...I,
	    LatticeCoverage LC,
	    FieldLayout FL,
	    ExecSpace ES>
  struct THIS :
    DynamicCompsProvider<CompsList<C...>>,
    DetectableAsField,
    // SubNodes<_E...>,
    BASE
  {
    /// Import the base expression
    using Base=BASE;
    
    using This=THIS;
    
#undef BASE
    
#undef THIS
    
    /// Referred lattice
    using L=Lattice<Universe<NDims,std::integer_sequence<int,I...>>>;
    
    /// Components
    using Comps=FieldCompsProvider<CompsList<C...>,L,LC,FL>;
    
    /// Fundamental tye
    using Fund=_Fund;
    
    /// Executes where allocated
    static constexpr ExecSpace execSpace=ES;
    
    /// Returns the size needed to allocate
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    decltype(auto) getLocSize() const
    {
      if constexpr(FL==FieldLayout::SIMD)
	return (lattice->simdLocEoVol);
      else
	return (lattice->locEoVol);
    }
    
    /// Returns the dynamic sizes
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    auto getDynamicSizes() const
    {
      return std::make_tuple(getLocSize());
    }
    
#define PROVIDE_EVAL(ATTRIB)					\
    template <typename...U>					\
    constexpr INLINE_FUNCTION					\
    ATTRIB Fund& eval(const U&...cs) ATTRIB			\
    {								\
      return data(cs...);					\
    }
    
    PROVIDE_EVAL(const);
    
    PROVIDE_EVAL(/* non const */);
    
#undef PROVIDE_EVAL
    
    /// Return whether can be assigned at compile time
    static constexpr bool canAssignAtCompileTime=
      not std::is_const_v<Fund>;
    
    /// Returns a const reference
    auto getRef() const
    {
      CRASH<<"Not yet implemented";
    }
    
    /// States whether the field can be simdified
    static constexpr bool canSimdify=
      false; /// Of course needs to be improved
    
    
    /// We keep referring to the original object
    static constexpr bool storeByRef=true;
    
    /// Returns that can assign
    constexpr INLINE_FUNCTION
    bool canAssign()
    {
      return canAssignAtCompileTime;
    }
    
    const L* lattice;
    
    DynamicTens<Comps,Fund,ES> data;
    
    /// Create a field
    Field(const L& lattice,
	  const bool& withBord=false) :
      lattice(&lattice)
    {
      if(withBord)
	CRASH<<"Not yet implemented";
      
      data.allocate(std::make_tuple(getLocSize()));
    }
  };
}

#endif
