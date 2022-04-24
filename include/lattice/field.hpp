#ifndef _FIELD_HPP
#define _FIELD_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file grill/field.hpp

#include <expr/assign/executionSpace.hpp>
#include <expr/comps/comps.hpp>
#include <expr/nodes/tensRef.hpp>
#include <lattice/fieldCompsProvider.hpp>
#include <lattice/lattice.hpp>

namespace grill
{
  PROVIDE_DETECTABLE_AS(Field);
  
#define UNIVERSE Universe<NDims>
  
#define LATTICE Lattice<UNIVERSE>
  
#define FIELD_COMPS typename FieldCompsProvider<CompsList<C...>,_Fund,LATTICE,LC,FL>::Comps
  
#define THIS					\
  Field<CompsList<C...>,_Fund,LATTICE,LC,FL,ES,IsRef>
  
#define BASE					\
    Node<THIS>
  
  /// Defines a field
  template <typename...C,
	    typename _Fund,
	    int NDims,
	    LatticeCoverage LC,
	    FieldLayout FL,
	    ExecSpace ES,
	    bool IsRef>
  struct THIS :
    DynamicCompsProvider<FIELD_COMPS>,
    DetectableAsField,
    // SubNodes<_E...>,
    BASE
  {
    /// Import the base expression
    using Base=BASE;
    
    using This=THIS;
    
#undef BASE
    
#undef THIS
    
    /// Importing assignment operator from BaseTens
    using Base::operator=;
    
    /// Copy assign
    INLINE_FUNCTION
    Field& operator=(const Field& oth)
    {
      Base::operator=(oth);
      
      return *this;
    }
    
    /// Move assign
    INLINE_FUNCTION
    Field& operator=(Field&& oth)
    {
      std::swap(data,oth.data);
      
      return *this;
    }
    
    /// Referred lattice
    using L=LATTICE;
    
    /// Components
    using Comps=
      FIELD_COMPS;
    
    /// Import dynamic comps
    using DynamicComps=
      typename DynamicCompsProvider<FIELD_COMPS>::DynamicComps;
    
#undef FIELD_COMPS
    
#undef LATTICE
    
#undef UNIVERSE
    
    /// Fundamental tye
    using Fund=
      typename FieldCompsProvider<CompsList<C...>,_Fund,L,LC,FL>::Fund;
    
    /// Internal storage type
    using Data=
      std::conditional_t<IsRef,
      TensRef<Comps,Fund,ES>,
      DynamicTens<Comps,Fund,ES>>;
    
    /// Executes where allocated
    static constexpr ExecSpace execSpace=ES;
    
    /// Returns the size needed to allocate
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    decltype(auto) getLocSize() const
    {
      if constexpr(FL==FieldLayout::SIMDIFIABLE or
		   FL==FieldLayout::SIMDIFIED)
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
    
    /////////////////////////////////////////////////////////////////
    
#define PROVIDE_GET_REF(ATTRIB)						\
  auto getRef() ATTRIB							\
  {									\
    return Field<CompsList<C...>,ATTRIB _Fund,L,LC,FL,ES,true>		\
      (*lattice,withHalo,data.storage,data.nElements,data.getDynamicSizes()); \
  }
  
  PROVIDE_GET_REF(const);
  
  PROVIDE_GET_REF(/* non const */);
  
  /////////////////////////////////////////////////////////////////
  
#undef PROVIDE_GET_REF
    
#define PROVIDE_SIMDIFY(ATTRIB)						\
  INLINE_FUNCTION							\
  auto simdify() ATTRIB							\
    {									\
      return Field<CompsList<C...>,ATTRIB _Fund,L,LC,FieldLayout::SIMDIFIED,ES,true> \
	(*lattice,withHalo,(ATTRIB void*)data.storage,data.nElements,data.getDynamicSizes()); \
    }
  
  PROVIDE_SIMDIFY(const);
  
  PROVIDE_SIMDIFY(/* non const */);
  
#undef PROVIDE_SIMDIFY
  
    /// States whether the field can be simdified
    static constexpr bool canSimdify=
      FL!=FieldLayout::SIMDIFIED and Data::canSimdify;
    
    /// Simdifying component
    using SimdifyingComp=
      typename Data::SimdifyingComp;
    
    /// We keep referring to the original object
    static constexpr bool storeByRef=not IsRef;
    
    /// Returns that can assign
    constexpr INLINE_FUNCTION
    bool canAssign()
    {
      return canAssignAtCompileTime;
    }
    
    /// Underneath lattice
    const L* lattice;
    
    /// Storage data
    Data data;
    
    /// Determine whether the halos are allocated
    const bool withHalo;
    
    /// Create a field
    Field(const L& lattice,
	  const bool& withHalo=false) :
      lattice(&lattice),
      withHalo(withHalo)
    {
      static_assert(not IsRef,"Can allocate only if not a reference");
      
      if(withHalo)
	CRASH<<"Not yet implemented";
      
      data.allocate(std::make_tuple(getLocSize()));
    }
    
    /// Create a refence to a field
    Field(const L& lattice,
	  const bool& withHalo,
	  void* storage,
	  const int64_t& nElements,
	  const DynamicComps& dynamicSizes) :
      lattice(&lattice),
      data((Fund*)storage,nElements,dynamicSizes),
      withHalo(withHalo)
    {
      static_assert(IsRef,"Can initialize as reference only if declared as a reference");
    }
    
    /// Copy constructor
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    Field(const Field& oth) :
      lattice(oth.lattice),
      data(oth.data),
      withHalo(oth.withHalo)
    {
    }
  };
}

#endif
