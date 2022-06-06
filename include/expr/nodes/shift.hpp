#ifndef _SHIFT_HPP
#define _SHIFT_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/shift.hpp

#include <expr/comps/dynamicCompsProvider.hpp>
#include <expr/nodes/node.hpp>
#include <expr/nodes/subNodes.hpp>
#include <expr/comps/comps.hpp>
#include <expr/nodes/conj.hpp>
#include <expr/nodes/producerDeclaration.hpp>
#include <expr/nodes/transp.hpp>
#include <lattice/lattice.hpp>
#include <lattice/parityProvider.hpp>
#include <metaprogramming/detectableAs.hpp>

namespace grill
{
  PROVIDE_DETECTABLE_AS(Shifter);
  
#define UNIVERSE Universe<NDims>
  
#define LATTICE Lattice<UNIVERSE>
  
  /// Shifter
  ///
  /// Forward declaration to capture the components
  template <typename _E,
	    typename _Comps,
	    typename _Fund,
	    typename L,
	    LatticeCoverage TLC>
  struct Shifter;
  
#define THIS					\
  Shifter<std::tuple<_E...>,CompsList<C...>,_Fund,LATTICE,TLC>
  
#define BASE					\
    Node<THIS>
  
  /// Shifter
  template <typename..._E,
	    typename...C,
	    typename _Fund,
	    int NDims,
	    LatticeCoverage TLC>
  struct THIS :
    DynamicCompsProvider<CompsList<C...>>,
    DetectableAsShifter,
    SubNodes<_E...>,
    ParityProvider<LATTICE,TLC>,
    BASE
  {
    /// Import the base expression
    using Base=BASE;
    
    using This=THIS;
    
#undef BASE
    
#undef THIS
    
    static_assert(sizeof...(_E)==1,"Expecting 1 argument");
    
    IMPORT_SUBNODE_TYPES;
    
    /// Components
    using Comps=
      CompsList<C...>;
    
    /// Fundamental tye
    using Fund=_Fund;
    
    using L=LATTICE;
    
    static constexpr LatticeCoverage latticeCoverage=TLC;
    
#undef LATTICE
    
    using Dir=typename L::Dir;
    
    const Ori ori;
    
    const Dir dir;
    
    /// Executes where allocated
    static constexpr ExecSpace execSpace=
      SubNode<0>::execSpace;
    
    /// Returns the dynamic sizes
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    decltype(auto) getDynamicSizes() const
    {
      return SUBNODE(0).getDynamicSizes();
    }
    
    /// Returns whether can assign
    INLINE_FUNCTION
    bool canAssign()
    {
      return false;
    }
    
    /// Return whether can be assigned at compile time
    static constexpr bool canAssignAtCompileTime=false;
    
    /// This is a lightweight object
    static constexpr bool storeByRef=false;
    
    /// Import assignment operator
    using Base::operator=;
    
    /// States whether the tensor can be simdified
    static constexpr bool canSimdify=
      SubNode<0>::canSimdify;
    
    /// Components on which simdifying
    using SimdifyingComp=
      std::conditional_t<canSimdify,typename SubNode<0>::SimdifyingComp,void>;
    
    /////////////////////////////////////////////////////////////////
    
#define PROVIDE_SIMDIFY(ATTRIB)					\
    /*! Returns a ATTRIB simdified view */			\
    INLINE_FUNCTION						\
    auto simdify() ATTRIB					\
    {								\
      return shift(SUBNODE(0).simdify(),ori,dir);			\
    }
    
    PROVIDE_SIMDIFY(const);
    
    PROVIDE_SIMDIFY(/* non const */);
    
#undef PROVIDE_SIMDIFY
    
    /////////////////////////////////////////////////////////////////
    
#define PROVIDE_GET_REF(ATTRIB)					\
    /*! Returns a reference */					\
    INLINE_FUNCTION						\
    auto getRef() ATTRIB					\
    {								\
      return recreateFromExprs(SUBNODE(0).getRef());		\
    }
    
    PROVIDE_GET_REF(const);
    
    PROVIDE_GET_REF(/* non const */);
    
#undef PROVIDE_GET_REF
    
    /////////////////////////////////////////////////////////////////
    
    //// Returns a similar shifter on a different orientation and direction
    template <typename T>
    INLINE_FUNCTION
    decltype(auto) recreateFromExprs(T&& t) const
    {
      return shift(std::forward<T>(t),ori,dir);
    }
    
    /////////////////////////////////////////////////////////////////
    
    /// Evaluate
    template <typename...TD>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    decltype(auto) eval(const TD&...td) const
    {
      using Parity=
	typename L::Parity;
      
      constexpr bool parityIsPassed=
	(std::is_same_v<std::decay_t<TD>,Parity> or...);
      
      Parity outerParity;
      
      if constexpr(parityIsPassed)
	outerParity=std::get<typename L::Parity>(std::make_tuple(td...));
      else
	outerParity=this->parity;
      
      auto compTransform=
	[this,&outerParity](const auto& c) INLINE_ATTRIBUTE
      {
	const auto lattice=SUBNODE(0).lattice;
	
	using I=std::decay_t<decltype(c)>;
	
	/// We need to store the inverce direction
	const Ori inverseOri=1-ori;
	
	if constexpr(std::is_same_v<I,typename L::LocEoSite>)
	  return lattice->loc.eoNeighbours(outerParity,c,inverseOri,dir);
	else
	  if constexpr(std::is_same_v<I,typename L::SimdLocEoSite>)
	    return lattice->simdLoc.eoNeighbours(outerParity,c,inverseOri,dir);
	  else
	    if constexpr(std::is_same_v<I,Parity>)
	      return L::oppositeParity(outerParity);
	    else
	      return c;
      };
      
      return SUBNODE(0)(compTransform(td)...);
    }
    
    /// Construct
    template <typename T>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Shifter(T&& arg,
	    const Ori& ori,
	    const Dir& dir) :
      SubNodes<_E...>(std::forward<T>(arg)),
      ori(ori),
      dir(dir)
    {
    }
  };
  
  /// Shifts an expression, returning a view as shifted in the given direction and orientation
  ///
  /// NB: when shifting forward, we need to look for backward neighbours
  template <typename _E,
	    typename D,
	    ENABLE_THIS_TEMPLATE_IF(isNode<_E>)>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
  decltype(auto) shift(_E&& e,const Ori& ori,const D& dir)
  {
    using E=std::decay_t<_E>;
    
    if constexpr(isCompsBinder<_E> or isConjugator<_E> or isTransposer<_E>)
      return e.recreateFromExprs(shift(e.template subNode<0>(),ori,dir));
    else
      if constexpr(isProducer<_E>)
	return e.recreateFromExprs(shift(e.template subNode<0>(),ori,dir),
				   shift(e.template subNode<1>(),ori,dir));
      else
	return Shifter<std::tuple<_E>,typename E::Comps,typename E::Fund,typename E::L,oppositeLatticeCoverage(E::latticeCoverage)>(std::forward<_E>(e),ori,dir);
  }
}

#endif
