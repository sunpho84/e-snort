#ifndef _ALGEBRIC_SUM_HPP
#define _ALGEBRIC_SUM_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/algebricSum.hpp

#include <expr/comps/comps.hpp>
#include <expr/nodes/conj.hpp>
#include <expr/nodes/node.hpp>
#include <expr/nodes/subNodes.hpp>
#include <metaprogramming/arithmeticTraits.hpp>
#include <tuples/uniqueTupleFromTuple.hpp>

namespace esnort
{
  PROVIDE_DETECTABLE_AS(AlgebricSummer);
  
  /// AlgebricSummer
  ///
  /// Forward declaration to capture the components
  template <typename _E,
	    typename _Comps,
	    typename _Fund,
	    typename _Comb,
	    typename _Is=std::make_integer_sequence<int,std::tuple_size_v<_E>>>
  struct AlgebricSummer;
  
#define THIS								\
  AlgebricSummer<std::tuple<_E...>,CompsList<C...>,_Fund,_Comb,std::integer_sequence<int,Is...>>
  
#define BASE					\
    Node<THIS>
  
  /// AlgebricSummer
  ///
  template <typename..._E,
	    typename...C,
	    typename _Fund,
	    typename _Comb,
	    int...Is>
  struct THIS :
    DynamicCompsProvider<CompsList<C...>>,
    DetectableAsAlgebricSummer,
    SubNodes<_E...>,
    BASE
  {
    /// Import the base expression
    using Base=BASE;
    
    using This=THIS;
    
#undef BASE
    
#undef THIS
    
    static_assert(sizeof...(_E)==2,"Expecting 2 addends");
    
    IMPORT_SUBNODE_TYPES;
    
    /// Components
    using Comps=
      CompsList<C...>;
    
    /// Fundamental tye
    using Fund=_Fund;
    
    /// Execution space
    static constexpr ExecSpace execSpace=
      commonExecSpace<std::remove_reference_t<_E>::execSpace...>();
    
    static_assert(execSpace!=ExecSpace::UNDEFINED,"Cannot define algebric sum in undefined exec space");
    
    /// List of dynamic comps
    using DynamicComps=
      typename DynamicCompsProvider<Comps>::DynamicComps;
    
    /// Sizes of the dynamic components
    const DynamicComps dynamicSizes;
    
    /// Returns the dynamic sizes
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    decltype(auto) getDynamicSizes() const
    {
      return dynamicSizes;
    }
    
    /// Returns whether can assign
    INLINE_FUNCTION
    constexpr bool canAssign()
    {
      return false;
    }
    
    /// This is a lightweight object
    static constexpr bool storeByRef=false;
    
    /// Import assignment operator
    using Base::operator=;
    
    /// Return whether can be assigned at compile time
    static constexpr bool canAssignAtCompileTime=false;
    
    /// Determine if algebric sum can be simdified - to be extended
    static constexpr bool simdifyCase()
    {
      return
	(SubNode<Is>::canSimdify and...) and
	std::is_same_v<typename SubNode<Is>::SimdifyingComp...>;
    }
    
    /// States whether the tensor can be simdified
    static constexpr bool canSimdify=
      simdifyCase();
    
    /// Type of the combining function
    using Comb=_Comb;
    
    /// Combining function
    Comb combine;
    
    /// \todo improve
    
    /// Components on which simdifying
    using SimdifyingComp=typename SubNode<0>::SimdifyingComp;
    
#define PROVIDE_SIMDIFY(ATTRIB)					\
    /*! Returns a ATTRIB simdified view */			\
    INLINE_FUNCTION						\
    auto simdify() ATTRIB					\
    {								\
      return							\
	combine(SUBNODE(Is).simdify()...);			\
    }
    
    PROVIDE_SIMDIFY(const);
    
    PROVIDE_SIMDIFY(/* non const */);
    
#undef PROVIDE_SIMDIFY
    
#define PROVIDE_GET_REF(ATTRIB)					\
    /*! Returns a reference */					\
    INLINE_FUNCTION						\
    auto getRef() ATTRIB					\
    {								\
      return							\
	combine(SUBNODE(Is).getRef()...);				\
    }
    
    PROVIDE_GET_REF(const);
    
    PROVIDE_GET_REF(/* non const */);
    
#undef PROVIDE_GET_REF
    
    /// Gets the components for the I-th addend
    template <int I,
	      typename...Cs>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    static auto getCompsForAddend(const Cs&...cs)
    {
      return tupleGetSubset<typename SubNode<I>::Comps>(std::make_tuple(cs...));
    }
    
    /// Evaluate
    template <typename...Cs>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Fund eval(const Cs&...cs) const
    {
      return
	combine(std::apply(SUBNODE(Is),getCompsForAddend<Is>(cs...))...);
    }
    
    /// Construct
    template <typename...T>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    AlgebricSummer(const DynamicComps& dynamicSizes,
		   UNIVERSAL_CONSTRUCTOR_IDENTIFIER,
		   Comb combine,
		   T&&...addends) :
      SubNodes<_E...>(addends...),
      dynamicSizes(dynamicSizes),
      combine(combine)
    {
    }
  };
  
  template <typename Comb,
	    typename..._E,
	    ENABLE_THIS_TEMPLATE_IF(isNode<_E> and...)>
  INLINE_FUNCTION HOST_DEVICE_ATTRIB
  auto algebricSum(Comb combine,_E&&...e)
  {
    /// Computes the result components
    using Comps=
      UniqueTupleFromTuple<TupleCat<typename std::decay_t<_E>::Comps...>>;
    
    /// Determine the fundamental type of the result
    using Fund=
      decltype(combine(typename std::decay_t<_E>::Fund{}...));
    
    /// Resulting type
    using Res=
      AlgebricSummer<std::tuple<decltype(e)...>,
		     Comps,
		     Fund,
		     Comb>;
    
    /// Resulting dynamic components
    const auto dc=
      dynamicCompsCombiner<typename Res::DynamicComps>(std::make_tuple(e.getDynamicSizes()...));
    
    return
      Res(dc,UNIVERSAL_CONSTRUCTOR_CALL,combine,std::forward<_E>(e)...);
  }
  
#define CATCH_OPERATOR(OP)						\
  /*! Catch the OP operator */						\
  template <typename E1,						\
	    typename E2,						\
	    ENABLE_THIS_TEMPLATE_IF(isNode<E1> and isNode<E2>)>		\
  INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB				\
  auto operator OP(E1&& e1,						\
		   E2&& e2)						\
  {									\
    auto combine=							\
      [](const auto&...s) CONSTEXPR_INLINE_ATTRIBUTE			\
      {									\
	return (s OP ...);						\
      };								\
									\
    return								\
      algebricSum(combine,std::forward<E1>(e1),std::forward<E2>(e2));	\
  }
  
  CATCH_OPERATOR(+);
  
  CATCH_OPERATOR(-);
  
#undef CATCH_OPERATOR
}

#endif
