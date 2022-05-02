#ifndef _BINDCOMPS_HPP
#define _BINDCOMPS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/bindComps.hpp

#include <expr/comps/comps.hpp>
#include <expr/comps/dynamicCompsProvider.hpp>
#include <expr/assign/executionSpace.hpp>
#include <expr/nodes/nodeDeclaration.hpp>
#include <expr/exprRefOrVal.hpp>
#include <expr/nodes/subNodes.hpp>
#include <metaprogramming/templateEnabler.hpp>
#include <metaprogramming/universalReference.hpp>
#include <tuples/tupleFilter.hpp>
#include <tuples/tupleHasType.hpp>

namespace grill
{
  PROVIDE_DETECTABLE_AS(CompsBinder);
  
  /// Component binder
  ///
  /// Forward declaration to capture the components
  template <typename _BC,
	    typename _E,
	    typename _Comps,
	    typename _Fund>
  struct CompsBinder;
  
#define THIS					\
  CompsBinder<CompsList<Bc...>,std::tuple<_E...>,CompsList<C...>,_Fund>
  
#define BASE					\
    Node<THIS>
  
  /// Component binder
  ///
  template <typename...Bc,
	    typename..._E,
	    typename...C,
	    typename _Fund>
  struct THIS :
    DynamicCompsProvider<CompsList<C...>>,
    DetectableAsCompsBinder,
    SubNodes<_E...>,
    BASE
  {
    /// Import the base expression
    using Base=BASE;
    
    using This=THIS;
    
#undef BASE
    
#undef THIS
    
    static_assert(sizeof...(_E)==1,"Expecting 1 argument");
    
    /// Components
    using Comps=
      CompsList<C...>;
    
    /// Fundamental tye
    using Fund=_Fund;
    
    IMPORT_SUBNODE_TYPES;
    
    /// Executes where allocated
    static constexpr ExecSpace execSpace=
      SubNode<0>::execSpace;
    
    /// Returns the dynamic sizes
    const auto getDynamicSizes() const
    {
      return tupleGetSubset<typename CompsBinder::DynamicComps>(SUBNODE(0).getDynamicSizes());
    }
    
    /// Returns whether can assign
    INLINE_FUNCTION
    bool canAssign()
    {
      return SUBNODE(0).canAssign();
    }
    
    /// This is a lightweight object
    static constexpr bool storeByRef=false;
    
    /// Import assignment operator
    using Base::operator=;
    
    This& operator=(const This& oth)
    {
      *static_cast<Base*>(this)=*static_cast<const Base*>(&oth);
      
      return *this;
    }
    
    /// Bound components
    using BoundComps=
      CompsList<Bc...>;
    
    /// Return whether can be assigned at compile time
    static constexpr bool canAssignAtCompileTime=
      SubNode<0>::canAssignAtCompileTime;
    
    /// States whether the tensor can be simdified
    static constexpr bool canSimdify=
      SubNode<0>::canSimdify and
      not tupleHasType<BoundComps,typename SubNode<0>::SimdifyingComp>;
    
    /// Components on which simdifying
    using SimdifyingComp=
      std::conditional_t<canSimdify,typename SubNode<0>::SimdifyingComp,void>;
    
    /// Components that have been bound
    const BoundComps boundComps;
    
#define PROVIDE_SIMDIFY(ATTRIB)						\
    /*! Returns a ATTRIB simdified view */				\
    INLINE_FUNCTION							\
    auto simdify() ATTRIB						\
    {									\
      if constexpr(0)							\
	LOGGER<<" simdifying binder from "<<typeid(SUBNODE(0)).name()	\
	      <<" to "<<typeid(SUBNODE(0).simdify()).name();		\
      									\
      return bindComps(SUBNODE(0).simdify(),boundComps);		\
    }
    
    PROVIDE_SIMDIFY(const);
    
    PROVIDE_SIMDIFY(/* non const */);
    
#undef PROVIDE_SIMDIFY
    
#define PROVIDE_GET_REF(ATTRIB)					\
    /*! Returns a reference */					\
    INLINE_FUNCTION						\
    auto getRef() ATTRIB					\
    {								\
      return SUBNODE(0).getRef()(std::get<Bc>(boundComps)...);	\
    }
    
    PROVIDE_GET_REF(const);
    
    PROVIDE_GET_REF(/* non const */);
    
#undef PROVIDE_GET_REF
    
#define PROVIDE_EVAL(ATTRIB)						\
    template <typename...U>						\
    HOST_DEVICE_ATTRIB constexpr INLINE_FUNCTION			\
    decltype(auto) eval(const U&...cs) ATTRIB				\
    {									\
      return								\
	SUBNODE(0).eval(std::get<Bc>(boundComps)...,cs...);		\
    }
    
    PROVIDE_EVAL(const);
    
    PROVIDE_EVAL(/*non const*/);
    
#undef PROVIDE_EVAL
    
    /// Construct
    template <typename T>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    CompsBinder(T&& arg,
		const BoundComps& boundComps) :
      SubNodes<_E...>(std::forward<T>(arg)),
      boundComps(boundComps)
    {
    }
  };
  
  /// Binds a subset of components
  template <typename _E,
	    typename...BCs,
	    ENABLE_THIS_TEMPLATE_IF(isNode<_E>)>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
  auto bindComps(_E&& e,
		 const CompsList<BCs...>& bc)
  {
    /// Base passed type
    using E=
      std::decay_t<_E>;
    
    /// Type returned when evaluating the expression
    using Fund=
      typename E::Fund;
    
    /// Components to bind
    using BoundComps=
      CompsList<BCs...>;
    
    /// Visible components
    using Comps=
      TupleFilterAllTypes<typename E::Comps,
			  BoundComps>;
    
    return
      CompsBinder<BoundComps,
		  std::tuple<decltype(e)>,
		  Comps,
		  Fund>(std::forward<_E>(e),bc);
  }
  
  // /// Rebind an already bound expression
  // template <typename CB,
  // 	    typename...BCs>
  // HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
  // auto compBind(const CompBinderFeat<CB>& cb,
  // 		const CompsList<BCs...>& bcs)
  // {
  //   return
  //     compBind(cb.defeat().nestedExpression,
  // 	       std::tuple_cat(cb.deFeat().boundComps,bcs));
  // }
}

#endif
