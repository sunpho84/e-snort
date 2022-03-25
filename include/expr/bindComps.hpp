#ifndef _BINDCOMPS_HPP
#define _BINDCOMPS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file bindComps.hpp

#include <expr/comps.hpp>
#include <expr/dynamicCompsProvider.hpp>
#include <expr/executionSpace.hpp>
#include <expr/exprDeclaration.hpp>
#include <expr/exprRefOrVal.hpp>
#include <metaprogramming/universalReference.hpp>
#include <tuples/tupleFilter.hpp>
#include "tuples/tupleHasType.hpp"

namespace esnort
{
  /// Component binder
  ///
  /// Forward declaration to capture the index sequence
  template <typename _BC,
	    typename _Be,
	    typename _Comps,
	    typename _Fund>
  struct CompsBinder;
  
#define THIS					\
  CompsBinder<CompsList<Bc...>,_Be,CompsList<C...>,_Fund>

#define BASE					\
    Expr<THIS>
  
  /// Component binder
  ///
  template <typename...Bc,
	    typename _Be,
	    typename...C,
	    typename _Fund>
  struct THIS :
    DynamicCompsProvider<CompsList<C...>>,
    BASE
  {
    /// Import the base expression
    using Base=BASE;
    
    using This=THIS;
    
#undef BASE
    
#undef THIS
    
    /// Components
    using Comps=
      CompsList<C...>;
    
    /// Fundamental tye
    using Fund=_Fund;
    
    /// Executes where allocated
    static constexpr ExecSpace execSpace=
      std::decay_t<_Be>::execSpace;
    
    /// Returns the dynamic sizes
    const auto getDynamicSizes() const
    {
      return tupleGetSubset<typename CompsBinder::DynamicComps>(boundExpr.getDynamicSizes());
    }
    
    /// Returns whether can assign
    INLINE_FUNCTION
    bool canAssign()
    {
      return boundExpr.canAssign();
    }
    
    /// Import assignment operator
    using Base::operator=;
    
    /// Type of the bound expression
    using BoundExpr=
      std::remove_reference_t<_Be>;
    
    /// Bound components
    using BoundComps=
      CompsList<Bc...>;
    
    /// Return whether can be assigned at compile time
    static constexpr bool canAssignAtCompileTime=
      std::decay_t<BoundExpr>::canAssignAtCompileTime;
    
    /// States whether the tensor can be simdified
    static constexpr bool canSimdify=
      BoundExpr::canSimdify and
      not tupleHasType<BoundComps,typename BoundExpr::SimdifyingComp>;
    
    /// Components on which simdifying
    using SimdifyingComp=
      std::conditional_t<canSimdify,typename BoundExpr::SimdifyingComp,void>;
    
    /// Returns a const simdified view
    INLINE_FUNCTION
    auto simdify() const
    {
      return bindComps(boundExpr.simdify(),boundComps);
    }
    
    /// Returns a simdified view
    INLINE_FUNCTION
    auto simdify()
    {
      return bindComps(boundExpr.simdify(),boundComps);
    }
    
    /// Components that have been bound
    const BoundComps boundComps;
    
    /// Expression that has been bound
    ExprRefOrVal<_Be> boundExpr;
    
#define PROVIDE_EVAL(ATTRIB)						\
    template <typename...U>						\
    HOST_DEVICE_ATTRIB constexpr INLINE_FUNCTION			\
    decltype(auto) eval(const U&...cs) ATTRIB				\
    {									\
      return								\
	this->boundExpr.eval(std::get<Bc>(boundComps)...,cs...);	\
    }
    
    PROVIDE_EVAL(const);
    
    PROVIDE_EVAL(/*non const*/);
    
#undef PROVIDE_EVAL
    
    /// Construct
    template <typename T>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    CompsBinder(T&& boundExpr,
		const BoundComps& boundComps) :
      boundComps(boundComps),
      boundExpr(std::forward<T>(boundExpr))
    {
    }
  };
  
  /// Binds a subset of components
  template <typename _E,
	    typename...BCs>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
  auto bindComps(_E&& e,
		const CompsList<BCs...>& bc,
		UNPRIORITIZE_UNIVERSAL_REFERENCE_CONSTRUCTOR)
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
		 decltype(e),
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
