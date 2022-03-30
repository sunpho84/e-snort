#ifndef _PROD_HPP
#define _PROD_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file prod.hpp

#include <expr/comps.hpp>
#include <expr/conj.hpp>
#include <expr/expr.hpp>
#include <expr/prodCompsDeducer.hpp>

namespace esnort
{
  PROVIDE_DETECTABLE_AS(Producer);
  
  /// Producer
  ///
  /// Forward declaration to capture the components
  template <typename _Ccs,
	    typename _E1,
	    typename _E2,
	    typename _Comps,
	    typename _Fund>
  struct Producer;
  
#define THIS					\
  Producer<CompsList<Cc...>,_E1,_E2,CompsList<C...>,_Fund>
  
#define BASE					\
    Expr<THIS>
  
  /// Producer
  ///
  template <typename...Cc,
	    typename _E1,
	    typename _E2,
	    typename...C,
	    typename _Fund>
  struct THIS :
    DynamicCompsProvider<CompsList<C...>>,
    DetectableAsProducer,
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
    
    /// Contracted components
    using ContractedComps=
      CompsList<Cc...>;
    
    /// Fundamental tye
    using Fund=_Fund;
    
    /// Type of the first factor
    using Fact1Expr=
      std::decay_t<_E1>;
    
    /// Type of the first factor
    using Fact2Expr=
      std::decay_t<_E2>;
    
    /// Executes where allocated
    static constexpr ExecSpace execSpace=
      (Fact1Expr::execSpace==ExecSpace::UNDEFINED)?
      Fact2Expr::execSpace:
      Fact1Expr::execSpace;
    
    static_assert(not (execSpace==ExecSpace::UNDEFINED),"Cannot define product in case the two execution spaces are both undefined");

    /// List of dynamic comps
    using DynamicComps=
      typename DynamicCompsProvider<Comps>::DynamicComps;
    
    /// Sizes of the dynamic components
    const DynamicComps dynamicSizes;
    
    /// Returns the dynamic sizes
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
    
    /// States whether the tensor can be simdified
    static constexpr bool canSimdify=
      false;
    /// \todo improve
    
    /// Components on which simdifying
    using SimdifyingComp=void;
    
    /// First factor
    ExprRefOrVal<_E1> fact1Expr;
    
    /// Second factor
    ExprRefOrVal<_E2> fact2Expr;
    
#define PROVIDE_SIMDIFY(ATTRIB)					\
    /*! Returns a ATTRIB simdified view */			\
    INLINE_FUNCTION						\
    auto simdify() ATTRIB					\
    {								\
      CRASH<<"";						\
    }
    
    PROVIDE_SIMDIFY(const);
    
    PROVIDE_SIMDIFY(/* non const */);
    
#undef PROVIDE_SIMDIFY
    
#define PROVIDE_GET_REF(ATTRIB)					\
    /*! Returns a reference */					\
    INLINE_FUNCTION						\
    auto getRef() ATTRIB					\
    {								\
      return fact1Expr.getRef()*fact2Expr.getRef();			\
    }
    
    PROVIDE_GET_REF(const);
    
    PROVIDE_GET_REF(/* non const */);
    
#undef PROVIDE_GET_REF
    
    /// Evaluate
    template <typename...NCcs> // Non contracted components
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Fund eval(const NCcs&...nccs) const
    {
      /// Detect complex product
      constexpr bool isComplProd=
	tupleHasType<Comps,ComplId>;
      
      Fund res=0;
      
      loopOnAllComps<ContractedComps>(dynamicSizes,[&res](const auto&...c) INLINE_ATTRIBUTE
      {
	res+=0.0;
      },nccs...);
      
      return res;
    }
    
    /// Construct
    template <typename T1,
	      typename T2>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Producer(T1&& fact1Expr,
	     T2&& fact2Expr,
	     const DynamicComps& dynamicSizes,
	     UNIVERSAL_CONSTRUCTOR_IDENTIFIER) :
      dynamicSizes(dynamicSizes),
      fact1Expr(std::forward<T1>(fact1Expr)),
      fact2Expr(std::forward<T2>(fact2Expr))
    {
    }
  };
  
  template <typename _E1,
	    typename _E2,
	    ENABLE_THIS_TEMPLATE_IF(isExpr<_E1> and isExpr<_E2>)>
  auto prod(_E1&& e1,
	    _E2&& e2)
  {
    using E1=std::decay_t<_E1>;
    
    using E2=std::decay_t<_E2>;
    
    /// Computes the product components
    using PCC=
      ProdCompsDeducer<typename E1::Comps,typename E2::Comps>;
    
    /// Gets the visible comps
    using VisibleComps=
      typename PCC::VisibleComps;
    
    /// Gets the contracted comps
    using ContractedComps=
      typename PCC::ContractedComps;
    
    /// Determine the fundamental type of the product
    using Fund=
      decltype(typename E1::Fund{}*typename E2::Fund{});
    
    /// Resulting type
    using Res=
      Producer<ContractedComps,
	       decltype(e1),
	       decltype(e2),
	       VisibleComps,
	       Fund>;
    
    /// Resulting dynamic components
    const auto& dc=
      dynamicCompsCombiner<typename Res::DynamicComps>(std::make_tuple(e1.getDynamicSizes(),e2.getDynamicSizes()));
    
    return
      Res(std::forward<_E1>(e1),
	  std::forward<_E2>(e2),
	  dc,UNIVERSAL_CONSTRUCTOR_CALL);
  }
  
  /// Catch the product operator
  template <typename E1,
	    typename E2,
	    ENABLE_THIS_TEMPLATE_IF(isExpr<E1> and isExpr<E2>)>
  auto operator*(E1&& e1,
		 E2&& e2)
  {
    return
      prod(std::forward<E1>(e1),std::forward<E2>(e2));
  }
}

#endif
