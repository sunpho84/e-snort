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
	    typename _E,
	    typename _Comps,
	    typename _Fund>
  struct Producer;
  
#define THIS					\
  Producer<CompsList<Cc...>,std::tuple<_E...>,CompsList<C...>,_Fund>
  
#define BASE					\
    Expr<THIS>
  
  /// Producer
  ///
  template <typename...Cc,
	    typename..._E,
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
    
    static_assert(sizeof...(_E)==2,"Expecting 2 factors");
    
    /// Components
    using Comps=
      CompsList<C...>;
    
    /// Contracted components
    using ContractedComps=
      CompsList<Cc...>;
    
    /// Fundamental tye
    using Fund=_Fund;
    
    /// Executes where allocated
    static constexpr ExecSpace execSpace=
      ExecSpace::UNDEFINED;
      
#warning    static_assert(not (execSpace==ExecSpace::UNDEFINED),"Cannot define product in case the two execution spaces are both undefined");
    
    template <int I>
    using FactExpr=
      std::decay_t<std::tuple_element_t<I,std::tuple<_E...>>>;
    
    /// Detect complex product
    static constexpr bool isComplProd=
      (tupleHasType<typename std::decay_t<_E>::Comps,ComplId> and...);
    
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
    
    static constexpr bool simdifyCase()
    {
      using S0=typename FactExpr<0>::SimdifyingComp;
      using S1=typename FactExpr<1>::SimdifyingComp;
      
      constexpr bool c0=FactExpr<0>::canSimdify and not tupleHasType<ContractedComps,Transp<S0>> and not (isComplProd and std::is_same_v<ComplId,S0>);
      constexpr bool c1=FactExpr<1>::canSimdify and not tupleHasType<ContractedComps,S1> and not (isComplProd and std::is_same_v<ComplId,S1>);
      
      return c0 and c1 and std::is_same_v<S0,S1>;
    }
    
    /// States whether the tensor can be simdified
    static constexpr bool canSimdify=
      simdifyCase();
    
    /// \todo improve
    
    /// Components on which simdifying
    using SimdifyingComp=typename FactExpr<0>::SimdifyingComp;
    
    /// First factor
    std::tuple<ExprRefOrVal<_E>...> factExprs;
    
    template <int I>
    INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB
    decltype(auto) factExpr() const
    {
      return std::get<I>(factExprs);
    }
    
#define PROVIDE_SIMDIFY(ATTRIB)					\
    /*! Returns a ATTRIB simdified view */			\
    INLINE_FUNCTION						\
    auto simdify() ATTRIB					\
    {								\
      return							\
	factExpr<0>().simdify()*				\
	factExpr<1>().simdify();				\
    }
    
    PROVIDE_SIMDIFY(const);
    
    PROVIDE_SIMDIFY(/* non const */);
    
#undef PROVIDE_SIMDIFY
    
#define PROVIDE_GET_REF(ATTRIB)					\
    /*! Returns a reference */					\
    INLINE_FUNCTION						\
    auto getRef() ATTRIB					\
    {								\
      return factExpr<0>().getRef()*				\
	factExpr<1>().getRef();					\
    }
    
    PROVIDE_GET_REF(const);
    
    PROVIDE_GET_REF(/* non const */);
    
#undef PROVIDE_GET_REF
    
    template <int I,
	      typename FC,
	      typename...NCcs>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    static auto getCompsForFact(const CompsList<NCcs...>& nccs)
    {
      using FreeC=TupleFilterAllTypes<typename FactExpr<I>::Comps,FC>;
      
      return tupleGetSubset<FreeC>(nccs);
    }
    
    /// Evaluate
    template <typename...NCcs> // Non contracted components
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Fund eval(const NCcs&..._nccs) const
    {
      const auto nccs=std::make_tuple(_nccs...);
      
      const auto fc0=getCompsForFact<0,CompsList<ComplId,Transp<Cc>...>>(nccs);
      const auto fc1=getCompsForFact<1,CompsList<ComplId,Cc...>>(nccs);
      
      Fund res=0.0;
      
      loopOnAllComps<ContractedComps>(dynamicSizes,[this,&nccs,&res,fc0,fc1,_nccs...](const auto&...c) INLINE_ATTRIBUTE
      {
	// Disable warning
	[](auto...){}(_nccs...);
	
	auto e0=[this,&fc0,&c...](const auto&...extra) INLINE_ATTRIBUTE
	{
	  return std::apply(factExpr<0>(),std::tuple_cat(fc0,std::make_tuple(c.transp()...,extra...)));
	};
	
	auto e1=[this,&fc1,&c...](const auto&...extra) INLINE_ATTRIBUTE
	{
	  return std::apply(factExpr<1>(),std::tuple_cat(fc1,std::make_tuple(c...,extra...)));
	};
	
	if constexpr(isComplProd)
	  {
	    const auto& reIm=std::get<ComplId>(nccs);
	    
	    if(reIm==Re)
	      {
		res+=e0(Re)*e1(Re);
		res-=e0(Im)*e1(Im);
	      }
	    else
	      {
		res+=e0(Re)*e1(Im);
		res+=e0(Im)*e1(Re);
	      }
	  }
	else
	  res+=e0()*e1();
      });
      
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
      factExprs({fact1Expr,fact2Expr})
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
	       std::tuple<decltype(e1),decltype(e2)>,
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
