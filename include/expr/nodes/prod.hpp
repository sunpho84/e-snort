#ifndef _PROD_HPP
#define _PROD_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/prod.hpp

#include <expr/comps/comps.hpp>
#include <expr/nodes/conj.hpp>
#include <expr/nodes/node.hpp>
#include <expr/comps/prodCompsDeducer.hpp>
#include <expr/nodes/producerDeclaration.hpp>
#include <expr/nodes/subNodes.hpp>
#include <metaprogramming/arithmeticTraits.hpp>
#include <metaprogramming/asConstexpr.hpp>

namespace grill
{
#define THIS					\
  Producer<CompsList<Cc...>,std::tuple<_E...>,CompsList<C...>,_Fund,std::integer_sequence<int,Is...>>
  
#define BASE					\
  Node<THIS>
  
  /// Producer
  template <typename...Cc,
	    typename..._E,
	    typename...C,
	    typename _Fund,
	    int...Is>
  struct THIS :
    DynamicCompsProvider<CompsList<C...>>,
    DetectableAsProducer,
    SubNodes<_E...>,
    BASE
  {
    /// Import the base expression
    using Base=BASE;
    
    using This=THIS;
    
#undef BASE
    
#undef THIS
    
    static_assert(sizeof...(_E)==2,"Expecting 2 factors");
    
    IMPORT_SUBNODE_TYPES;
    
    /// Components
    using Comps=
      CompsList<C...>;
    
    /// Contracted components
    using ContractedComps=
      CompsList<Cc...>;
    
    /// Fundamental tye
    using Fund=_Fund;
    
    /// Execution space
    static constexpr ExecSpace execSpace=
      commonExecSpace<std::remove_reference_t<_E>::execSpace...>();
    
    static_assert(execSpace!=ExecSpace::HOST_DEVICE,"Cannot define product in undefined exec space");
    
    /// Detect complex product
    static constexpr bool isComplProd=
      (tupleHasType<typename std::decay_t<_E>::Comps,ComplId> and...);
    
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
    
    template <int I>
    using SimdifyingCompOfSubNode=
      typename SubNode<I>::SimdifyingComp;
    
    /// Determine if product can be simdified - to be extended
    static constexpr bool simdifyCase()
    {
      return
	(SubNode<Is>::canSimdify and...) and
	((not tupleHasType<ContractedCompsForFact<Is>,SimdifyingCompOfSubNode<Is>>) and...) and
	std::is_same_v<SimdifyingCompOfSubNode<Is>...> and
	not (isComplProd and (std::is_same_v<ComplId,SimdifyingCompOfSubNode<Is>> or...));
    }
    
    /// States whether the operations can be simdified
    static constexpr bool canSimdify=
      simdifyCase();
    
    /// \todo improve
    
    /// Components on which simdifying
    using SimdifyingComp=
      typename SubNode<0>::SimdifyingComp;
    
#define PROVIDE_SIMDIFY(ATTRIB)					\
    /*! Returns a ATTRIB simdified view */			\
    INLINE_FUNCTION						\
    auto simdify() ATTRIB					\
    {								\
      return							\
	(SUBNODE(Is).simdify()*...);				\
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
	(SUBNODE(Is).getRef()*...);				\
    }
    
    PROVIDE_GET_REF(const);
    
    PROVIDE_GET_REF(/* non const */);
    
#undef PROVIDE_GET_REF
    
    template <int I>
    using ContractedCompsForFact=
      std::conditional_t<(I==0),
			 CompsList<Transp<Cc>...>,
			 CompsList<Cc...>>;
    
    /// Gets the components for the I-th factor
    template <int I,
	      typename FC,
	      typename...NCcs>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    static auto getCompsForFact(const CompsList<NCcs...>& nccs)
    {
      using FreeC=TupleFilterAllTypes<typename SubNode<I>::Comps,FC>;
      
      return tupleGetSubset<FreeC>(nccs);
    }
    
    /// Evaluate
    template <typename...NCcs> // Non contracted components
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Fund eval(const NCcs&..._nccs) const
    {
      const auto allNccs=
	std::make_tuple(_nccs...);
      
      using MaybeComplId=
	std::conditional_t<isComplProd,CompsList<ComplId>,CompsList<>>;
      
      Fund res;
      setToZero(res);
      
      loopOnAllComps<ContractedComps>(dynamicSizes,[this,&allNccs,&res](const auto&..._ccs) INLINE_ATTRIBUTE
      {
	auto ccs2=
	  std::make_tuple(std::make_tuple(_ccs.transp()...),
			  std::make_tuple(_ccs...));
	
	/// Gets the evaluator for a given subnode
	auto getSubNodeEvaluer=
	  [this,&allNccs,&ccs2](auto i) INLINE_ATTRIBUTE
	  {
	    constexpr int I=i();
	    
	    return
	      [this,&allNccs,&ccs=std::get<I>(ccs2)](const auto&...maybeReIm) INLINE_ATTRIBUTE
	      {
		/// Put together the comonents to be removed
		using CompsToRemove=
		  TupleCat<ContractedCompsForFact<I>,MaybeComplId>;
		
		/// Non contracted components
		auto nccs=
		  getCompsForFact<I,CompsToRemove>(allNccs);
		
		/// Result
		const auto res=
		  std::apply(SUBNODE(I),std::tuple_cat(ccs,nccs,std::make_tuple(maybeReIm...)));
		
		return res;
	      };
	  };
	
	/// Takes the two evaluators
	auto [e0,e1]=
	  std::make_tuple(getSubNodeEvaluer(asConstexpr<Is>)...);
	
	if constexpr(isComplProd)
	  {
	    const auto& reIm=std::get<ComplId>(allNccs);
	    
	    if(reIm==Re)
	      {
		sumAssignTheProd(res,e0(Re),e1(Re));
		subAssignTheProd(res,e0(Im),e1(Im));
	      }
	    else
	      {
		sumAssignTheProd(res,e0(Re),e1(Im));
		sumAssignTheProd(res,e0(Im),e1(Re));
	      }
	  }
	else
	  sumAssignTheProd(res,e0(),e1());
      });
      
      return res;
    }
    
    /// Construct
    template <typename...T>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Producer(const DynamicComps& dynamicSizes,
	     UNIVERSAL_CONSTRUCTOR_IDENTIFIER,
	     T&&...facts) :
      SubNodes<_E...>(facts...),
      dynamicSizes(dynamicSizes)
    {
    }
  };
  
  template <typename..._E,
	    ENABLE_THIS_TEMPLATE_IF(isNode<_E> and...)>
  INLINE_FUNCTION HOST_DEVICE_ATTRIB
  auto prod(_E&&...e)
  {
    /// Computes the product components
    using PCC=
      ProdCompsDeducer<typename std::decay_t<_E>::Comps...>;
    
    /// Gets the visible comps
    using VisibleComps=
      typename PCC::VisibleComps;
    
    /// Gets the contracted comps
    using ContractedComps=
      typename PCC::ContractedComps;
    
    /// Determine the fundamental type of the product
    using Fund=
      decltype((typename std::decay_t<_E>::Fund{}*...));
    
    /// Resulting type
    using Res=
      Producer<ContractedComps,
	       std::tuple<decltype(e)...>,
	       VisibleComps,
	       Fund>;
    
    /// Resulting dynamic components
    const auto dc=
      dynamicCompsCombiner<typename Res::DynamicComps>(std::make_tuple(e.getDynamicSizes()...));
    
    return
      Res(dc,UNIVERSAL_CONSTRUCTOR_CALL,std::forward<_E>(e)...);
  }
  
  /// Catch the product operator
  template <typename E1,
	    typename E2,
	    ENABLE_THIS_TEMPLATE_IF(isNode<E1> and isNode<E2>)>
  INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB
  auto operator*(E1&& e1,
		 E2&& e2)
  {
    return
      prod(std::forward<E1>(e1),std::forward<E2>(e2));
  }
}

#endif
