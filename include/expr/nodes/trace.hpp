#ifndef _TRACE_HPP
#define _TRACE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/trace.hpp

#include <expr/comps/comps.hpp>
#include <expr/comps/tracerCompsDeducer.hpp>
#include <expr/comps/dynamicCompsProvider.hpp>
#include <expr/nodes/subNodes.hpp>
#include <expr/nodes/node.hpp>
#include <metaprogramming/arithmeticOperatorsViaCast.hpp>

namespace grill
{
  PROVIDE_DETECTABLE_AS(Tracer);
  
  /// Tracer
  ///
  /// Forward declaration to capture the components
  template <typename Tc,
	    typename _E,
	    typename _Comps,
	    typename _Fund>
  struct Tracer;
  
#define THIS					\
  Tracer<CompsList<Tc...>,std::tuple<_E...>,CompsList<C...>,_Fund>
  
#define BASE					\
  Node<THIS>
  
  /// Tracer
  ///
  template <typename...Tc,
	    typename..._E,
	    typename...C,
	    typename _Fund>
  struct THIS :
    DynamicCompsProvider<CompsList<C...>>,
    DetectableAsTracer,
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
    using TracedComps=
      CompsList<Tc...>;
    
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
    
    /// This is a lightweight object
    static constexpr bool storeByRef=false;
    
    /// Import assignment operator
    using Base::operator=;
    
    /// Return whether can be assigned at compile time
    static constexpr bool canAssignAtCompileTime=false;
    
    /// States whether the tensor can be simdified
    static constexpr bool canSimdify=
      SubNode<0>::canSimdify and not tupleHasType<TracedComps,typename SubNode<0>::SimdifyingComp>;
    
    /// Components on which simdifying
    using SimdifyingComp=
      typename SubNode<0>::SimdifyingComp;
    
    /////////////////////////////////////////////////////////////////
    
#define PROVIDE_SIMDIFY(ATTRIB)					\
    /*! Returns a ATTRIB simdified view */			\
    INLINE_FUNCTION						\
    auto simdify() ATTRIB					\
    {								\
      return trace(SUBNODE(0).simdify());			\
    }
    
    PROVIDE_SIMDIFY(const);
    
    PROVIDE_SIMDIFY(/* non const */);
    
#undef PROVIDE_SIMDIFY
    
    /////////////////////////////////////////////////////////////////
    
    //// Returns a tracer on a different expression
    template <typename T>
    INLINE_FUNCTION
    decltype(auto) recreateFromExprs(T&& t) const
    {
      return trace(std::forward<T>(t));
    }
    
    /////////////////////////////////////////////////////////////////
    
#define PROVIDE_GET_REF(ATTRIB)					\
    /*! Returns a reference */					\
    INLINE_FUNCTION						\
    auto getRef() ATTRIB					\
    {								\
      return trace(SUBNODE(0).getRef());			\
    }
    
    PROVIDE_GET_REF(const);
    
    PROVIDE_GET_REF(/* non const */);
    
#undef PROVIDE_GET_REF
    
    /////////////////////////////////////////////////////////////////
    
    /// Evaluate
    template <typename...NTc>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Fund eval(const NTc&...nTCs) const
    {
      /// Result
      Fund res;
      setToZero(res);
      
      loopOnAllComps<TracedComps>(getDynamicSizes(),
				  [this,&res,&nTCs...](const auto&...tCs) INLINE_ATTRIBUTE
				  {
				    /// First argument
				    res+=
				      this->template subNode<0>()(nTCs...,tCs...,transp(tCs)...);
				  });
      
      return
	res;
    }
    
    /// Construct
    template <typename T,
	      ENABLE_THIS_TEMPLATE_IF(not isTracer<T>)>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Tracer(T&& arg) :
      SubNodes<_E...>(std::forward<T>(arg))
    {
    }
    
    /// Copy constructor
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Tracer(const Tracer& oth) :
      SubNodes<_E...>(oth.template subNode<0>())
    {
    }
  };
  
  /// Trace an expression
  template <typename _E,
	    ENABLE_THIS_TEMPLATE_IF(isNode<_E>)>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
  decltype(auto) trace(_E&& e)
  {
    /// Base passed type
    using E=
      std::decay_t<_E>;
    
    using CompsDeducer=
      TracerCompsDeducer<typename E::Comps>;
    
    using TracedComps=typename CompsDeducer::TracedComps;
    
    using Comps=typename CompsDeducer::VisibleComps;
    
    using Fund=typename E::Fund;
    
    return
      Tracer<TracedComps,std::tuple<_E>,Comps,Fund>(std::forward<_E>(e));
  }
}

#endif
