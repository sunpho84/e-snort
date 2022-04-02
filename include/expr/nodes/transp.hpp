#ifndef _TRANSP_HPP
#define _TRANSP_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/transp.hpp

#include <expr/comps/comp.hpp>
#include <expr/comps/comps.hpp>
#include <expr/nodes/node.hpp>
#include <expr/nodes/subNodes.hpp>

namespace esnort
{
  PROVIDE_DETECTABLE_AS(Transposer);
  
  /// Transposer
  ///
  /// Forward declaration to capture the components
  template <typename _E,
	    typename _Comps,
	    typename _Fund>
  struct Transposer;
  
#define THIS					\
  Transposer<std::tuple<_E...>,CompsList<C...>,_Fund>
  
#define BASE					\
    Node<THIS>
  
  /// Transposer
  ///
  template <typename..._E,
	    typename...C,
	    typename _Fund>
  struct THIS :
    DynamicCompsProvider<CompsList<C...>>,
    DetectableAsTransposer,
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
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    decltype(auto) getDynamicSizes() const
    {
      return SUBNODE(0).getDynamicSizes();
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
    
    /// Return whether can be assigned at compile time
    static constexpr bool canAssignAtCompileTime=
      SubNode<0>::canAssignAtCompileTime;
    
    /// States whether the tensor can be simdified
    static constexpr bool canSimdify=
      SubNode<0>::canSimdify;
    
    /// Components on which simdifying
    using SimdifyingComp=
      Transp<typename SubNode<0>::SimdifyingComp>;
    
#define PROVIDE_SIMDIFY(ATTRIB)					\
    /*! Returns a ATTRIB simdified view */			\
    INLINE_FUNCTION						\
    auto simdify() ATTRIB					\
    {								\
      return transp(SUBNODE(0).simdify());			\
    }
    
    PROVIDE_SIMDIFY(const);
    
    PROVIDE_SIMDIFY(/* non const */);
    
#undef PROVIDE_SIMDIFY
    
#define PROVIDE_GET_REF(ATTRIB)					\
    /*! Returns a reference */					\
    INLINE_FUNCTION						\
    auto getRef() ATTRIB					\
    {								\
      return transp(SUBNODE(0).getRef());			\
    }
    
    PROVIDE_GET_REF(const);
    
    PROVIDE_GET_REF(/* non const */);
    
#undef PROVIDE_GET_REF
    
    /// Evaluate
    template <typename...TD>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Fund eval(const TD&...td) const
    {
      return SUBNODE(0)(transp(td)...);
    }
    
    /// Construct
    template <typename T>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Transposer(T&& arg,
	       UNIVERSAL_CONSTRUCTOR_IDENTIFIER) :
      SubNodes<_E...>(std::forward<T>(arg))
    {
    }
  };
  
  /// Transpose an expression
  template <typename _E,
	    ENABLE_THIS_TEMPLATE_IF(isNode<_E>)>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
  decltype(auto) transp(_E&& e)
  {
#if 0
    LOGGER<<"Now inside transp";
#endif
    
    /// Base passed type
    using E=
      std::decay_t<_E>;
    
    if constexpr(isTransposer<E>)
      return e.template subNode<0>;
    else
      {
	/// Components
	using Comps=
	  TranspMatrixTensorComps<typename E::Comps>;
	
	if constexpr(not compsAreTransposable<Comps>)
	  {
#if 0
	    LOGGER<<"no need to transpose, returning the argument, which is "<<&e<<" "<<demangle(typeid(_E).name())<<(std::is_lvalue_reference_v<decltype(e)>?"&":(std::is_rvalue_reference_v<decltype(e)>?"&&":""));
#endif
	    
	    return RemoveRValueReference<_E>(e);
	  }
	else
	  {
	    /// Type returned when evaluating the expression
	    using Fund=
	      typename E::Fund;
	    
	    return
	      Transposer<_E,Comps,Fund>(std::forward<_E>(e),
					UNIVERSAL_CONSTRUCTOR_CALL);
	  }
      }
  }
}

#endif
