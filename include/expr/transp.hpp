#ifndef _TRANSP_HPP
#define _TRANSP_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/transp.hpp

#include <expr/comp.hpp>
#include <expr/comps.hpp>
#include <expr/expr.hpp>

namespace esnort
{
  /// Transposer
  ///
  /// Forward declaration to capture the components
  template <typename _Te,
	    typename _Comps,
	    typename _Fund>
  struct Transposer;
  
#define THIS					\
  Transposer<_Te,CompsList<C...>,_Fund>

#define BASE					\
    Expr<THIS>
  
  /// Component binder
  ///
  template <typename _Te,
	    typename...C,
	    typename _Fund>
  struct THIS :
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
    
    /// Type of the tranposed expression
    using TranspExpr=
      std::decay_t<_Te>;
    
    /// Executes where allocated
    static constexpr ExecSpace execSpace=
      TranspExpr::execSpace;
    
    /// Returns the dynamic sizes
    decltype(auto) getDynamicSizes() const
    {
      return transpExpr.getDynamicSizes();
    }
    
    /// Returns whether can assign
    INLINE_FUNCTION
    bool canAssign()
    {
      return transpExpr.canAssign();
    }
    
    /// This is a lightweight object
    static constexpr bool storeByRef=false;
    
    /// Import assignment operator
    using Base::operator=;
    
    /// Return whether can be assigned at compile time
    static constexpr bool canAssignAtCompileTime=
      TranspExpr::canAssignAtCompileTime;
    
    /// States whether the tensor can be simdified
    static constexpr bool canSimdify=
      TranspExpr::canSimdify;
    
    /// Components on which simdifying
    using SimdifyingComp=
      typename TranspExpr::SimdifyingComp;
    
    /// Expression that has been transposed
    ExprRefOrVal<_Te> transpExpr;
    
#define PROVIDE_SIMDIFY(ATTRIB)					\
    /*! Returns a ATTRIB simdified view */			\
    INLINE_FUNCTION						\
    auto simdify() ATTRIB					\
    {								\
      return transp(transpExpr.simdify());			\
    }
    
    PROVIDE_SIMDIFY(const);
    
    PROVIDE_SIMDIFY(/* non const */);
    
#undef PROVIDE_SIMDIFY

#define PROVIDE_GET_REF(ATTRIB)					\
    /*! Returns a reference */					\
    INLINE_FUNCTION						\
    auto getRef() ATTRIB					\
    {								\
      return transp(transpExpr.getRef());			\
    }
    
    PROVIDE_GET_REF(const);
    
    PROVIDE_GET_REF(/* non const */);
    
#undef PROVIDE_GET_REF
    
    /// Evaluate
    template <typename...TD>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Fund eval(const TD&...td) const
    {
      return transpExpr(transp(td)...);
    }
    
    /// Construct
    template <typename T>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Transposer(T&& transpExpr,
	       UNIVERSAL_CONSTRUCTOR_IDENTIFIER) :
      transpExpr(std::forward<T>(transpExpr))
    {
    }
  };
  
  /// Recognizes a transposer
  ///
  /// Default case
  template <typename T>
  inline
  constexpr bool isTransposer=false;
  
  /// Recognizes an actual transposer
  template <typename _Te,
	    typename...C,
	    typename _Fund>
  inline
  constexpr bool isTransposer<Transposer<_Te,CompsList<C...>,_Fund>> =true;
  
  /// Transpose an expression
  template <typename _E,
	    ENABLE_THIS_TEMPLATE_IF(isExpr<_E>)>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
  decltype(auto) transp(_E&& e)
  {
    /// Base passed type
    using E=
      std::decay_t<_E>;
    
    if constexpr(isTransposer<E>)
      return e.transpExpr;
    else
      {
	/// Components
	using Comps=
	  TranspMatrixTensorComps<typename E::Comps>;
	
	if constexpr(not compsAreTransposable<Comps>)
	  return e;
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
