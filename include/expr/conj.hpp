#ifndef _CONJ_HPP
#define _CONJ_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/conj.hpp

#include <expr/comp.hpp>
#include <expr/comps.hpp>
#include <expr/expr.hpp>

namespace esnort
{
  DEFINE_UNTRANSPOSABLE_COMP(ComplId,int,2);
  
  /// Conjugator
  ///
  /// Forward declaration to capture the components
  template <typename _Ce,
	    typename _Comps,
	    typename _Fund>
  struct Conjugator;
  
#define THIS					\
  Conjugator<_Ce,CompsList<C...>,_Fund>

#define BASE					\
    Expr<THIS>
  
  /// Component binder
  ///
  template <typename _Ce,
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
    
    /// Type of the conjugated expression
    using ConjugExpr=
      std::decay_t<_Ce>;
    
    /// Executes where allocated
    static constexpr ExecSpace execSpace=
      ConjugExpr::execSpace;
    
    /// Returns the dynamic sizes
    decltype(auto) getDynamicSizes() const
    {
      return conjugExpr.getDynamicSizes();
    }
    
    /// Returns whether can assign
    INLINE_FUNCTION
    bool canAssign()
    {
      return conjugExpr.canAssign();
    }
    
    /// This is a lightweight object
    static constexpr bool storeByRef=false;
    
    /// Import assignment operator
    using Base::operator=;
    
    /// Return whether can be assigned at compile time
    static constexpr bool canAssignAtCompileTime=
      ConjugExpr::canAssignAtCompileTime;
    
    /// States whether the tensor can be simdified
    static constexpr bool canSimdify=
      ConjugExpr::canSimdify and
      not std::is_same_v<ComplId,typename ConjugExpr::SimdifyingComp>;
    
    /// Components on which simdifying
    using SimdifyingComp=
      std::conditional_t<canSimdify,typename ConjugExpr::SimdifyingComp,void>;
    
    /// Expression that has been conjugated
    ExprRefOrVal<_Ce> conjugExpr;
    
#define PROVIDE_SIMDIFY(ATTRIB)					\
    /*! Returns a ATTRIB simdified view */			\
    INLINE_FUNCTION						\
    auto simdify() ATTRIB					\
    {								\
      return conj(conjugExpr.simdify());			\
    }
    
    PROVIDE_SIMDIFY(const);
    
    PROVIDE_SIMDIFY(/* non const */);
    
#undef PROVIDE_SIMDIFY

#define PROVIDE_GET_REF(ATTRIB)					\
    /*! Returns a reference */					\
    INLINE_FUNCTION						\
    auto getRef() ATTRIB					\
    {								\
      return conj(conjugExpr.getRef());				\
    }
    
    PROVIDE_GET_REF(const);
    
    PROVIDE_GET_REF(/* non const */);
    
#undef PROVIDE_GET_REF
    
    /// Evaluate
    template <typename...TD>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Fund eval(const TD&...td) const
    {
      /// Compute the real or imaginary component
      const ComplId& reIm=
	std::get<ComplId>(std::make_tuple(td...));
      
      /// Nested result
      decltype(auto) nestedRes=
	this->conjugExpr(td...);
      
      if(reIm==0)
	return nestedRes;
      else
	return -nestedRes;
    }
    
    /// Construct
    template <typename T>
    HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
    Conjugator(T&& conjugExpr) :
      conjugExpr(std::forward<T>(conjugExpr))
    {
    }
  };
  
  /// Recognizes a conjugator
  ///
  /// Default case
  template <typename T>
  inline
  constexpr bool isConjugator=false;
  
  /// Recognizes an actual conjugator
  template <typename _Ce,
	    typename...C,
	    typename _Fund>
  inline
  constexpr bool isConjugator<Conjugator<_Ce,CompsList<C...>,_Fund>> =true;
  
  /// Conjugate an expression
  template <typename _E>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
  decltype(auto) conj(_E&& e)
  {
    /// Base passed type
    using E=
      std::decay_t<_E>;
    
    if constexpr(isConjugator<E>)
      return e.conjugExpr;
    else
      {
	/// Components
	using Comps=
	  typename E::Comps;
	
	if constexpr(not tupleHasType<Comps,ComplId>)
	  return e;
	else
	  {
	    /// Type returned when evaluating the expression
	    using Fund=
	      typename E::Fund;
	    
	    return
	      Conjugator<_E,Comps,Fund>(std::forward<_E>(e));
	  }
      }
  }
  
#define FOR_REIM_PARTS(NAME)		\
  FOR_ALL_COMPONENT_VALUES(ComplId,NAME)
  
  /// Real component index - we cannot rely on a constexpr inline as the compiler does not propagate it correctly
#define Re ComplId(0)
  
  /// Imaginary component index
#define Im ComplId(1)
  
  /// Returns the real part, subscribing the complex component to Re value
  template <typename _E>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
  decltype(auto) real(_E&& e)
  {
    return
      //e(ComplId(0));
      e(Re);
  }
  
  /// Returns the imaginary part, subscribing the complex component to Im value
  template <typename _E>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
  decltype(auto) imag(_E&& e)
  {
    return
      e(Im);
  }
}

#endif
