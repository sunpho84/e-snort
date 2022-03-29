#ifndef _EXPR_HPP
#define _EXPR_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/expr.hpp
///
/// \brief Declare base expression, to issue the assignments

#include <type_traits>

#include <expr/bindComps.hpp>
#include <expr/compLoops.hpp>
#include <expr/deviceAssign.hpp>
#include <expr/directAssign.hpp>
#include <expr/executionSpace.hpp>
#include <expr/exprDeclaration.hpp>
#include <expr/simdAssign.hpp>
#include <expr/threadAssign.hpp>
#include <ios/logger.hpp>
#include <metaprogramming/crtp.hpp>
#include <tuples/tupleHasType.hpp>
#include <tuples/tupleFilter.hpp>

namespace esnort
{
  namespace constraints
  {
    PROVIDE_HAS_MEMBER(getDynamicSizes);
    PROVIDE_HAS_MEMBER(Comps);
    PROVIDE_HAS_MEMBER(canAssign);
    PROVIDE_HAS_MEMBER(execSpace);
    PROVIDE_HAS_MEMBER(getRef);
    PROVIDE_HAS_MEMBER(eval);
    PROVIDE_HAS_MEMBER(storeByRef);
    PROVIDE_HAS_MEMBER(canSimdify);
    PROVIDE_HAS_MEMBER(hasDynamicComps);
    PROVIDE_HAS_MEMBER(canAssignAtCompileTime);
  }
  
  template <typename T>
  struct Expr :
    DetectableAsExpr
  {
    // /// Define the move-assignment operator
    // INLINE_FUNCTION
    // Expr& operator=(Expr&& oth)
    // {
    //   return this->operator=<T>(std::forward<Expr>(oth));
    // }
    
    /// Returns whether can assign: this is actually used when no other assignability is defined
    constexpr bool canAssign() const
    {
      return false;
    }
    
    static constexpr bool canAssignAtCompileTime=false;
    
    /// Used to check that the derived type satisfy the Expr criterion
    constexpr Expr()
    {
      using namespace constraints;
      
      static_assert(hasMember_getDynamicSizes<T> and
		    hasMember_canAssign<T> and
		    hasMember_Comps<T> and
		    hasMember_execSpace<T> and
		    hasMember_eval<T> and
		    hasMember_getRef<T> and
		    hasMember_canSimdify<T> and
		    hasMember_canAssignAtCompileTime<T> and
		    hasMember_hasDynamicComps<T> and
		    hasMember_storeByRef<T>,
		    "Incomplete expression");
   }
    
    /// Assert assignability
    template <typename U>
    INLINE_FUNCTION
    constexpr void assertCanAssign(const Expr<U>& _rhs)
    {
      static_assert(tuplesContainsSameTypes<typename T::Comps,typename U::Comps>,"Cannot assign two expressions which differ for the components");
      
      static_assert(T::canAssignAtCompileTime,"Trying to assign to a non-assignable expression");
      
      auto& lhs=DE_CRTPFY(T,this);
      const auto& rhs=DE_CRTPFY(const U,&_rhs);
      
      if constexpr(not T::canAssignAtCompileTime)
	CRASH<<"Trying to assign to a non-assignable expression";
      
      if constexpr(U::hasDynamicComps)
	if(lhs.getDynamicSizes()!=rhs.getDynamicSizes())
	  CRASH<<"Dynamic comps not agreeing";
      
      static_assert(T::execSpace==U::execSpace or
		    U::execSpace==ExecSpace::UNDEFINED,"Cannot assign among different execution space, first change one of them");
    }
    
    /// Assign from another expression
    template <typename Rhs>
    INLINE_FUNCTION
    T& assign(const Expr<Rhs>& u)
    {
      assertCanAssign(u);
      
      auto& lhs=DE_CRTPFY(T,this);
      const auto& rhs=DE_CRTPFY(const Rhs,&u);
      
#if ENABLE_SIMD
      if constexpr(T::canSimdify and Rhs::canSimdify and std::is_same_v<typename T::SimdifyingComp,typename Rhs::SimdifyingComp>)
	simdAssign(lhs,rhs);
      else
#endif
#if ENABLE_DEVICE_CODE
	if constexpr(Rhs::execSpace==ExecSpace::DEVICE)
	  deviceAssign(lhs,rhs);
	else
#endif
#if ENABLE_THREADS
	  if constexpr(Rhs::nDynamicComps==1)
	    threadAssign(lhs,rhs);
	  else
#endif
	    directAssign(lhs,rhs);
      
      return lhs;
    }
    
    /// Assign from another expression
    template <typename Rhs>
    INLINE_FUNCTION
    T& operator=(const Expr<Rhs>& u)
    {
      return this->assign(u);
    }
    
    /// Define the assignment operator with the same expression type,
    /// in terms of the templated version
    INLINE_FUNCTION
    Expr& operator=(const Expr& oth)
    {
      return this->assign(oth);
    }
    
    /// Returns the expression as a dynamic tensor
    auto fillDynamicTens() const;
    
#define PROVIDE_SUBSCRIBE(ATTRIB)					\
    template <typename...C>						\
    HOST_DEVICE_ATTRIB constexpr INLINE_FUNCTION			\
    decltype(auto) operator()(const C&...cs) ATTRIB			\
    {									\
      /*! Leftover components */					\
      using ResidualComps=						\
	TupleFilterAllTypes<typename T::Comps,std::tuple<C...>>;	\
      									\
      ATTRIB auto& t=DE_CRTPFY(ATTRIB T,this);				\
      									\
      if constexpr(std::tuple_size_v<ResidualComps> ==0)		\
	return t.eval(cs...);						\
      else								\
	return bindComps(t,std::make_tuple(cs...));			\
    }
    
    PROVIDE_SUBSCRIBE(const);
    
    PROVIDE_SUBSCRIBE(/* non const */);
    
#undef PROVIDE_SUBSCRIBE
  };
}

#endif
