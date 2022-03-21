#ifndef _EXPR_HPP
#define _EXPR_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/expr.hpp
///
/// \brief Declare base expression, to issue the assignments

#include <type_traits>

#include <expr/compLoops.hpp>
#include <expr/deviceAssign.hpp>
#include <expr/directAssign.hpp>
#include <expr/executionSpace.hpp>
#include <expr/threadAssign.hpp>
#include <ios/logger.hpp>
#include <metaprogramming/crtp.hpp>
#include <tuples/tupleHasType.hpp>
#include <tuples/tupleFilter.hpp>

namespace esnort
{
  /// Base type representing an expression
  template <typename T>
  struct Expr
  {
    // Seems unnecessary
    /// Define the assignment operator with the same expression type,
    /// in terms of the templated version
    // INLINE_FUNCTION
    // Expr& operator=(const Expr& oth)
    // {
    //   return this->operator=<T>(oth);
    // }
    
    /// Returns whether can assign: this is actually used when no other assignability is defined
    constexpr bool canAssign() const
    {
      return false;
    }
    
    static constexpr bool canAssignAtCompileTime=false;
    
    /// Assert assignability
    template <typename U>
    constexpr void assertCanAssign(const Expr<U>& _rhs)
    {
      static_assert(tuplesContainsSameTypes<typename T::Comps,typename U::Comps>,"Cannot assign two expressions which differ for the components");
      
      static_assert(T::canAssignAtCompileTime,"Trying to assign to a non-assignable expression");
      
      auto& lhs=DE_CRTPFY(T,this);
      const auto& rhs=DE_CRTPFY(const U,&_rhs);
      
      if(not lhs.canAssign())
	CRASH<<"Trying to assign to a non-assignable expression";
      
      if constexpr(U::hasDynamicComps)
	if(lhs.getDynamicSizes()!=rhs.getDynamicSizes())
	  CRASH<<"Dynamic comps not agreeing";
      
      static_assert(T::execSpace==U::execSpace or
		    U::execSpace==ExecutionSpace::UNDEFINED,"Cannot assign among different execution space, first change one of them");
    }
    
    /// Assign from another expression
    template <typename Rhs>
    INLINE_FUNCTION
    T& operator=(const Expr<Rhs>& u)
    {
      assertCanAssign(u);
      
      auto& lhs=DE_CRTPFY(T,this);
      const auto& rhs=DE_CRTPFY(const Rhs,&u);
      
#if ENABLE_SIMD
      if constexpr(T::canSimdify and Rhs::canSimdify)
	lhs.simdify()=rhs.simdify();
      else
#endif
#if ENABLE_DEVICE_CODE
	if constexpr(Rhs::execSpace==ExecutionSpace::DEVICE)
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
	return compBind(t,std::make_tuple(cs...));			\
    }
    
    PROVIDE_SUBSCRIBE(const);
    
    PROVIDE_SUBSCRIBE(/* non const */);
    
#undef PROVIDE_SUBSCRIBE
  };
}

#endif
