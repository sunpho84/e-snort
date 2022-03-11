#ifndef _EXPR_HPP
#define _EXPR_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/expr.hpp
///
/// \brief Declare base expression, to issue the assignments

#include <type_traits>

#include <expr/executionSpace.hpp>
#include <ios/logger.hpp>
#include <metaprogramming/crtp.hpp>
#include <tuples/tupleHasType.hpp>
#include <tuples/tupleFilter.hpp>

namespace esnort
{
  DEFINE_CRTP_INHERITANCE_DISCRIMINER_FOR_TYPE(Expr)
  
  /// Base type representing an expression
  template <typename T>
  struct Expr :
    Crtp<T,crtp::ExprDiscriminer>
  {
    /// Define the assignment operator with the same expression type,
    /// in terms of the templated version
    Expr& operator=(const Expr& oth)
    {
      return this->operator=<T>(oth);
    }
    
    /// Returns whether can assign: this is actually used when no other assignability is defined
    constexpr bool canAssign() const
    {
      return false;
    }
    
    static constexpr bool canAssignAtCompileTime=false;
    
    /// Assert assignability
    template <typename U>
    constexpr void assertCanAssign(const Expr<U>& rhs)
    {
      static_assert(tuplesContainsSameTypes<typename T::Comps,typename U::Comps>,"Cannot assign two expressions which differ for the components");
      
      static_assert(T::canAssignAtCompileTime,"Trying to assign to a non-assignable expression");
      
      if(not this->crtp().canAssign())
	CRASH<<"Trying to assign to a non-assignable expression";
      
      if constexpr(U::hasDynamicComps)
	if(this->crtp().getDynamicSizes()!=rhs.crtp().getDynamicSizes())
	  CRASH<<"Dynamic comps not agreeing";
      
      static_assert(T::execSpace==U::execSpace or
		    U::execSpace==ExecutionSpace::UNDEFINED,"Cannot assign among different execution space, first change one of them");
    }
    
    /// Assign from another expression
    template <typename Lhs>
    T& operator=(const Expr<Lhs>& u)
    {
      assertCanAssign(u);
      
      if constexpr(Lhs::execSpace==ExecutionSpace::DEVICE)
	{
	  /// For the time being, we assume that there is a single
	  /// dynamic component, and we loop with the gpu threads on
	  /// it, then we loop internally on the others
	  LOGGER<<"Using device kernel";
	}
      else
	{
	  /// We use threads only if there is at least one dynamic component
	  LOGGER<<"Using thread kernel";
	}
      
      return this->crtp();
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
      if constexpr(std::tuple_size_v<ResidualComps> ==0)		\
	return this->crtp().eval(cs...);				\
      else								\
	return compBind(this->crtp(),std::make_tuple(cs...));		\
    }
    
    PROVIDE_SUBSCRIBE(const);
    
    PROVIDE_SUBSCRIBE(/* non const */);
    
#undef PROVIDE_SUBSCRIBE
  };
}

#endif
