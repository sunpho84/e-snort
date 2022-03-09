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
#include <metaprogramming/crtp.hpp>
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
    
    /// Assign from another expression
    template <typename U>
    T& operator=(const Expr<U>& u)
    {
      /// Gets the lhs by casting to actual type
      decltype(auto) lhs=this->crtp();
      
      /// Gets the rhs by casting to actual type
      decltype(auto) rhs=u.crtp();
      
      /// Type of the lhs
      using Lhs=std::decay_t<decltype(lhs)>;
      
      /// Type of the rhs
      using Rhs=std::decay_t<decltype(rhs)>;
      
      /// Execution space for the lhs
      constexpr ExecutionSpace lhsExecSpace=Lhs::execSpace;
      
      /// Execution space for the rhs
      constexpr ExecutionSpace rhsExecSpace=Rhs::execSpace;
      
      if constexpr(lhsExecSpace==rhsExecSpace)
	lhs()=rhs();
      else
	{
	  /// Decide which side of the assignment will change the
	  /// execution space. This is done in terms of an euristic cost
	  /// of the change, but we might refine in the future. The
	  /// assumption is that ultimately, the results will be stored in
	  /// any case on the lhs execution space, so we should avoid
	  /// moving the lhs unless it is much lest costly.
	  if constexpr (Lhs::execSpaceChangeCost<Rhs::execSpaceChangeCost)
	    ;
	  else
#warning messagelhs()=rhs.template changeExecSpaceTo<lhsExecSpace>()()
	    ;
	}
      
      return this->crtp();
    }

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
