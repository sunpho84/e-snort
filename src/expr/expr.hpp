#ifndef _EXPR_HPP
#define _EXPR_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/expr.hpp
///
/// \brief Declare base expression, to issue the assignments

#include <type_traits>

#include <assign/assign.hpp>
#include <expr/executionSpace.hpp>
#include <metaprogramming/crtp.hpp>

namespace esnort
{
  /// Base type representing an expression
  template <typename T>
  struct Expr :
    Crtp<T>
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
      constexpr ExecutionSpace lhsExecSpace=Lhs::execSpace();
      
      /// Execution space for the rhs
      constexpr ExecutionSpace rhsExecSpace=Rhs::execSpace();
      
      /// Decide which side of the assignment will change the
      /// execution space. This is done in terms of an euristic cost
      /// of the change, but we might refine in the future. The
      /// assumption is that ultimately, the results will be stored in
      /// any case on the lhs execution space, so we should avoid
      /// moving the lhs unless it is much lest costly.
      constexpr WhichSideToChange whichSideToChange=
		  (Rhs::execSpaceChangeCost()>Lhs::execSpaceChangeCost())?
		  (WhichSideToChange::LHS):
      (WhichSideToChange::RHS);
      
      /// Issue the acutal assignmentx
      Assign<lhsExecSpace,rhsExecSpace,whichSideToChange>::exec(lhs,rhs);
      
      return this->crtp();
    }
  };
}

#endif
