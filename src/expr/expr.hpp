#ifndef _EXPR_HPP
#define _EXPR_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <type_traits>

#include <assign/assign.hpp>
#include <expr/executionSpace.hpp>
#include <metaprogramming/crtp.hpp>

namespace esnort
{
  template <typename T>
  struct Expr :
    Crtp<T>
  {
    template <typename U>
    T& operator=(const Expr<U>& u)
    {
      decltype(auto) lhs=this->crtp();
      decltype(auto) rhs=u.crtp();
      
      using Lhs=std::decay_t<decltype(lhs)>;
      using Rhs=std::decay_t<decltype(rhs)>;
      
      constexpr ExecutionSpace lhsExecSpace=Lhs::execSpace();
      constexpr ExecutionSpace rhsExecSpace=Rhs::execSpace();
      
      constexpr WhichSideToChange whichSideToChange=
		  (Rhs::execSpaceChangeCost()>Lhs::execSpaceChangeCost())?
		  CHANGE_EXEC_SPACE_LHS_SIDE:
      CHANGE_EXEC_SPACE_RHS_SIDE;

      
      Assign<lhsExecSpace,rhsExecSpace,whichSideToChange>::exec(lhs,rhs);
      
      return this->crtp();
    }
  };
}

#endif
