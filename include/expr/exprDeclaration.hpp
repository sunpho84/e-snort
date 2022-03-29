#ifndef _EXPRDECLARATION_HPP
#define _EXPRDECLARATION_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <metaprogramming/detectableAs.hpp>

/// \file expr/exprDeclaration.hpp
///
/// \brief Declares an expression

namespace esnort
{
  /// Base type representing an expression
  template <typename T>
  struct Expr;
  
  PROVIDE_DETECTABLE_AS(Expr);
}

#endif
