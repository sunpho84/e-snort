#ifndef _EXPRDECLARATION_HPP
#define _EXPRDECLARATION_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/exprDeclaration.hpp
///
/// \brief Declares an expression

#include <metaprogramming/detectableAs.hpp>

namespace esnort
{
  /// Base type representing an expression
  template <typename T>
  struct Expr;
  
  PROVIDE_DETECTABLE_AS(Expr);
}

#endif
