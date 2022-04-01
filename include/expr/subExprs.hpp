#ifndef _SUBEXPRS_HPP
#define _SUBEXPRS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <tuple>

#include <expr/exprRefOrVal.hpp>

/// \file expr/subExprs.hpp
///
/// \brief Type to hold subexpressions

namespace esnort
{
  /// Holds the subexpressions
  template <typename..._E>
  struct SubExprs
  {
    /// Subexpressions
    std::tuple<ExprRefOrVal<_E>...> subExprs;

#define PROVIDE_SUBEXPR(ATTRIB)				\
    /*! Proxy for the I-subexpression */		\
    template <int I>					\
    INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB	\
    decltype(auto) subExpr() ATTRIB			\
    {							\
      return std::get<I>(subExprs);			\
    }
    
    PROVIDE_SUBEXPR(const);
    
    PROVIDE_SUBEXPR(/* non const */);
    
#undef PROVIDE_SUBEXPR
    
    /// Type of the I-th subexpressions
    template <int I>
    using SubExpr=
      std::decay_t<std::tuple_element_t<I,std::tuple<_E...>>>;
    
    /// Constructor
    template <typename...T>
    SubExprs(T&&...t) :
      subExprs(std::forward<T>(t)...)
    {
    }
  };
  
  /// Access subexpressions
#define SUBEXPR(I)				\
  this->template subExpr<I>()

#define IMPORT_SUBEXPR_TYPES				\
  /*! Import subexpressions type */			\
  template <int I>					\
  using SubExpr=					\
    typename SubExprs<_E...>::template SubExpr<I>
}

#endif
