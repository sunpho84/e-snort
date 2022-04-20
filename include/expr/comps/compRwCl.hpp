#ifndef _COMP_RW_CL_HPP
#define _COMP_RW_CL_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/comps/compRwCl.hpp
///
/// \brief Implements a tensor comp row or column type

namespace grill
{
  /// Row or column
  enum class RwCl{ROW,CLN};
  
  /// Transposed of a row or column
  ///
  /// Forward declaration
  template <RwCl>
  RwCl transpRwCl;
  
  /// Transposed of a row
  template <>
  inline constexpr RwCl transpRwCl<RwCl::ROW> =
    RwCl::CLN;
  
  /// Transposed of a column
  template <>
  inline constexpr RwCl transpRwCl<RwCl::CLN> =
    RwCl::ROW;
}

#endif
