#ifndef _DEMANGLE_HPP
#define _DEMANGLE_HPP

/// \file demangle.hpp
///
/// \brief Demangle symbols using abi

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <string>

namespace grill
{
  /// Demangle a string
  ///
  /// If the compiler has no abi functionality, the original string is
  /// returned.
  std::string demangle(const std::string& what);  ///< What to demangle
}

#endif
