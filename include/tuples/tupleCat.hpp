#ifndef _TUPLECAT_HPP
#define _TUPLECAT_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file tupleCat.hpp

#include <tuple>

namespace esnort
{
  /// Type of the tuple obtained catting all passed tuples
  template <typename...TP>
  using TupleCat=
    decltype(std::tuple_cat(*(TP*)nullptr...));
}

#endif
