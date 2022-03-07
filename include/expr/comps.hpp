#ifndef _COMPS_HPP
#define _COMPS_HPP

#include <tuple>
#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file comps.hpp

namespace esnort
{
  /// Collection of components
  template <typename...Tc>
  using Comps=
    std::tuple<Tc...>;
  
  /// Alias to make it easier to understand tensor instantiation
  template <typename...Tc>
  using OfComps=
    Comps<Tc...>;
  
  /// Returns the number of components of a Comps
  template <typename T>
  constexpr int nOfComps=
    std::tuple_size<typename T::Comps>::value;
}

#endif
