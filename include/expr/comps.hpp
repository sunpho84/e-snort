#ifndef _COMPS_HPP
#define _COMPS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file comps.hpp

#include <tuple>

namespace esnort
{
  /// Collection of components
  template <typename...Tc>
  using CompsList=
    std::tuple<Tc...>;
  
  /// Alias to make it easier to understand tensor instantiation
  template <typename...Tc>
  using OfComps=
    CompsList<Tc...>;
}

#endif
