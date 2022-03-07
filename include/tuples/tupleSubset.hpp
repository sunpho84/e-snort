#ifndef _TUPLESUBSET_HPP
#define _TUPLESUBSET_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file tupleSubset.hpp

#include <tuple>

namespace esnort
{
  /// Get the list elements from a tuple
  template <typename...Tout,
	    typename...Tin>
  auto tupleGetMany(const std::tuple<Tin...>& in)
  {
    return std::make_tuple(std::get<Tout>(in)...);
  }
  
  /// Get the list elements from a tuple
  template <typename...Tout,
	    typename...Tin>
  void tupleFillWithSubset(std::tuple<Tout...>& out,
			   const std::tuple<Tin...>& in)
  {
    out=tupleGetMany<Tout...>(in);
  }
  
  /// Get the list elements of the passed tuple
  ///
  /// \example
  /// auto tupleGetSubset<std::tuple<int>>(std::make_tuple(1,10.0));
  template <typename TpOut,
	    typename...Tin>
  auto tupleGetSubset(const std::tuple<Tin...>& in)
  {
    TpOut out;
    
    tupleFillWithSubset(out,in);
    
    return out;
  }
}

#endif
