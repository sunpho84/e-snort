#ifndef _TUPLEREPLACETYPE_HPP
#define _TUPLEREPLACETYPE_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file tupleReplaceType.hpp

#include <tuple>

namespace esnort
{
  namespace internal
  {
    /// Replace type A with B in tuple Tp
    ///
    /// Internal implementation, forward declaration
    template <typename Tp,
	      typename A,
	      typename B>
    struct _TupleReplaceType;
    
    /// Replace type A with B in tuple Tp
    ///
    /// Internal implementation
    template <typename...Tp,
	      typename A,
	      typename B>
    struct _TupleReplaceType<std::tuple<Tp...>,A,B>
    {
      using type=
	std::tuple<std::conditional_t<std::is_same_v<Tp,A>,B,Tp>...>;
    };
  }
  
  /// Replace type A with B in tuple Tp
  template <typename Tp,
	    typename A,
	    typename B>
  using TupleReplaceType=
    typename internal::_TupleReplaceType<Tp,A,B>::type;
}

#endif
