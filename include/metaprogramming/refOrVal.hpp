#ifndef _REFORVAL_HPP
#define _REFORVAL_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <type_traits>

namespace esnort
{
  namespace details
  {
    template <bool IsRef,
	      typename T>
    using _ConditionalRef=
      std::conditional_t<IsRef,T&,T>;
  }
  
  template <bool IsRef,
	    typename T>
  using ConditionalRef=
    details::_ConditionalRef<IsRef,std::remove_reference_t<T>>;
}

#endif
