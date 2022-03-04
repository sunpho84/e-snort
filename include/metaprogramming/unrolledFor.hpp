#ifndef _UNROLLED_FOR_HPP
#define _UNROLLED_FOR_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file unrolledFor.hpp

#include <utility>

#include <metaprogramming/call.hpp>
#include <metaprogramming/inline.hpp>

namespace esnort
{
  namespace internal
  {
    /// Unroll a loop
    ///
    /// Actual implementation
    template <int Begin,
	      int...Is,
	      typename F>
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    void unrolledForInternal(std::integer_sequence<int,Is...>,F&& f)
    {
      /// Dummy initialized list, discarded at compile time
      ///
      /// The attribute avoids compiler warning.
      [[ maybe_unused ]]
      auto list=
	{call(std::forward<F>(f),Begin+Is)...,0};
    }
  }
  
  /// Unroll a loop, wrapping the actual implementation
  template <int Begin,
	    int End,
            typename F>
  INLINE_FUNCTION
  void unrolledFor(F&& f)
  {
    internal::unrolledForInternal<Begin>(std::make_integer_sequence<int,End-Begin>{},std::forward<F>(f));
  }

#define _UNROLLED_FOR_HEADER(VAR_NAME,BEGIN,END) \
  unrolledFor<BEGIN,END>([&](const auto& VAR_NAME) INLINE_ATTRIBUTE
  
  /// Create an unrolled for
  ///
  /// Hides the complexity
#define UNROLLED_FOR(HEADER,BODY...)				\
  _UNROLLED_FOR_HEADER HEADER					\
  BODY)
}

#endif
