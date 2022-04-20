#ifndef _FOR_EACH_IN_TUPLE_HPP
#define _FOR_EACH_IN_TUPLE_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file forEachInTuple.hpp
///
/// \brief Provides functions to unroll tuple

#include <tuple>

#include <metaprogramming/inline.hpp>

namespace grill
{
  namespace resources
  {
    /// Wraps the function to be called
    ///
    /// Identycal to the \c call function above, but it works only on
    /// host, as the auto lambda is not supported by nvcc
    template <typename F,
	      typename...Args>
    INLINE_FUNCTION
    int tupleCall(F&& f,Args&&...args)
    {
      f(std::forward<Args>(args)...);
      
      return 0;
    }
    
    /// Loop on tuple arguments
    ///
    /// Actual implementation
    template <int...Is,
	      typename Tp,
	      typename F>
    INLINE_FUNCTION
    void forEachInTupleInternal(std::integer_sequence<int,Is...>,Tp&& tp,F&& f)
    {
      /// Dummy initialized list, discarded at compile time
      ///
      /// The attribute avoids compiler warning.
      [[ maybe_unused ]]
	auto list=
	{tupleCall(std::forward<F>(f),std::get<Is>(tp))...,0};
    }
  }
  
  /// Loop on tuple arguments
  template <typename Tp,
	    typename F>
  INLINE_FUNCTION
  void forEachInTuple(Tp&& tp,F&& f)
  {
    constexpr int tupleSize=
      std::tuple_size<std::remove_reference_t<Tp>>::value;
    
    resources::forEachInTupleInternal(std::make_integer_sequence<int,tupleSize>{},
				      std::forward<Tp>(tp),
				      std::forward<F>(f));
  }
}

#endif
