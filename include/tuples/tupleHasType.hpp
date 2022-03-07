#ifndef _TUPLEHASTYPE_HPP
#define _TUPLEHASTYPE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file tupleHasType.hpp

#include <tuple>

namespace esnort
{
  /// Predicate returning whether the type is present in the list
  ///
  /// Forward definition
  template <int N,
	    typename Tp>
  struct TypeIsInList;
  
  /// Predicate returning whether the type is present in the list
  template <int N,
	    typename...Tp>
  struct TypeIsInList<N,std::tuple<Tp...>>
  {
    /// Internal implementation
    template <typename T>
    struct t
    {
      /// Predicate result
      static constexpr bool value=((std::is_same<T,Tp>::value+...)==N);
    };
  };
  
  template <typename Tp,
	    typename T,
	    int N=1>
  constexpr bool tupleHasType=
    TypeIsInList<N,Tp>::template t<T>::value;
  
}

#endif
