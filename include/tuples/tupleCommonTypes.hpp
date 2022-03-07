#ifndef _TUPLECOMMONTYPES_HPP
#define _TUPLECOMMONTYPES_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file tupleCommonTypes.hpp

#include <tuples/tupleFilter.hpp>
#include <tuples/tupleHasType.hpp>

namespace esnort
{
  /// Returns a tuple containing all types common to the two tuples
  template <typename TupleToSearch,
	    typename TupleBeingSearched>
  using TupleCommonTypes=
    TupleFilter<TypeIsInList<1,TupleToSearch>::template t,TupleBeingSearched>;
}

#endif
