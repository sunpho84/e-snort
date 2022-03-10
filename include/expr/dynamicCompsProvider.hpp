#ifndef _DYNAMICCOMPSPROVIDER_HPP
#define _DYNAMICCOMPSPROVIDER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/dynamicCompsProvider.hpp

#include <expr/comp.hpp>
#include <expr/comps.hpp>
#include <tuples/tupleDiscriminate.hpp>
#include <tuples/tupleSubset.hpp>

namespace esnort
{
  template <typename...C>
  struct DynamicCompsProvider
  {
    using DynamicStaticComps=
      TupleDiscriminate<SizeIsKnownAtCompileTime,CompsList<C...>>;
    
    /// List of all statically allocated components
    using StaticComps=
      typename DynamicStaticComps::Valid;
    
    /// List of all dynamically allocated components
    using DynamicComps=
      typename DynamicStaticComps::Invalid;
    
    static constexpr int nDynamicComps=
      std::tuple_size_v<DynamicComps>;
    
    /// Returns dynamic comps from a list
    template <typename...T,
	      typename...I>
    static DynamicComps filterDynamicComps(const BaseComp<T,I>&...td)
    {
      return tupleGetSubset<DynamicComps>(std::make_tuple(td.crtp()...));
    }
  };
}

#endif
