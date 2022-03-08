#ifndef _DYNAMICCOMPSPROVIDER_HPP
#define _DYNAMICCOMPSPROVIDER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/dynamicCompsProvider.hpp

#include <expr/comp.hpp>
#include <expr/comps.hpp>
#include <tuples/tupleDiscriminate.hpp>

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
    
    /// Sizes of the dynamic components
    DynamicComps dynamicSizes;
    
    static constexpr int nDynamicComps=
      std::tuple_size_v<DynamicComps>;
  };
}

#endif
