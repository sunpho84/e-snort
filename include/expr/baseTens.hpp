#ifndef _BASETENS_HPP
#define _BASETENS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/baseTens.hpp

#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
#include <resources/memory.hpp>

namespace esnort
{
  /// Tensor
  ///
  /// Forward declaration
  template <typename T,
	    typename F,
	    ExecutionSpace ES>
  struct BaseTens :
    Expr<T>
  {
    using Expr<T>::operator=;
    
    /// Assign from another dynamic tensor of the very same type
    template <typename OtherT,
	      ExecutionSpace OtherES>
    BaseTens& operator=(const BaseTens<OtherT,F,OtherES>& _oth)
    {
      T& t=this->crtp();
      const OtherT& oth=_oth.crtp();
      
      if(t.storageSize!=oth.storageSize)
	CRASH<<"Storage size not agreeing";
      
      LOGGER<<"Copying a "<<execSpaceName<ES><<" tensor into a "<<execSpaceName<OtherES><<" one";
      
      memory::memcpy<ES,OtherES>(t.storage,oth.storage,oth.storageSize);
      
      return *this;
    }
  };
  
}

#endif
