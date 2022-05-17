#ifndef _CALL_HPP
#define _CALL_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file call.hpp
///
/// \brief Calls the passed function

#include <tuple>

#include <metaprogramming/inline.hpp>

namespace grill::internal
{
  /// Wraps the function to be called
#define PROVIDE_CALL(NAME,ATTRIB)			\
  template <typename F,					\
	    typename...Args>				\
  INLINE_FUNCTION HOST_DEVICE_ATTRIB ATTRIB		\
  int NAME(F&& f,					\
	   Args&&...args)				\
  {							\
    f(std::forward<Args>(args)...);			\
    							\
    return 0;						\
  }
  
  PROVIDE_CALL(call,HOST_DEVICE_ATTRIB)
  
  PROVIDE_CALL(hostCall,HOST_ATTRIB)
  
#undef PROVIDE_CALL
}

#endif
