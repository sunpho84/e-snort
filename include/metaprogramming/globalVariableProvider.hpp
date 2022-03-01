#ifndef _GLOBALVARIABLEPROVIDER_HPP
#define _GLOBALVARIABLEPROVIDER_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file globalVariableProvider.hpp

#ifdef DEFINE_GLOBAL_VARIABLES
 
# define DEFINE_OR_DECLARE_GLOBAL_VARIABLE(TYPE,NAME,ARGS...) TYPE NAME ARGS

#else

# define DEFINE_OR_DECLARE_GLOBAL_VARIABLE(TYPE,NAME,ARGS...) extern TYPE NAME

#endif


#endif
