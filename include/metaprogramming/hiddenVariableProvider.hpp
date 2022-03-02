#ifndef _HIDDENVARIABLEPROVIDER_HPP
#define _HIDDENVARIABLEPROVIDER_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file hiddenVariableProvider.hpp

#ifdef DEFINE_HIDDEN_VARIABLES_ACCESSORS

# define DEFINE_OR_DECLARE_HIDDEN_VARIABLE_WITH_CONST_ACCESSOR(TYPE,NAME,ARGS...) \
  TYPE _ ## NAME ARGS;							\
  									\
  const TYPE& NAME=_ ## NAME

#else

# define DEFINE_OR_DECLARE_HIDDEN_VARIABLE_WITH_CONST_ACCESSOR(TYPE,NAME,ARGS...) \
  namespace writeAccess{						\
    extern TYPE& NAME;							\
  }									\
  extern const TYPE& NAME

#endif

#endif
