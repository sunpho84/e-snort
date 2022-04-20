#ifndef _STATIC_MEMBER_WITH_INITIALIZATOR_HPP
#define _STATIC_MEMBER_WITH_INITIALIZATOR_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file StaticMemberWithInitializator.hpp
///
/// \brief Provides a static member with an initalizator

namespace grill
{
  /// Provides a static member with an initalizator
  ///
  /// A method NAME() is provided, which gives access to the internal
  /// data. This is implemented to avoid out-of-line initialization,
  /// following the advice of https://stackoverflow.com/a/31608148
#define PROVIDE_STATIC_MEMBER_WITH_INITIALIZATOR(T,NAME,VALUE,DESCRIPTION) \
  /*! DESCRIPTION */							\
  static int& NAME()							\
  {									\
    /*! Internal storage */						\
    static int _ ## NAME						\
      =VALUE;								\
    									\
    return _ ## NAME;							\
  }
}

#endif
