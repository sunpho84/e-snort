#ifndef _SCOPE_FORMATTER_HPP
#define _SCOPE_FORMATTER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file scopeFormatter.hpp
///
/// \brief Implements a number of scope-changer

namespace esnort
{
  /// Mark the stream to be more indented
#define SCOPE_INDENT(VAR)					\
  /*! Indent current scope */					\
  ScopeIndenter NAME2(SCOPE_INDENTER,__LINE__)(VAR)
  
  /// Set for current scope
#define SET_FOR_CURRENT_SCOPE(NAME,VAR,...)			\
  auto NAME3(SET,NAME,__LINE__)=getScopeChangeVar(VAR,__VA_ARGS__)
  
  /// Set the precision for current scope
#define SCOPE_REAL_PRECISION(STREAM,VAL)					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_PRECISION,STREAM.realPrecision,VAL)
  
  /// Set the format for current scope
#define SCOPE_REAL_FORMAT(STREAM,VAL)					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_FORMAT,STREAM.realFormat,VAL)
  
  /// Set general for the current scope
#define SCOPE_REAL_FORMAT_GENERAL(STREAM)					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_FORMAT_GENERAL,STREAM.realFormat,RealFormat::GENERAL)
  
  /// Set fixed for the current scope
#define SCOPE_REAL_FORMAT_FIXED(STREAM)					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_FORMAT_FIXED,STREAM.realFormat,RealFormat::FIXED)
  
  /// Set engineer for the current scope
#define SCOPE_REAL_FORMAT_ENGINEER(STREAM)					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_FORMAT_ENGINEER,STREAM.realFormat,RealFormat::ENGINEER)
  
  /// Set printing always sign at the beginning of a number for current scope
#define SCOPE_ALWAYS_PUT_SIGN(STREAM)			\
  SET_FOR_CURRENT_SCOPE(STREAM_ALWAYS_PRINT_SIGN,STREAM.alwaysPrintSign,true)
  
  /// Set not to print always sign at the beginning of a number for current scope
#define SCOPE_NOT_ALWAYS_PUT_SIGN(STREAM)			\
  SET_FOR_CURRENT_SCOPE(STREAM_NOT_ALWAYS_PRINT_SIGN,STREAM.alwaysPrintSign,false)
  
  /// Set printing or not zero
#define SCOPE_ALWAYS_PRINT_ZERO(STREAM)			\
  SET_FOR_CURRENT_SCOPE(STREAM_ALWAYS_PRINT_ZERO,STREAM.alwaysPrintZero,true)
  
  /// Set printing or not zero
#define SCOPE_ALWAYS_PRINT_ZERO(STREAM)			\
  SET_FOR_CURRENT_SCOPE(STREAM_ALWAYS_PRINT_ZERO,STREAM.alwaysPrintZero,true)
  
  /// Allows all ransk to print for current scope
#define SCOPE_ALL_RANKS_CAN_PRINT(STREAM)			\
  SET_FOR_CURRENT_SCOPE(STREAM_ALL_RANKS_CAN_PRINT,STREAM.onlyMasterRankPrint,false)
}

#endif
