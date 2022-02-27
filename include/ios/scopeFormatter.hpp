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
#define SCOPE_INDENT()						\
  /*! Indent current scope */					\
  ScopeIndenter NAME2(SCOPE_INDENTER,__LINE__)(resources::logger)
  
  /// Set for current scope
#define SET_FOR_CURRENT_SCOPE(NAME,VAR,ARGS...)				\
  auto NAME3(SET,NAME,__LINE__)=getScopeChangeVar(VAR,ARGS)
  
  /// Set the precision for current scope
#define SCOPE_REAL_PRECISION(VAL)					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_PRECISION,resources::logger.realPrecision,VAL)
  
  /// Set the format for current scope
#define SCOPE_REAL_FORMAT()					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_FORMAT,resources::logger.realFormat,resources::logger)
  
  /// Set general for the current scope
#define SCOPE_REAL_FORMAT_GENERAL()					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_FORMAT_GENERAL,resources::logger.realFormat,RealFormat::GENERAL)
  
  /// Set fixed for the current scope
#define SCOPE_REAL_FORMAT_FIXED()					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_FORMAT_FIXED,resources::logger.realFormat,RealFormat::FIXED)
  
  /// Set engineer for the current scope
#define SCOPE_REAL_FORMAT_ENGINEER()					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_FORMAT_ENGINEER,resources::logger.realFormat,RealFormat::ENGINEER)
  
  /// Set printing always sign at the beginning of a number for current scope
#define SCOPE_ALWAYS_PUT_SIGN()						\
  SET_FOR_CURRENT_SCOPE(STREAM_ALWAYS_PRINT_SIGN,resources::logger.alwaysPrintSign,true)
  
  /// Set not to print always sign at the beginning of a number for current scope
#define SCOPE_NOT_ALWAYS_PUT_SIGN()					\
  SET_FOR_CURRENT_SCOPE(STREAM_NOT_ALWAYS_PRINT_SIGN,resources::logger.alwaysPrintSign,false)
  
  /// Set printing or not zero
#define SCOPE_ALWAYS_PRINT_ZERO()					\
  SET_FOR_CURRENT_SCOPE(STREAM_ALWAYS_PRINT_ZERO,resources::logger.alwaysPrintZero,true)
  
  /// Allows all ransk to print for current scope
#define SCOPE_ALL_RANKS_CAN_PRINT()			\
  SET_FOR_CURRENT_SCOPE(STREAM_ALL_RANKS_CAN_PRINT,resources::logger.onlyMasterRankPrint,false)
}

#endif
