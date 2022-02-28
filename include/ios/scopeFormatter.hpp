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
  auto NAME3(SET,NAME,__LINE__)=getScopeChangeVar(Logger::indentLev,Logger::indentLev+1)
  
  /// Set for current scope
#define SET_FOR_CURRENT_SCOPE(NAME,VAR,ARGS...)				\
  auto NAME3(SET,NAME,__LINE__)=getScopeChangeVar(VAR,ARGS)
  
  /// Set the precision for current scope
#define SCOPE_REAL_PRECISION(VAL)					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_PRECISION,Logger::logFile.realPrecision,VAL)
  
  /// Set the format for current scope
#define SCOPE_REAL_FORMAT()					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_FORMAT,Logger::logFile.realFormat,Logger::logFile)
  
  /// Set general for the current scope
#define SCOPE_REAL_FORMAT_GENERAL()					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_FORMAT_GENERAL,Logger::logFile.realFormat,File::RealFormat::GENERAL)
  
  /// Set fixed for the current scope
#define SCOPE_REAL_FORMAT_FIXED()					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_FORMAT_FIXED,Logger::logFile.realFormat,File::RealFormat::FIXED)
  
  /// Set engineer for the current scope
#define SCOPE_REAL_FORMAT_ENGINEER()					\
  SET_FOR_CURRENT_SCOPE(STREAM_REAL_FORMAT_ENGINEER,Logger::logFile.realFormat,File::RealFormat::ENGINEER)
  
  /// Set printing always sign at the beginning of a number for current scope
#define SCOPE_ALWAYS_PUT_SIGN()						\
  SET_FOR_CURRENT_SCOPE(STREAM_ALWAYS_PRINT_SIGN,Logger::logFile.alwaysPrintSign,true)
  
  /// Set not to print always sign at the beginning of a number for current scope
#define SCOPE_NOT_ALWAYS_PUT_SIGN()					\
  SET_FOR_CURRENT_SCOPE(STREAM_NOT_ALWAYS_PRINT_SIGN,Logger::logFile.alwaysPrintSign,false)
  
  /// Set printing or not zero
#define SCOPE_ALWAYS_PRINT_ZERO()					\
  SET_FOR_CURRENT_SCOPE(STREAM_ALWAYS_PRINT_ZERO,Logger::logFile.alwaysPrintZero,true)
  
  /// Allows all ransk to print for current scope
#define SCOPE_ALL_RANKS_CAN_PRINT()			\
  SET_FOR_CURRENT_SCOPE(STREAM_ALL_RANKS_CAN_PRINT,Logger::logFile.onlyMasterRankPrint,false)
}

#endif
