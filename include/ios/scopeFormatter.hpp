#ifndef _SCOPE_FORMATTER_HPP
#define _SCOPE_FORMATTER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file scopeFormatter.hpp
///
/// \brief Implements a number of scope-changer

namespace grill
{
  /// Mark the stream to be more indented
#define SCOPE_INDENT()						\
  /*! Indent current scope */					\
  SET_FOR_CURRENT_SCOPE(Logger::indentLev,Logger::indentLev+1)
  
  /// Set the precision for current scope
#define SCOPE_REAL_PRECISION(VAL)					\
  SET_FOR_CURRENT_SCOPE(Logger::logFile.realPrecision,VAL)
  
  /// Set the format for current scope
#define SCOPE_REAL_FORMAT()					\
  SET_FOR_CURRENT_SCOPE(Logger::logFile.realFormat,Logger::logFile)
  
  /// Set general for the current scope
#define SCOPE_REAL_FORMAT_GENERAL()					\
  SET_FOR_CURRENT_SCOPE(Logger::logFile.realFormat,File::RealFormat::GENERAL)
  
  /// Set fixed for the current scope
#define SCOPE_REAL_FORMAT_FIXED()					\
  SET_FOR_CURRENT_SCOPE(Logger::logFile.realFormat,File::RealFormat::FIXED)
  
  /// Set engineer for the current scope
#define SCOPE_REAL_FORMAT_ENGINEER()					\
  SET_FOR_CURRENT_SCOPE(Logger::logFile.realFormat,File::RealFormat::ENGINEER)
  
  /// Set printing always sign at the beginning of a number for current scope
#define SCOPE_ALWAYS_PUT_SIGN()						\
  SET_FOR_CURRENT_SCOPE(Logger::logFile.alwaysPrintSign,true)
  
  /// Set not to print always sign at the beginning of a number for current scope
#define SCOPE_NOT_ALWAYS_PUT_SIGN()					\
  SET_FOR_CURRENT_SCOPE(Logger::logFile.alwaysPrintSign,false)
  
  /// Set printing or not zero
#define SCOPE_ALWAYS_PRINT_ZERO()					\
  SET_FOR_CURRENT_SCOPE(Logger::logFile.alwaysPrintZero,true)
  
  /// Allows all ransk to print for current scope
#define SCOPE_ALL_RANKS_CAN_PRINT()					\
  SET_FOR_CURRENT_SCOPE(Logger::onlyMasterRankPrint,false)
}

#endif
