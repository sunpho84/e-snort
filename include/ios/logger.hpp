#ifndef _LOGGER_HPP
#define _LOGGER_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file logger.hpp
///
/// \brief Header file to define a logger wrapping FILE*
///
/// The internal class is used to really print, while the external one
/// to determine whether to print or not, and to lock in case the
/// threads are present. If printing is not needed, a fake internal
/// logger is passed, printing on /dev/null
///
/// If threads are running, and all threads are allowed to print, the
/// Logger is locked so that only one thread at the time can print,
/// and all lines are prepended with thread id indication.
///
/// If MPI is running, and all ranks are allowed, to print, each line
/// is prepended with the rank id. No cross-ranks lock is issued.

#include <cstdio>

#include <debug/backtracing.hpp>
#include <debug/crash.hpp>
#include <ios/file.hpp>
#include <ios/scopeFormatter.hpp>
#include <ios/textFormat.hpp>
#include <ios/loggerGlobalVariablesDeclarations.hpp>
#include <metaprogramming/operatorExists.hpp>
#include <resources/Mpi.hpp>
#include <resources/timer.hpp>
#include <resources/scopeDoer.hpp>
#include <resources/threads.hpp>

namespace esnort
{
  DEFINE_BINARY_OPERATOR_IMPLEMENTATION_CHECK(canPrint,CanPrint,<<);
  
  /// Write output to a file, using different level of indentation
  namespace Logger
  {
    /// Increase indentation
    INLINE_FUNCTION
    void indentMore()
    {
      indentLev++;
    }
    
    /// Decrease indentation
    INLINE_FUNCTION
    void indentLess()
    {
      indentLev--;
    }
    
    /// Single line in the logger
    class LoggerLine
    {
      /// Store whether actually printing
      bool actuallyPrint;
      
      /// Store wether the line has to be ended or not
      bool hasToEndLine;
      
      /// Store whether the line is crashing
      bool hasToCrash;
      
      /// Mark that the color has changed in this line
      bool colorChanged;
      
      /// Mark that the style has changed in this line
      bool styleChanged;
      
      /// Forbids copying a line
      LoggerLine(const LoggerLine&)=delete;
      
      /// Starts a new line
      void startNewLine(const bool actuallyPrint)
      {
	if(not actuallyPrint)
	  return;
	
	const bool prependRank=
	  Mpi::nRanks!=1 and not onlyMasterRankPrint;
	
	const bool prependThread=
	  (threads::isInsideParallelSection() and 
	   not onlyMasterThreadPrint);
	
	/// Total number of the character written
	int rc=0;
	
	// Prepend with time
	if(prependTime)
	  {
	    SCOPE_REAL_FORMAT_FIXED();
	    SCOPE_REAL_PRECISION(10);
	    SCOPE_ALWAYS_PRINT_ZERO();
	    SCOPE_NOT_ALWAYS_PUT_SIGN();
	    rc+=
	      (Logger::logFile<<durationInSec(timings.currentMeasure())<<" s").getRc();
	  }
	
	// Prepend with rank
	if(prependRank)
	  rc+=
	    (Logger::logFile<<" Rank "<<Mpi::rank).getRc();
	
	// Prepend with thread
	if(prependThread)
	  rc+=
	    (Logger::logFile<<" Thread "<<threads::threadId()).getRc();
	
	// Mark the margin
	if(rc)
	  Logger::logFile<<":\t";
	
	// Writes the given number of spaces
	for(int i=0;i<Logger::indentLev;i++)
	  Logger::logFile<<' ';
      }
      
    public:
      
      /// Construct
      LoggerLine(const bool& actuallyPrint) :
	actuallyPrint(actuallyPrint),
	hasToEndLine(true),
	hasToCrash(false),
	colorChanged(false),
	styleChanged(false)
      {
	if(threads::isInsideParallelSection() and not Logger::onlyMasterThreadPrint)
	  Logger::lock.acquire();
    
	startNewLine(actuallyPrint);
      }
      
      /// Move constructor
      LoggerLine(LoggerLine&& oth) :
	actuallyPrint(oth.actuallyPrint),
	hasToEndLine(true),
	hasToCrash(oth.hasToCrash),
	colorChanged(oth.colorChanged),
	styleChanged(oth.styleChanged)
	
      {
      	oth.hasToEndLine=
      	  false;
      }
      
      /// Ends the line
      void endLine()
      {
	if(actuallyPrint)
	  Logger::logFile<<'\n';
      }
      
      /// Destroy (end the line)
      ~LoggerLine()
      {
	if(not actuallyPrint)
	  return;
	
	// Wrap everything here
	if(hasToEndLine)
	  {
	    // Ends the quoted text
	    if(hasToCrash)
	      *this<<"\"";
	    
	    // Reset color
	    if(colorChanged)
	      *this<<TextColor::DEFAULT;
	    
	    // Reset style
	    if(styleChanged)
	      *this<<TextStyle::RESET;
	    
	    // Ends the line
	    endLine();
	    
	    if(threads::isInsideParallelSection() and not Logger::onlyMasterThreadPrint)
	      lock.release();
	    
	    if(hasToCrash)
	      {
		printBacktraceList();
		exit(1);
	      }
	  }
      }
      
      /// Prints after putting a space
      template <typename T>               // Type of the obected to print
      LoggerLine& operator*(T&& t)      ///< Object to be printed
      {
	if(actuallyPrint)
	  Logger::logFile*std::forward<T>(t);
	
	return
	  *this;
      }
      
      /// Catch-all print
      ///
      /// The SFINAE is needed to avoid that the method is used when
      /// File does not know how to print
      template <typename T,                                           // Type of the quantity to print
		ENABLE_THIS_TEMPLATE_IF(not canPrint<LoggerLine,T>    // SFINAE needed to avoid ambiguous overload
		and canPrint<File,const T&>)>
      LoggerLine& operator<<(const T& t)                             ///< Object to print
      {
	if(actuallyPrint)
	  Logger::logFile<<t;
	
	return
	  *this;
      }
      
      /// Print a C-style variadic message
      LoggerLine& printVariadicMessage(const char* format, ///< Format to print
				       va_list ap)         ///< Variadic part
      {
	if(actuallyPrint)
	  Logger::logFile.printVariadicMessage(format,ap);
	
	return
	  *this;
      }
      
      /// Changes the color of the line
      LoggerLine& operator<<(const TextColor& c)
      {
	colorChanged=
	  true;
	
	return
	  *this<<
	  TEXT_CHANGE_COLOR_HEAD<<
	  static_cast<char>(c)<<
	  TEXT_CHANGE_COLOR_TAIL;
      }
      
      /// Changes the style of the line
      LoggerLine& operator<<(const TextStyle& c)
      {
	styleChanged=
	  true;
	
	return
	  *this<<
	  TEXT_CHANGE_STYLE_HEAD<<
	  static_cast<char>(c)<<
	  TEXT_CHANGE_STYLE_TAIL;
      }
      
      /// Prints crash information
      ///
      /// Then sets the flag \c hasToCrash to true, such that at
      /// destroy of the line, crash happens
      LoggerLine& operator<<(const Crasher& cr)
      {
	this->hasToCrash=
	  true;
	
	return
	  (*this)<<TextColor::RED<<" ERROR in function "<<cr.getFuncName()<<" at line "<<cr.getLine()<<" of file "<<cr.getPath()<<": \"";
      }
      
      /// Prints a string, parsing newline
      LoggerLine& operator<<(const char* str)
      {
	if(not actuallyPrint)
	  return *this;
	
	if(str==nullptr)
	  Logger::logFile<<str;
	else
	  {
	    /// Pointer to the first char of the string
	    const char* p=
	      str;
	    
	    // Prints until finding end of string
	    while(*p!='\0')
	      {
		// starts a new line
		if(*p=='\n')
		  {
		    endLine();
		    startNewLine(actuallyPrint);
		  }
		else
		  // Prints the char
		  *this<<*p;
		
		// Increment the char
		p++;
	      }
	  }
	
	return
	  *this;
      }
      
      /// Prints a c++ string
      LoggerLine& operator<<(const std::string& str)
      {
	return
	  *this<<str.c_str();
      }
      
      /////////////////////////////////////////////////////////////////
      
      // /// Create a new line, and print on it
      // template <typename T,
      // 	      typename=EnableIf<canPrint<LoggerLine,T>>,            // SFINAE needed to avoid ambiguous overload
      // 	      typename=EnableIf<not canPrint<Logger,RemRef<T>>>>    // SFINAE to avoid ambiguous reimplementation
      // LoggerLine operator<<(T&& t)
      // {
      //   return
      // 	std::move(getNewLine()<<forw<T>(t));
      // }
    };
  }
  
  INLINE_FUNCTION
  Logger::LoggerLine logger(const int verbosityLv=0)
  {
    const bool actuallyPrint=
      ((Mpi::isMaster or not Logger::onlyMasterRankPrint) and
       (threads::isMaster() or not Logger::onlyMasterThreadPrint) and
       (verbosityLv<=Logger::verbosityLv));
    
    return actuallyPrint;
  }
  
  /// Shortcut to avoid having to put ()
#define LOGGER					\
  logger()
  
  /// Verbose logger or not, capital worded for homogeneity
#define VERBOSE_LOGGER(LV)			\
  logger(LV)
}

#endif
