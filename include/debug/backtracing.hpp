#ifndef _BACKTRACING_HPP
#define _BACKTRACING_HPP

/// \file backtracing.hpp
///
/// \brief Defines the backtracing function

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <regex>
#include <vector>
#include <execinfo.h>

#include <debug/demangle.hpp>

namespace esnort
{
  /// Decompose the backtrace symbol
  struct BackTraceSymbol
  {
    /// Compilation unit
    std::string compilUnit;
    
    /// Symbol name
    std::string symbol;
    
    /// Offset
    std::string offset;
    
    /// Address
    std::string address;
    
    /// Parse the string
    explicit BackTraceSymbol(const std::string& str)
    {
      const std::regex regex("[(\[+() \\]]");
      const std::sregex_token_iterator first{str.begin(),str.end(),regex,-1},last;
      const std::vector<std::string> tokens{first,last};
      
      if(tokens.size()>0) compilUnit=tokens[0];
      if(tokens.size()>1) symbol=tokens[1];
      if(tokens.size()>2) offset=tokens[2];
      if(tokens.size()>5) address=tokens[5];
    }
  };
  
  /// Gets the backtracing symbols list
  std::vector<BackTraceSymbol> getBackTraceList();
  
  /// Print a symbol to a stream
  template <typename T>                     // Type of the stream
  T& operator<<(T&& os,                    ///< Stream
		const BackTraceSymbol& s)  ///< Symbol to print
  {
    return
      os<<s.compilUnit<<
      "  symbol: "<<
      ((s.symbol!="")?
#if CAN_DEMANGLE
       demangle(s.symbol)
#else
       s.symbol
#endif
       :"n.a")<<
      " offset: "<<((s.offset!="")?s.offset:"n.a")<<
      " address: "<<s.address<<" -- addr2line -e "<<s.compilUnit<<" "<<s.offset;
  }
  
  /// Write the list of called routines
  void printBacktraceList();
}

#endif
