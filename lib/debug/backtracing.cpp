#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file backtracing.cpp
///
/// \brief Defines the backtracing function

#include <debug/backtracing.hpp>
#include <ios/logger.hpp>

namespace esnort
{
  /// Gets the backtracing symbols list
  std::vector<BackTraceSymbol> getBackTraceList()
  {
    constexpr int NMAX_REW=128;
    
    /// Rewinded stack
    void* callstack[NMAX_REW];
    
    /// Gets the stack and number of lines to be rewinded
    int nRew=
      backtrace(callstack,NMAX_REW);
    
    /// Gets the symbols list
    char** strs=
      backtrace_symbols(callstack,nRew);
    
    /// Result
    std::vector<BackTraceSymbol> res;
    res.reserve(nRew);
    
    for(int i=0;i<nRew;i++)
      res.emplace_back(strs[i]);
    
    free(strs);
    
    return
      res;
  }
  
  void printBacktraceList()
  {
    runLog()<<"Backtracing...";
    
    for(auto &p : getBackTraceList())
      runLog()<<p;
  }
}
