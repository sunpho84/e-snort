#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file environmentFlags.cpp
///
/// \brief Routines needed to read environment variables

#include <sstream>

#include <debug/attachDebuggerGlobalVariablesDeclarations.hpp>
#include <ios/logger.hpp>
#include <metaprogramming/forEachInTuple.hpp>
#include <resources/memory.hpp>

namespace esnort::envFlags
{
  void readAll()
  {
    LOGGER;
    LOGGER<<"Flags ";
    
    SCOPE_INDENT();
    
#define ADD_FLAG(NAME,DEFAULT,TAG,DESCRIPTION)		\
    std::make_tuple(&NAME,DEFAULT,TAG,DESCRIPTION)
    
    const auto flagList=
      std::make_tuple(ADD_FLAG(waitToAttachDebugger,false,"WAIT_TO_ATTACH_DEBUGGER","to be used to wait for gdb to attach"),
		      ADD_FLAG(Logger::verbosityLv,1,"VERBOSITY_LV","Level of verbosity of the program"),
		      ADD_FLAG(memory::useCache,true,"USE_CACHE","Use memory cache")
		      );
    
    forEachInTuple(flagList,[](auto& f)
			  {
			    /// Flag to be parsed
			    auto& flag=*std::get<0>(f);
			    
			    /// Default value
			    const auto& def=std::get<1>(f);
			    
			    /// Tag to be used to parse
			    const char* tag=std::get<2>(f);
			    
			    /// Description to be used to parse
			    const char* descr=std::get<3>(f);
			    
			    LOGGER<<"- "<<tag<<" ("<<descr<<") ";
			    
			    /// Search for the tag
			    const char* p=getenv(tag);
			    
			    if(p!=NULL)
			      {
				/// Convert from string
				std::istringstream is(p);
				is>>flag;
				
				LOGGER<<"  changed from default value "<<def<<" to: "<<flag;
			      }
			    else
			      {
				LOGGER<<"  set to default value: "<<def;
				flag=def;
			      }
			  });
  }
}
