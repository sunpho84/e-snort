#ifndef _ALIVER_HPP
#define _ALIVER_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file aliver.hpp

#include "../lib/gitInfo.hpp"

#include <ios/logger.hpp>
#include <metaprogramming/singleInstance.hpp>

namespace esnort
{
  /// Class used to provocate initialization of Mpi
  class Aliver :
    public SingleInstance<Aliver>
  {
    
    /// Prints the banner
    void printBanner()
      const
    {
      runLog()<<"";
      runLog()<<TextColor::BROWN<<"          ▄▄        ▄█▄        ▄▄        \t"<<TextColor::PURPLE<< "                 ▄█▄                  ";
      runLog()<<TextColor::BROWN<<"          █░█       █░█       █░█         \t"<<TextColor::PURPLE<< "                 █░█                   ";
      runLog()<<TextColor::BROWN<<"     ▄▄    █░█      █░█      █░█    ▄▄   \t"<<TextColor::PURPLE<<  "                 █░█                  ";
      runLog()<<TextColor::BROWN<<"     █░█    █░█     █░█     █░█    █░█   \t"<<TextColor::PURPLE<<  "                 █░█                  ";
      runLog()<<TextColor::BROWN<<"      █░█    █░█  ███████  █░█    █░█    \t"<<TextColor::PURPLE<<  "               ███████                ";
      runLog()<<TextColor::BROWN<<"       █░█    █████░░░░░█████    █░█     \t"<<TextColor::PURPLE<<  "           █████░█░█░█████            ";
      runLog()<<TextColor::BROWN<<"        █░█  ██░░░░░░░░░░░░░██  █░█      \t"<<TextColor::PURPLE<<  "          ██░░░░░█░█░░░░░██           ";
      runLog()<<TextColor::BROWN<<"         █░██░░░░░░░░░░░░░░░░░██░█       \t"<<TextColor::PURPLE<<  "        ██░░░░░░░█░█░░░░░░░██         ";
      runLog()<<TextColor::BROWN<<"    ▄▄▄▄▄▄███████████░███████████▄▄▄▄▄▄ \t" <<TextColor::PURPLE<< "       ██░░░░░░░░█░█░░░░░░░░██        ";
      runLog()<<TextColor::BROWN<<"   █░░░░░░█░████████░░░████████░█░░░░░░█ \t"<<TextColor::PURPLE<<  "       █░░░░░░░░░█░█░░░░░░░░░█        ";
      runLog()<<TextColor::BROWN<<"    ▀▀▀▀▀▀█░░░████░░░░░░░████░░░█▀▀▀▀▀▀ \t" <<TextColor::PURPLE<< "       █░░░░░░░░░█░█░░░░░░░░░█        ";
      runLog()<<TextColor::BROWN<<"          ██░░░░░░░░░░░░░░░░░░░░█        \t"<<TextColor::PURPLE<<  "       ██░░░░░░░░█░█░░░░░░░░░█        ";
      runLog()<<TextColor::BROWN<<"         █░██░░░░░███████░░░░░░█░█       \t"<<TextColor::PURPLE<<  "        ██░░░░░░░█░█░░░░░░░░█         ";
      runLog()<<TextColor::BROWN<<"        █░█  █░░░░░░░░░░░░░░░██ █░█      \t"<<TextColor::PURPLE<<  "          █░░░░░░█░█░░░░░░██          ";
      runLog()<<TextColor::BROWN<<"       █░█    ██░░░░░░░░░░░██    █░█     \t"<<TextColor::PURPLE<<  "           ██░░░░█░█░░░░██            ";
      runLog()<<TextColor::BROWN<<"      █░█     █░███████████░█     █░█    \t"<<TextColor::PURPLE<<  "             ███████████              ";
      runLog()<<TextColor::BROWN<<"     █░█     █░█    █░█    █░█     █░█   \t"<<TextColor::PURPLE<<  "                 █░█                  ";
      runLog()<<TextColor::BROWN<<"     ▀▀     █░█     █░█     █░█     ▀▀  \t" <<TextColor::PURPLE<<  "                 █░█                  ";
      runLog()<<TextColor::BROWN<<"           █░█      █░█      █░█        \t" <<TextColor::PURPLE<<  "                 █░█                 ";
      runLog()<<TextColor::BROWN<<"          █░█       █░█       █░█       \t" <<TextColor::PURPLE<<  "                 █░█                 ";
      runLog()<<TextColor::BROWN<<"          ▀▀        ▀█▀        ▀▀       \t" <<TextColor::PURPLE<< "                 ▀█▀                ";
      runLog()<< "";
    }
    
    /// Prints the version, and contacts
    void printVersionContacts()
      const
    {
      runLog()<<"\nInitializing "<<PACKAGE_NAME<<" library, v"<<PACKAGE_VERSION<<", send bug report to <"<<PACKAGE_BUGREPORT<<">";
    }
    
    /// Prints the git info
    void printGitInfo()
      const
    {
      runLog()<<"Commit "<<GIT_HASH<<" made at "<<GIT_TIME<<" by "<<GIT_COMMITTER<<" with message: \""<<GIT_LOG<<"\"";
    }
    
    /// Prints configure info
    void printConfigurePars()
      const
    {
      runLog()<<"Configured at "<<CONFIG_TIME<<" with flags: "<<CONFIG_FLAGS<<"";
    }
    
    /// Says bye bye
    void printBailout()
      const
    {
      runLog()<<"\n Ciao!\n";
    }
    
  public:
    
    /// Creates
    Aliver()
    {
      printBanner();
      printVersionContacts();
      printGitInfo();
      printConfigurePars();
      
      // threads.workOn([](const int threadID){runLog()<<"ANNA";});
      
      // {
      // 	ALLOWS_ALL_THREADS_TO_PRINT_FOR_THIS_SCOPE(runLog);
      // 	ALLOWS_ALL_RANKS_TO_PRINT_FOR_THIS_SCOPE(runLog);
	
      // 	threads.workOn([](const int threadID){runLog()<<"ANNA";});
      // }
      // threads.loopSplit(0,10,[](const int& rank,const int& i){printf("Rank %d prints again %d\n",rank,i);});
    }
    
    /// Destroyer
    ~Aliver()
    {
      printBailout();
    }
  };
}

#endif
