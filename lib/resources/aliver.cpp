#include "resources/memory.hpp"
#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file aliver.cpp

#include <signal.h>

#include <debug/attachDebugger.hpp>
#include <debug/signalTrap.hpp>
#include <ios/logger.hpp>
#include <metaprogramming/singleInstance.hpp>
#include <resources/aliver.hpp>
#include <resources/device.hpp>
#include <resources/environmentFlags.hpp>
#include <resources/gitInfo.hpp>

#include <resources/Mpi.hpp>
#include <resources/threads.hpp>

namespace esnort
{
  /// Prints the banner
  void printBanner()
  {
    LOGGER<<"";
    LOGGER<<TextColor::BROWN<<"          ▄▄        ▄█▄        ▄▄        \t"<<TextColor::PURPLE<< "                 ▄█▄                  ";
    LOGGER<<TextColor::BROWN<<"          █░█       █░█       █░█         \t"<<TextColor::PURPLE<< "                 █░█                   ";
    LOGGER<<TextColor::BROWN<<"     ▄▄    █░█      █░█      █░█    ▄▄   \t"<<TextColor::PURPLE<<  "                 █░█                  ";
    LOGGER<<TextColor::BROWN<<"     █░█    █░█     █░█     █░█    █░█   \t"<<TextColor::PURPLE<<  "                 █░█                  ";
    LOGGER<<TextColor::BROWN<<"      █░█    █░█  ███████  █░█    █░█    \t"<<TextColor::PURPLE<<  "               ███████                ";
    LOGGER<<TextColor::BROWN<<"       █░█    █████░░░░░█████    █░█     \t"<<TextColor::PURPLE<<  "           █████░█░█░█████            ";
    LOGGER<<TextColor::BROWN<<"        █░█  ██░░░░░░░░░░░░░██  █░█      \t"<<TextColor::PURPLE<<  "          ██░░░░░█░█░░░░░██           ";
    LOGGER<<TextColor::BROWN<<"         █░██░░░░░░░░░░░░░░░░░██░█       \t"<<TextColor::PURPLE<<  "        ██░░░░░░░█░█░░░░░░░██         ";
    LOGGER<<TextColor::BROWN<<"    ▄▄▄▄▄▄███████████░███████████▄▄▄▄▄▄ \t" <<TextColor::PURPLE<< "       ██░░░░░░░░█░█░░░░░░░░██        ";
    LOGGER<<TextColor::BROWN<<"   █░░░░░░█░████████░░░████████░█░░░░░░█ \t"<<TextColor::PURPLE<<  "       █░░░░░░░░░█░█░░░░░░░░░█        ";
    LOGGER<<TextColor::BROWN<<"    ▀▀▀▀▀▀█░░░████░░░░░░░████░░░█▀▀▀▀▀▀ \t" <<TextColor::PURPLE<< "       █░░░░░░░░░█░█░░░░░░░░░█        ";
    LOGGER<<TextColor::BROWN<<"          ██░░░░░░░░░░░░░░░░░░░░█        \t"<<TextColor::PURPLE<<  "       ██░░░░░░░░█░█░░░░░░░░░█        ";
    LOGGER<<TextColor::BROWN<<"         █░██░░░░░███████░░░░░░█░█       \t"<<TextColor::PURPLE<<  "        ██░░░░░░░█░█░░░░░░░░█         ";
    LOGGER<<TextColor::BROWN<<"        █░█  █░░░░░░░░░░░░░░░██ █░█      \t"<<TextColor::PURPLE<<  "          █░░░░░░█░█░░░░░░██          ";
    LOGGER<<TextColor::BROWN<<"       █░█    ██░░░░░░░░░░░██    █░█     \t"<<TextColor::PURPLE<<  "           ██░░░░█░█░░░░██            ";
    LOGGER<<TextColor::BROWN<<"      █░█     █░███████████░█     █░█    \t"<<TextColor::PURPLE<<  "             ███████████              ";
    LOGGER<<TextColor::BROWN<<"     █░█     █░█    █░█    █░█     █░█   \t"<<TextColor::PURPLE<<  "                 █░█                  ";
    LOGGER<<TextColor::BROWN<<"     ▀▀     █░█     █░█     █░█     ▀▀  \t" <<TextColor::PURPLE<<  "                 █░█                  ";
    LOGGER<<TextColor::BROWN<<"           █░█      █░█      █░█        \t" <<TextColor::PURPLE<<  "                 █░█                 ";
    LOGGER<<TextColor::BROWN<<"          █░█       █░█       █░█       \t" <<TextColor::PURPLE<<  "                 █░█                 ";
    LOGGER<<TextColor::BROWN<<"          ▀▀        ▀█▀        ▀▀       \t" <<TextColor::PURPLE<< "                 ▀█▀                ";
    LOGGER<< "";
  }
  
  /// Prints the version, and contacts
  void printVersionContacts()
  {
    LOGGER<<"\nInitializing "<<PACKAGE_NAME<<" library, v"<<PACKAGE_VERSION<<", send bug report to <"<<PACKAGE_BUGREPORT<<">";
  }
  
  /// Prints the git info
  void printGitInfo()
  {
    using namespace git;
    
    LOGGER<<"Commit "<<hash<<" made at "<<time<<" by "<<committer<<" with message: \""<<log<<"\"";
  }
  
  /// Prints configure info
  void printConfigurePars()
  {
    LOGGER<<"Configured at "<<CONFIG_TIME<<" with flags: "<<CONFIG_FLAGS<<"";
  }
  
  void printAvailableFeatures()
  {
#if ENABLE_AVX512
    LOGGER<<"Avx512 availale";
#endif

#if ENABLE_AVX
    LOGGER<<"Avx availale";
#endif
    
#if ENABLE_MMX
    LOGGER<<"Mmx availale";
#endif
  }
  
  /// Says bye bye
  void printBailout()
  {
    LOGGER<<"\n Ciao!\n";
  }
  
  void initialize(int narg,char** arg)
  {
    Mpi::initialize();
    
    setSignalTraps();
    
    printBanner();
    printVersionContacts();
    printGitInfo();
    printConfigurePars();
    printAvailableFeatures();
    
    envFlags::readAll();
    
    possiblyWaitToAttachDebugger();
    memory::initialize();
    threads::initialize();
  }
  
  void finalize()
  {
    memory::finalize();
    device::finalize();
    Mpi::finalize();
    printBailout();
    
  }
}
