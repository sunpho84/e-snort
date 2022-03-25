#include <resources/memory.hpp>
#include <expr/executionSpace.hpp>

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file memory.cpp

namespace esnort::memory
{
  void initialize()
  {
    manager<ExecSpace::HOST>.initialize();
#if ENABLE_DEVICE_CODE
    manager<ExecSpace::DEVICE>.initialize();
#endif
  }
  
  void finalize()
  {
    manager<ExecSpace::HOST>.finalize();
#if ENABLE_DEVICE_CODE
    manager<ExecSpace::DEVICE>.finalize();
#endif
  }
}
