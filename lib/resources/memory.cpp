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
    manager<ExecutionSpace::HOST>.initialize();
#if ENABLE_DEVICE_CODE
    manager<ExecutionSpace::DEVICE>.initialize();
#endif
  }
  
  void finalize()
  {
    manager<ExecutionSpace::HOST>.finalize();
#if ENABLE_DEVICE_CODE
    manager<ExecutionSpace::DEVICE>.finalize();
#endif
  }
}
