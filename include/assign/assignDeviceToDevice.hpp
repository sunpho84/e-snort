#ifndef _ASSIGNDEVICETODEVICE_HPP
#define _ASSIGNDEVICETODEVICE_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file assignDevicetoDevice.hpp

#include <assign/assignBase.hpp>
#include <assign/assignHostToHost.hpp>
#include <resources/device.hpp>

namespace esnort
{
  template <WhichSideToChange W>
  struct Assign<ExecutionSpace::DEVICE,ExecutionSpace::DEVICE,W>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs)
    {
#if not ENABLE_DEVICE_CODE
      Assign<ExecutionSpace::HOST,ExecutionSpace::HOST,WhichSideToChange::LHS>::exec(std::forward<Lhs>(lhs),std::forward<Rhs>(rhs));
#else
      runLog()<<"Launching the kernel D to D";
      
      auto devLhs=lhs.getRef();
      const auto devRhs=rhs.getRef();
      
      DEVICE_LOOP(i,0,1,
		  return devLhs()=devRhs();
		  );
#endif
    }
  };
}

#endif
