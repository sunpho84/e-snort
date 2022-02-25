#ifndef _ASSIGNDEVICETODEVICE_HPP
#define _ASSIGNDEVICETODEVICE_HPP

#include <assign/assignBase.hpp>
#include <assign/assignHostToHost.hpp>
#include <cuda/cuda.hpp>

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
      
      auto f=[=] CUDA_DEVICE (const int& i) mutable
      {
	return devLhs()=devRhs();
      };
      
      Cuda::launchKernel(f,0,1);
#endif
    }
  };
}

#endif
