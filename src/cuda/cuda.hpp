#ifndef _CUDA_HPP
#define _CUDA_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

namespace esnort
{
  template <typename F>
  CUDA_GLOBAL
  void cuda_generic_kernel(F f)
  {
    f();
  }
  
  inline void decript_cuda_error(cudaError_t rc,const char *templ,...)
  {
    if(rc!=cudaSuccess)
      {
	va_list ap;
	va_start(ap,templ);
	char mess[1024];
	vsnprintf(mess,1024,templ,ap);
	va_end(ap);
	
	fprintf(stderr,"%s, cuda raised error: %s\n",mess,cudaGetErrorString(rc));
	exit(1);
      }
  }
  
  inline void freeCuda(void* ptr)
  {
    decript_cuda_error(cudaFree(ptr),"");
  }
}

#endif
