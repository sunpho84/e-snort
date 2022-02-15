#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <cstdio>
#include <type_traits>

template <typename IMin,
	  typename IMax,
	  typename F>
__global__
void cuda_generic_kernel(const IMin min,
			 const IMax max,
			 F f)
{
  const auto i=
    min+blockIdx.x*blockDim.x+threadIdx.x;
  
  if(i<max)
    f(i);
}

inline void thread_barrier_internal()
{
#ifdef COMPILING_FOR_DEVICE
  cudaDeviceSynchronize();
#endif
}

template <typename IMin,
	  typename IMax,
	  typename F>
void cuda_parallel_for(const int line,
		       const char *file,
		       const IMin min,
		       const IMax max,
		       F f)
{
  const auto length=(max-min);
  const dim3 block_dimension(128);
  int i=(length+block_dimension.x-1)/block_dimension.x;
  const dim3 grid_dimension(i);
  
  extern int verbosity_lv;
  printf("at line %d of file %s launching kernel on loop [%ld,%ld) using blocks of size %d and grid of size %d\n",
	 line,file,(int64_t)min,(int64_t)max,block_dimension.x,grid_dimension.x);
  
  cuda_generic_kernel<<<grid_dimension,block_dimension>>>(min,max,std::forward<F>(f));
  thread_barrier_internal();
  
  printf(" finished\n");
}

#ifdef __NVCC__
  #ifndef __CUDA_ARCH__
__host__ constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
  #else
__device__ constexpr std::integral_constant<bool,true> isOnDevice() { return {}; }
  #endif
#else
__host__ constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
__device__ constexpr std::integral_constant<bool,true> isOnDevice() { return {}; }
#endif

struct A
{
  double data[3]={};
  
#ifdef __NVCC__
  #ifndef __CUDA_ARCH__
__host__ static constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
  #else
__device__ static constexpr std::integral_constant<bool,true> isOnDevice() { return {}; }
  #endif
#else
__host__ static constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
__device__ static constexpr std::integral_constant<bool,true> isOnDevice() { return {}; }
#endif
  //   static constexpr auto isOnDevice() __host__ __device__
  // {
  //   return ::isOnDevice();
  // }
};

//__managed__
template <typename I>
static __managed__ int iss;

template <typename T>
struct Ref
{
  const T& t;
  
  Ref(const T& t) : t(t)
  {
  }
};

int main()
{
  //iss<int> =1;
  
  A j;
  
  Ref ref(j);
  
  printf("%d\n",A::isOnDevice()());
  
//   auto e=std::integral_constant<bool,A::isOnDevice()>{};
// #ifndef __CUDA_ARCH__
//   int* r=A::isOnDevice();
//  #endif
    
//  #ifndef __CUDA_ARCH__
//     int* r=(decltype(j)::isOnDevice());
// #endif
    cuda_parallel_for(__LINE__,__FILE__,0,12,[ref]  __device__ (const int i)
  {
    //int* r=(std::decay_t<decltype(ref.t)>::isOnDevice());
    // auto e=std::integral_constant<bool,A::isOnDevice()>{};
    //int* r=isOnDevice();
    // i+j.data[0];
  });
  
  return 0;
}
