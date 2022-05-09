#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <grill.hpp>

using namespace grill;

DECLARE_TRANSPOSABLE_COMP(Spin,int,4,spin);

// template <int I>
// constexpr void callS()
// {
// }

// template <typename I>
// void call(I i)
// {
//   auto c=[i](const auto& c,auto&& f,auto _j,auto _max) __attribute((always_inline))
//   {
//     using j=decltype(_j);
//     using max=decltype(_max);
    
//     // static_assert(j()<max(),"");
    
//     if(i==_j)
//       f(i);
//     else
//       if constexpr(j()<max())
// 	c(c,f,std::integral_constant<int,j()+1>(),_max);
//   };
  
//   c(c,[](const int i)
//   {
//     LOGGER<<i;
//   },std::integral_constant<int,3>(),std::integral_constant<int,10>());
// }

void test()
{
  // call(3);
  
  StackTens<OfComps<Spin>,
    std::vector<int>> s;
  
  s(Spin(0)).push_back(4);
  
  LOGGER<<s.storage<<" "<<s(Spin(0)).size();
}

int main(int narg,char**arg)
{
  grill::runProgram(narg,arg,[](int narg,char** arg){test();});
  
  return 0;
}
