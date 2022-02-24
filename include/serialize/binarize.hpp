#ifndef _BINARIZE_HPP
#define _BINARIZE_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file binarize.hpp
///
/// \brief Converts to and from binary

#include <cstring>
#include <string>
#include <vector>

#include <ios/minimalLogger.hpp>
#include <metaprogramming/crtp.hpp>
#include <metaprogramming/hasMember.hpp>
#include <metaprogramming/templateEnabler.hpp>

namespace esnort
{
  PROVIDE_HAS_MEMBER(binarize);
  PROVIDE_HAS_MEMBER(deBinarize);
  
  /// Determine whether a type is serializable
  template <typename T>
  [[ maybe_unused ]]
  constexpr bool isBinarizable=
    hasMember_binarize<T> and hasMember_deBinarize<T>;
  
  /// Class to convert objects to and from binary
  class Binarizer
  {
    /// Buffer
    std::vector<char> buf;
    
    /// Reading position
    size_t readPos{0};
    
    /// Push data on the back
    Binarizer& pushBack(const void* data,   ///< Data to be pushed
			const size_t& size) ///< Data size
    {
      buf.insert(buf.end(),(char*)data,(char*)data+size);
      
      return
	*this;
    }
    
    /// Push data on the back
    Binarizer& readAdvancing(void* out,          ///< Data to be filled
			     const size_t& size) ///< Data size
    {
      memcpy(out,&buf[readPos],size);
      
      readPos+=
	size;
      
      return
	*this;
    }
    
  public:
    
    /// Used size
    size_t size()
      const
    {
      return
	buf.size();
    }
    
    /// Add both begin and end method, with and not const
#define PROVIDE_BEGIN_END(CV)				\
    /*! Pointer to the beginning of the buffer */	\
    CV auto begin()					\
      CV						\
    {							\
      return						\
	buf.begin();					\
    }							\
							\
    /*! Pointer to the end of the buffer */		\
    CV auto end()					\
      CV						\
    {							\
      return						\
	buf.end();					\
    }
    
    PROVIDE_BEGIN_END(/* */);
    PROVIDE_BEGIN_END(const);
    
#undef PROVIDE_BEGIN_END
    
    /// Write on the binarizer, if the type has a member binarize
    template <typename T,
	      ENABLE_THIS_TEMPLATE_IF(hasMember_binarize<T>)>
    Binarizer& binarize(const T& rhs)
    {
      return
	rhs.binarize(*this);
    }
    
    /// Read from the binarizer, if the type has a member deBinarize
    template <typename T,
	      ENABLE_THIS_TEMPLATE_IF(hasMember_deBinarize<T>)>
    Binarizer& deBinarize(T& rhs)
    {
      return
	rhs.deBinarize(*this);
    }
    
    /// Write on the binarizer
    template <typename T,
	      ENABLE_THIS_TEMPLATE_IF(std::is_trivially_copyable_v<T>)>
    Binarizer& binarize(const T& rhs)
    {
      return
	pushBack(&rhs,sizeof(T));
    }
    
    /// Read from the binarizer
    template <typename T,
	      ENABLE_THIS_TEMPLATE_IF(std::is_trivially_copyable_v<T>)>
    Binarizer& deBinarize(T& rhs)
    {
      return
	readAdvancing(&rhs,sizeof(T));
    }
    
    // /// Binarize a tuple-like
    // template <typename T,
    // 	      ENABLE_THIS_TEMPLATE_IF(isTupleLike<T>)>
    // Binarizer& binarize(T&& rhs)     ///< Input
    // {
    //    forEach(rhs,
    // 	      [this](auto& s)
    // 	      {
    // 		this->binarize(s);
    // 	      });
      
    //   return
    // 	*this;
    // }
    
    // /// DeBinarize a tuple-like
    // template <typename T,
    // 	      ENABLE_THIS_TEMPLATE_IF((isTupleLike<T>)>
    // Binarizer& deBinarize(T&& rhs)     ///< Output
    // {
    //   forEach(rhs,
    // 	      [this](auto& s)
    // 	      {
    // 		this->deBinarize(s);
    // 	      });
      
    //   return
    // 	*this;
    // }
    
    // /// Binarize a vector-like
    // template <typename T,
    // 	      ENABLE_THIS_TEMPLATE_IF((isVectorLike<T>)>
    // Binarizer& binarize(T&& rhs)     ///< Input
    // {
    //   /// Number of elements
    //   const size_t nel=
    // 	rhs.size();
      
    //   this->binarize(nel);
      
    //   for(size_t iel=0;iel<nel;iel++)
    // 	this->binarize(rhs[iel]);
      
    //   return
    // 	*this;
    // }
    
    // /// DeBinarize a vector-like
    // template <typename T,
    // 	      ENABLE_THIS_TEMPLATE_IF((isVectorLike<T>)>
    // Binarizer& deBinarize(T&& rhs)     ///< Output
    // {
    //   /// Number of elements
    //   size_t nel;
      
    //   this->deBinarize(nel);
      
    //   rhs.resize(nel);
      
    //   for(size_t iel=0;iel<nel;iel++)
    // 	this->deBinarize(rhs[iel]);
      
    //   return
    // 	*this;
    // }
    
    /// Restart from head
    void restartReading()
    {
      readPos=
	0;
    }
    
    /// Restart to write
    void clear()
    {
      buf.clear();
      restartReading();
    }
  };
  
  DEFINE_CRTP_INHERITANCE_DISCRIMINER_FOR_TYPE(Binarizable)
  
  /// Add binarizable functionality via CRTP
  template <typename T>
  class Binarizable :
    Crtp<T,crtp::BinarizableDiscriminer>
  {
  public:
    
    /// Binarize a Serializable
    template <typename B=Binarizer>
    Binarizer& binarize(B&& out={})             ///< Output
      const
    {
      return
	out.binarize(CRTP_THIS());
    }
    
    /// DeBinarize a Serializable
    template <typename B=Binarizer>
    Binarizer& deBinarize(B&& rhs)               ///< Input
    {
      return
	rhs.deBinarize(CRTP_THIS());
    }
  };
}

#endif
