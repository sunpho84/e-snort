#ifndef _ESNORT_HPP
#define _ESNORT_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file esnort.hpp

#include <debug/backtracing.hpp>
#include <debug/attachDebugger.hpp>
#include <debug/crash.hpp>
#include <debug/demangle.hpp>
#include <debug/minimalCrash.hpp>

#include <expr/comp.hpp>
#include <expr/compRwCl.hpp>
#include <expr/conj.hpp>
#include <expr/dagger.hpp>
#include <expr/deviceAssign.hpp>
#include <expr/directAssign.hpp>
#include <expr/dynamicTens.hpp>
#include <expr/indexComputer.hpp>
#include <expr/stackTens.hpp>
#include <expr/threadAssign.hpp>
#include <expr/transp.hpp>

#include <ios/file.hpp>
#include <ios/logger.hpp>
#include <ios/minimalLogger.hpp>

#include <metaprogramming/arithmeticOperatorsViaCast.hpp>
#include <metaprogramming/unrolledFor.hpp>

#include <resources/aliver.hpp>
#include <resources/SIMD.hpp>

#include <tuples/tupleDiscriminate.hpp>

/// Global namespace
namespace esnort
{
}

#endif
