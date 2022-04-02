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

#include <expr/comps/comp.hpp>
#include <expr/comps/compRwCl.hpp>
#include <expr/nodes/conj.hpp>
#include <expr/operations/dagger.hpp>
#include <expr/assign/deviceAssign.hpp>
#include <expr/assign/directAssign.hpp>
#include <expr/nodes/dynamicTens.hpp>
#include <expr/comps/indexComputer.hpp>
#include <expr/nodes/prod.hpp>
#include <expr/nodes/sum.hpp>
#include <expr/nodes/stackTens.hpp>
#include <expr/assign/threadAssign.hpp>
#include <expr/nodes/transp.hpp>

#include <ios/file.hpp>
#include <ios/logger.hpp>
#include <ios/minimalLogger.hpp>

#include <metaprogramming/arithmeticOperatorsViaCast.hpp>
#include <metaprogramming/unrolledFor.hpp>

#include <resources/aliver.hpp>
#include <resources/SIMD.hpp>

#include <tuples/tupleDiscriminate.hpp>
#include <tuples/tupleSubset.hpp>
#include <tuples/tupleCat.hpp>
#include <tuples/uniqueTuple.hpp>
#include <tuples/uniqueTupleFromTuple.hpp>

/// Global namespace
namespace esnort
{
}

#endif
