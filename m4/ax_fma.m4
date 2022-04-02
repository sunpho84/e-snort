# usage: AX_FMA
AC_DEFUN([AX_FMA], [

#introduce flags
AC_ARG_WITH(fma,
	AS_HELP_STRING([--with-fma], [Specify the flag to use to enable fma]),
	with_fma="${withval}",
	with_fma=-mfma)
AC_MSG_RESULT(with suggested fma flag ... ${with_fma})

AC_MSG_CHECKING([for fma])
for fma_flags in "" $with_fma
do
	SAVE_CXXFLAGS=$CXXFLAGS
	CXXFLAGS="$CXXFLAGS $fma_flags"
	
	AC_RUN_IFELSE(
		[AC_LANG_PROGRAM([[#include <immintrin.h>
		__m128d testa,testb,testc;
		]],
		   [[_mm_fnmadd_pd(testa,testb,testc);
		     ]])],
  	       [fma_available=yes],
	       [fma_available=no])
	CXXFLAGS=$SAVE_CXXFLAGS
	
	if test "${fma_available}" == "yes"
	then
		break
	fi
done
if test "${fma_available}" == "yes"
then
	AC_MSG_RESULT([yes, flag needed (if any): $fma_flags])
else
	AC_MSG_RESULT([no])
fi

#determine if enable the package
AX_GET_ENABLE(fma,${fma_available},[(automatically enabled if found)])

#check activability
if test "$enable_fma" == "yes"
then
	if test "${fma_available}" == "no"
	then
		AC_MSG_ERROR(["Cannot enable fma not supported, please provide/improve hint using --with-fma flag"])
	fi

	AC_DEFINE([ENABLE_FMA],1,[Enable fma])

	CXXFLAGS="$CXXFLAGS $fma_flags"
fi
])
