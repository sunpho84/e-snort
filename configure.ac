AC_INIT([e-snort],[1.0],[fr.sanfilippo@gmail.com])

#include all config flags
config_flags="$*"
AC_DEFINE_UNQUOTED([CONFIG_FLAGS],["$config_flags"],[Flags passed to configure])

#silent automake
AM_SILENT_RULES([yes])

#get configuration time
AC_MSG_CHECKING([configuration date and time])
CONFIG_TIME=`date`
AC_DEFINE_UNQUOTED([CONFIG_TIME], "$CONFIG_TIME", [time when configured])
AC_MSG_RESULT([$CONFIG_TIME])

#parameters definition
AC_CONFIG_SRCDIR([src])
AC_CONFIG_AUX_DIR([config])
AM_INIT_AUTOMAKE([1.14 no-define foreign subdir-objects])
AM_CONFIG_HEADER(src/config.hpp)
AC_CONFIG_MACRO_DIR([m4])
AC_CANONICAL_HOST

#checks for programs
AC_PROG_CC
AC_PROG_FC
AC_PROG_CXX
AC_PROG_RANLIB
AC_LANG([C++])

AC_ARG_ENABLE(enforce-c++17,
	AS_HELP_STRING([--enable-enforce-c++17],[Enable enforcement of c++17]),
	enable_enforce_cpp17="${enableval}",
	enable_enforce_cpp17="yes")
AC_MSG_RESULT([enabling enforcing c++17... $enable_enforce_cpp17])
if test "$enable_enforce_cpp17" == "yes"
then
	AX_CXX_COMPILE_STDCXX(17,noext,mandatory)
else
	AX_CXX_COMPILE_STDCXX(17,noext,optional)
fi

LIBRARY_RESULT="---"
SUMMARY_RESULT="---"

AX_ASSEMBLY_REPORT

#check if we enable SIMD autosearch
AX_SIMD_TYPE(mmx,-mmmx,_mm_load_pd,_mm_store_pd,MMX)
AX_SIMD_TYPE(avx,-mavx,_mm256_load_pd,_mm256_store_pd,AVX)
AX_SIMD_TYPE(avx512,-mavx512f,_mm512_load_pd,_mm512_store_pd,AVX512)

#check if we enable SIMD
if test "$enable_mmx" == "yes" -o "$enable_avx" == "yes" -o "$enable_avx512" == "yes"
then
	default_enable_simd="yes"
else
	default_enable_simd="no"
fi

AC_ARG_ENABLE(SIMD,
	AS_HELP_STRING([--enable-SIMD],[Enable SIMD]),
	enable_SIMD="${enableval}",
	enable_SIMD="$default_enable_simd")
AC_MSG_RESULT([enabling SIMD... $enable_SIMD])

if test "$enable_SIMD" == "yes"
then
     ENABLE_SIMD=1
     CXXFLAGS="$CXXFLAGS $X86_FEATURE_CFLAGS"
else
     ENABLE_SIMD=0	
fi
AC_DEFINE_UNQUOTED([ENABLE_SIMD],${ENABLE_SIMD},[Enable SIMD])
SUMMARY_RESULT="$SUMMARY_RESULT
SIMD             : $enable_SIMD"

if test "$enable_SIMD" == "yes"
then
	AX_FMA
fi
SUMMARY_RESULT="$SUMMARY_RESULT
FMA              : $enable_FMA"

#check if we enable splitting MPI autosearch
AC_ARG_ENABLE(MPI-autosearch,
	AS_HELP_STRING([--enable-MPI-autosearch],[Enable MPI autosearch]),
	enable_MPI_autosearch="${enableval}",
	enable_MPI_autosearch="yes")
AC_MSG_RESULT([enabling MPI autosearch... $enable_MPI_autosearch])

#set MPI for c++
if test "$enable_MPI_autosearch" == "yes"
then
	AC_LANG_PUSH([C++])
	LX_FIND_MPI
	AC_LANG_POP([C++])
	AC_MSG_RESULT([with C++ MPI link options... $MPI_CXXLDFLAGS])
fi

#MPI
AC_ARG_ENABLE(MPI,
	AS_HELP_STRING([--disable-MPI],[Enable MPI]),
	enable_MPI="${enableval}",
	enable_MPI="yes")
AM_CONDITIONAL([ENABLE_MPI],[test "$enable_MPI" == "yes" ],[true],[false])
AC_MSG_RESULT([enabling MPI... $enable_MPI])
SUMMARY_RESULT="$SUMMARY_RESULT
MPI              : $enable_MPI"

if test "$enable_MPI" == "yes"
then
	# Check if we have MPI
	if test "$have_CXX_mpi" == "yes" -o "$enable_MPI_autosearch" != "yes"
	then
		CPPFLAGS="$MPI_CXXFLAGS $CPPFLAGS"
		LIBS="$MPI_CXXLIBS $LIBS"
		LDFLAGS="$MPI_CXXLDFLAGS $LDFLAGS"
		AC_DEFINE([ENABLE_MPI],1,[Enable MPI])
	else
		AC_MSG_ERROR(["Unable to find MPI"])
	fi
fi

#check openmp
AX_OPENMP

#threads
AC_ARG_ENABLE(threads,
	AS_HELP_STRING([--enable-threads],[Enable threads]),
	enable_threads="${enableval}",
	enable_threads="${have_openmp}")
AM_CONDITIONAL([THREADS],[test "$enable_threads" == "yes" ],[true],[false])
if test "$enable_threads" == "yes"
then
	ENABLE_THREADS=1
	CFLAGS="$CFLAGS $OPENMP_CFLAGS"
	CPPFLAGS="$CPPFLAGS $OPENMP_CPPFLAGS"
	CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"
else
	ENABLE_THREADS=0
fi
AC_DEFINE_UNQUOTED([ENABLE_THREADS],${ENABLE_THREADS},[Enable threads])
AC_MSG_RESULT([enabling threads... $enable_threads])
SUMMARY_RESULT="$SUMMARY_RESULT
Threads          : $enable_threads"

#cuda
AX_CHECK_CUDA

AH_VERBATIM([COMPILING_FOR_DEVICE],
#ifdef __CUDA_ARCH__
 
# define COMPILING_FOR_DEVICE 1
 
#else
 
# define COMPILING_FOR_DEVICE 0

#endif
)

#max verbosity lv
AC_ARG_ENABLE(max-verbosity-lv,
        AS_HELP_STRING([--enable-max-verbosity-lv=lv],[Set max verbosity lv]),
        max_verbosity_lv="${enableval}",
        max_verbosity_lv=2)
if ! printf "%f" "$max_verbosity_lv" >/dev/null 2>&1; then AC_MSG_ERROR([String $max_verbosity_lv is not a number]); fi
AC_DEFINE_UNQUOTED([MAX_VERBOSITY_LV],$max_verbosity_lv,[Max_verbosity_lv])
AC_MSG_RESULT([enabling max verbosity lv... $max_verbosity_lv])
SUMMARY_RESULT="$SUMMARY_RESULT
max-verbosity-lv : $max_verbosity_lv"

#demangle
AC_CHECK_HEADERS([cxxabi.h])
if test "$ac_cv_header_cxxabi_h" == "yes"
then
	CAN_DEMANGLE=1
else
	CAN_DEMANGLE=0
fi
AC_DEFINE_UNQUOTED([CAN_DEMANGLE],${CAN_DEMANGLE},[Enable demangling])
AC_MSG_RESULT([can demangle... $ac_cv_header_cxxabi_h])

#device code
AC_ARG_ENABLE(device-code,
	AS_HELP_STRING([--enable-device-code],[Enable device code]),
	enable_device_code="${enableval}",
	enable_device_code="no")
AM_CONDITIONAL([DEVICE_CODE],[test "$enable_device_code" == "yes" ],[true],[false])
if test "$enable_device_code" == "yes"
then
	ENABLE_DEVICE_CODE=1
	HOST_ATTRIB="__host__"
	DEVICE_ATTRIB="__device__"
	HOST_DEVICE_ATTRIB="__host__ __device__"
else
	ENABLE_DEVICE_CODE=0
fi
AC_DEFINE_UNQUOTED([ENABLE_DEVICE_CODE],${ENABLE_DEVICE_CODE},[Enable device code])
AC_DEFINE_UNQUOTED([HOST_ATTRIB],${HOST_ATTRIB},[Whether this function is HOST only])
AC_DEFINE_UNQUOTED([DEVICE_ATTRIB],${DEVICE_ATTRIB},[Whether to duplicate functions for DEVICE only])
AC_DEFINE_UNQUOTED([HOST_DEVICE_ATTRIB],${HOST_DEVICE_ATTRIB},[Whether to duplicate functions for host and device])
AH_VERBATIM([COMPILING_FOR_DEVICE],
#ifdef __CUDA_ARCH__
 
# define COMPILING_FOR_DEVICE 1

#else

# define COMPILING_FOR_DEVICE 0

#endif
)
AC_MSG_RESULT([enabling device code... $enable_device_code])
SUMMARY_RESULT="$SUMMARY_RESULT
Device code      : $enable_device_code"

if test "$enable_device_code" = "yes"
then
     case "$CXX" in
     	  nvcc) CPPFLAGS="-dc --expt-relaxed-constexpr --expt-extended-lambda -x cu $CPPFLAGS";;
     	  clang++) CPPFLAGS="-xcuda $CPPFLAGS";;
     	  *) AC_MSG_ERROR(["Unkwnown how to compile device code with $CXX"])
     esac
fi

#AX_CXXFLAGS_WARN_ALL
AC_MSG_CHECKING(whether compiler understands -Wall)
OLD_CXXFLAGS="$CXXFLAGS"
CXXFLAGS="$CXXFLAGS -Wall"
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],[])],
  AC_MSG_RESULT(yes),
  AC_MSG_RESULT(no)
  CXXFLAGS="$OLD_CXXFLAGS")

#macros to concatenate
AC_DEFINE([_CONCAT(X,Y)],[X##Y],[Internally concatenation])
AC_DEFINE([CONCAT(X,Y)],[_CONCAT(X,Y)],[Concatenate two pieces to produce a new token])
AC_DEFINE([CONCAT2(s1,s2)],[CONCAT(s1,s2)],[Wrapper to beat CPP])
AC_DEFINE([CONCAT3(s1,s2,s3)],[CONCAT(CONCAT2(s1,s2),s3)],[Concatenate three])
AC_DEFINE([NAME2(s1,s2)],[CONCAT3(s1,_,s2)],[Link with a _])
AC_DEFINE([NAME3(s1,s2,s3)],[NAME2(CONCAT3(s1,_,s2),s3)],[Name with two _])
AC_DEFINE([NAME4(s1,s2,s3,s4)],[NAME3(CONCAT3(s1,_,s2),s3,s4)],[Name with four _])
AC_DEFINE([REORDER_BARRIER()],[__asm volatile ("")],[Avoid spilling])
AC_DEFINE([MACRO_GUARD(...)],[do{__VA_ARGS__}while(0)],[Guard semicolon])
AC_DEFINE([TO_STRING_INTERNAL(x)],[#x],[Internally stringify])
AC_DEFINE([TO_STRING(x)],[TO_STRING_INTERNAL(x)],[Stringify])
AC_DEFINE([LINE_AS_STRING],[TO_STRING(__LINE__)],[Instantiate the line number in a string])
AC_DEFINE([ASM_BOOKMARK(COMMENT)],
	[asm(_ASM_BOOKMARK_SYMBOL "Bookmark file: \"" __FILE__ "\", line: " LINE_AS_STRING  ", " COMMENT)],
	[Include a comment in the assembler, recognizable in the compilation])
AC_DEFINE([ASM_BOOKMARK_BEGIN(COMMENT)],[ASM_BOOKMARK("BEGIN " COMMENT)],[Put a BEGIN for asm bookmark section])
AC_DEFINE([ASM_BOOKMARK_END(COMMENT)],[ASM_BOOKMARK("END " COMMENT)],[Put a END for asm bookmark section])

AH_VERBATIM([COMPILING_FOR_DEVICE],
#ifdef __CUDA_ARCH__
 
# define _ASM_BOOKMARK_SYMBOL "//"
 
#else
 
# define _ASM_BOOKMARK_SYMBOL "#"

#endif
)

# Configure doxygen
DX_INIT_DOXYGEN($PACKAGE_NAME,[config/Doxyfile],[docs])
DX_PDF_FEATURE(OFF)
DX_PS_FEATURE(OFF)
DX_MAN_FEATURE(ON)
DX_HTML_FEATURE(ON)
AC_SUBST([DOXYGEN_OUTPUT_DIR],[docs])

#result of conf to screen
AC_MSG_RESULT([
--------------------------------------------------
Configuration summary for $PACKAGE_NAME ($VERSION)
--------------------------------------------------
CC                  : $CC
CPPFLAGS            : $CPPFLAGS
CXXFLAGS            : $CXXFLAGS
CXX                 : $CXX
---
LDFLAGS             : $LDFLAGS
LIBS                : $LIBS
---
Installation prefix : $prefix
$LIBRARY_RESULT
$SUMMARY_RESULT
--------------------------------------------------
Configuration OK
--------------------------------------------------
])

#create config.hpp
AC_CONFIG_FILES(
	Makefile
	lib/Makefile
	src/Makefile
        config/Doxyfile)

AC_OUTPUT
