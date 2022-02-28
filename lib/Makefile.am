# Include the header directory
AM_CPPFLAGS= \
	-I$(top_srcdir)/include

# Build objects in subdirectories
AUTOMAKE_OPTIONS= \
	subdir-objects

# Target to build the library
lib_LIBRARIES= \
	$(top_builddir)/lib/libesnort.a

%D%/gitInfo.hpp: $(top_srcdir)/.git/HEAD $(top_srcdir)/.git/index $(top_srcdir)/config/getGitInfo.sh Makefile
	@echo "Reading git version"
	@$(top_srcdir)/config/getGitInfo.sh $(top_srcdir) > $@
$(top_srcdir)/.git/HEAD:
$(top_srcdir)/.git/index:

BUILT_SOURCES=%D%/gitInfo.hpp

# Library sources
__top_builddir__lib_libesnort_a_SOURCES= \
	%D%/debug/backtracing.cpp \
	%D%/debug/demangle.cpp \
	%D%/debug/minimalCrash.cpp \
	%D%/ios/minimalLogger.cpp \
	%D%/resources/globalVariablesDefinitions.cpp \
	%D%/esnort.cpp

