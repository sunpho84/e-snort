include $(top_srcdir)/config/baseMakefile.am

LDADD=$(top_builddir)/lib/libesnort.a

AM_CPPFLAGS= \
	-I$(top_srcdir)/include

# bin_PROGRAMS+= 
# 	$(top_builddir)/bin/testExecutionSpace

bin_PROGRAMS+= \
	$(top_builddir)/bin/smallTest

# __top_builddir__bin_testExecutionSpace_SOURCES= 
# 	%D%/testExecutionSpace.cpp

__top_builddir__bin_smallTest_SOURCES= \
	%D%/smallTest.cpp

assembly_reports+=\
        %D%/testExecutionSpace.s \
        %D%/smallTest.s

#generate assembly report if asked
if ASSEMBLY_REPORT
BUILT_SOURCES+=$(assembly_reports)
CLEANFILES+=$(assembly_reports)
endif

