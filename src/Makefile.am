include $(top_srcdir)/config/baseMakefile.am

LDADD=$(top_builddir)/lib/libesnort.a

AM_CPPFLAGS= \
	-I$(top_srcdir)/include

bin_PROGRAMS+= \
	$(top_builddir)/bin/testExecutionSpace

__top_builddir__bin_testExecutionSpace_SOURCES= \
	%D%/testExecutionSpace.cpp

assembly_reports+=\
        %D%/testExecutionSpace.s

#generate assembly report if asked
if ASSEMBLY_REPORT
BUILT_SOURCES+=$(assembly_reports)
CLEANFILES+=$(assembly_reports)
endif

