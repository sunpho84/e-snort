include $(top_srcdir)/config/baseMakefile.am

bin_PROGRAMS+= \
	$(top_builddir)/bin/main \
	$(top_builddir)/bin/testRefTransfer \
	$(top_builddir)/bin/testExecutionSpace

__top_builddir__bin_main_SOURCES= \
	%D%/main.cpp

__top_builddir__bin_testRefTransfer_SOURCES= \
	%D%/testRefTransfer.cpp

__top_builddir__bin_testExecutionSpace_SOURCES= \
	%D%/testExecutionSpace.cpp

assembly_reports+=\
        %D%/main.s \
        %D%/testExecutionSpace.s

#generate assembly report if asked
if ASSEMBLY_REPORT
BUILT_SOURCES+=$(assembly_reports)
CLEANFILES+=$(assembly_reports)
endif

