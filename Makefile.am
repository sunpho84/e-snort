#include m4 dir
ACLOCAL_AMFLAGS=-I m4

#Doxygen rules
@DX_RULES@

include $(top_srcdir)/config/baseMakefile.am

#include subdirs
SUBDIRS= \
	src

#generate assembly report if asked
if ASSEMBLY_REPORT
BUILT_SOURCES+=$(assembly_reports)
CLEANFILES+=$(assembly_reports)
endif

#adds extra files to distribution
EXTRA_DIST= \
	config \
	README.md

#Clean also Doxygen files
MOSTLYCLEANFILES= \
        ${DX_CLEANFILES}
