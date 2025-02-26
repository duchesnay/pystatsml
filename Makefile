# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = build
NTBOOK        = $(shell ls data_manipulation/*.ipynb numerical_methods/*.ipynb statistics/*.ipynb statistics/lmm/lmm.ipynb ml_supervised/*.ipynb ml_unsupervised/*.ipynb deep_learning/*.ipynb nlp/*.ipynb)
# Notebook to execute. Exclude DL file (requires GPU)
NTBOOK_TO_EXE = $(shell ls data_manipulation/*.ipynb numerical_methods/*.ipynb statistics/*.ipynb statistics/lmm/lmm.ipynb ml_supervised/*.ipynb ml_unsupervised/*.ipynb nlp/*.ipynb)

#NTBOOK        = $(shell ls statistics/*.ipynb)
NTBOOK_FILES  = $(NTBOOK:.ipynb=_files)
#SRC           = $(shell ls python/*.py)
RST           = $(NTBOOK:.ipynb=.rst) $(SRC:.py=.rst)
#$(info $(NTBOOK))
#$(info $(RST))
#$(info $(NTBOOK_FILES))
#$(info $(PYTORST))

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed, then set the SPHINXBUILD environment variable to point to the full path of the '$(SPHINXBUILD)' executable. Alternatively you can add the directory with the executable to your PATH. If you don't have Sphinx installed, grab it from http://sphinx-doc.org/)
endif

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .



#$(shell find notebooks -name "*.ipynb" -exec bash -c -exec sh -c 'echo "$${1%.ipynb}.rst"' _ {} \;)

.SUFFIXES: .rst .ipynb .py

.PHONY: help clean html dirhtml singlehtml htmlhelp epub latex latexpdf text changes linkcheck doctest coverage gettext exe

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  pdf        to make LaTeX files and run them through pdflatex"
	@echo "  html       to make standalone HTML files"
	@echo "  exe        to run jupyter notebooks except those in deep_learning that requires GPU."
	@echo "  clean      rm BUILDDIR, auto_gallery, rst files"
	@echo "  cleanall   rm BUILDDIR, auto_gallery, rst files and clear output of notebooks"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  epub       to make an epub"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  text       to make text files"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation (if enabled)"
	@echo "  coverage   to run coverage check of the documentation (if enabled)"

# Rule to convert notebook to rst
#.ipynb.rst:
%.rst : %.ipynb
	jupyter nbconvert --to rst $<
	mv $@ $@.filtered
	cat $@.filtered|bin/filter_fix_rst.py > $@
	rm -f $@.filtered

#	jupyter nbconvert --to rst --stdout $< | bin/filter_fix_rst.py > $@
#	jupyter nbconvert --to rst $< --output $@

debug:
	@echo $(RST)


rst: $(RST)

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf auto_gallery/
	rm -f $(RST)
	rm -rf $(NTBOOK_FILES)

cleanall:
	rm -rf $(BUILDDIR)/*
	rm -rf auto_gallery/
	rm -f $(RST)
	rm -rf $(NTBOOK_FILES)
	for nb in $(NTBOOK) ; do jupyter nbconvert --clear-output $$nb; done

exe:
	@echo "Execute notebooks" 
	for nb in $(NTBOOK_TO_EXE) ; do jupyter nbconvert --to notebook --execute $$nb --output $$(basename $$nb); done
#	$(EXEIPYNB) $(NTBOOK)
#	@echo toto nbconvert --to notebook --execute $< --output $(basename $<)
# jupyter nbconvert --to notebook --execute scientific_python/scipy_matplotlib.ipynb

html: rst
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml: rst
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

singlehtml: rst
	$(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(BUILDDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML page is in $(BUILDDIR)/singlehtml."

docx: rst
	$(SPHINXBUILD) -b docx $(ALLSPHINXOPTS) $(BUILDDIR)/docx
	@echo
	@echo "Build finished. The docx page is in $(BUILDDIR)/docx."

epub: rst
	$(SPHINXBUILD) -b epub $(ALLSPHINXOPTS) $(BUILDDIR)/epub
	@echo
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

latex: rst
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

latexpdf: rst
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."
	cp build/latex/StatisticsMachineLearningPython.pdf StatisticsMachineLearningPython.pdf

pdf: latexpdf

text: rst
	$(SPHINXBUILD) -b text $(ALLSPHINXOPTS) $(BUILDDIR)/text
	@echo
	@echo "Build finished. The text files are in $(BUILDDIR)/text."

changes: rst
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck: rst
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

