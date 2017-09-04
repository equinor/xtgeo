xt.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys
for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

APPLICATION := xtgeo

# JRIV override browser
BROWSER := firefox

PYTHON := python${PYTHON_SHORT}
PIP := pip${PYTHON_SHORT}

PYVER := $(shell python -c "import sys; print('{0[0]}.{0[1]}'.format(sys.version_info))")

RESPYPATH := ${SDP_BINDIST_ROOT}
RESPYPATHFULL := ${RESPYPATH}/lib/python${PYVER}/site-packages

TARGET := ${RESPYPATHFULL}

USRPYPATH := ${MY_BINDIST}/lib/python${PYVER}/site-packages

DOCINSTALL := /project/sdpdocs/Users/jriv/libs

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts


clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +


clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr TMP/

lint: ## check style with flake8
	flake8 ${APPLICATION} tests

test:  ## run tests quickly with the default Python
	${PYTHON} setup.py test

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source ${APPLICATION} -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docsrun: clean ## generate Sphinx HTML documentation, including API docs
	rm -f docs/xtgeo.rst
	rm -f docs/modules.rst
	rm -fr docs/_build
	sphinx-apidoc -o docs/ xtgeo
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

docs: docsrun ## generate and display Sphinx HTML documentation...
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

dist: clean ## builds source and wheel package
	# python setup.py sdist
	${PYTHON} setup.py bdist_wheel


oldinstall: clean ## install the package to the active Python's site-packages
	${PYTHON} setup.py install


install: dist ## version to VENV install place
#	${PIP} uninstall --yes ${APPLICATION}
	${PIP} install --upgrade ./dist/*


siteinstall: dist ## Install in project/res (Trondheim)
	echo $(HOST)
	\rm -fr  ${TARGET}/${APPLICATION}*
	${PIP} install --target ${TARGET} --upgrade  ./dist/${APPLICATION}*.whl
	/project/res/bin/res_perm ${TARGET}/${APPLICATION}*


userinstall: dist ## Install on user directory (need a MY_BINDIST env variable)
	\rm -fr  ${USRPYPATH}/${APPLICATION}*
	${PIP} install --target ${USRPYPATH} --upgrade  ./dist/*.whl
#	rsync -v -L --chmod=a+rx bin/* ${MY_BINDIST}/bin/.

docinstall: docsrun
	rsync -av --delete docs/_build/html ${DOCINSTALL}/${APPLICATION}
	/project/res/bin/res_perm ${DOCINSTALL}/${APPLICATION}
