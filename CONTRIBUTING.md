# Contributing

Contributions are welcome and greatly appreciated! There are several ways to
contribute to XTGeo.

## Types of Contributions

### Report Bugs

Report bugs at
[https://github.com/equinor/xtgeo/issues](https://github.com/equinor/xtgeo/issues).

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug", "help
wanted", or "good first issue" is open to whomever wants to fix it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with
"enhancement", "help wanted", or "good first issue" is open to whomever wants
to implement it.

### Write Documentation

XTGeo could always use improved documentation, whether as part of the
official XTGeo docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
[https://github.com/equinor/xtgeo/issues](https://github.com/equinor/xtgeo/issues).

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.

## Get Started!

Ready to contribute? Here's how to set up XTGeo for local development.

1. Fork the XTGeo repository from the Equinor repository to your GitHub
   user account.

2. Clone your fork locally:

   ```sh
   git clone git@github.com:your_name_here/xtgeo
   cd xtgeo
   git remote add upstream git@github.com:equinor/xtgeo
   git remote -v
   # origin	git@github.com:your_name_here/xtgeo (fetch)
   # origin	git@github.com:your_name_here/xtgeo (push)
   # upstream	git@github.com:equinor/xtgeo (fetch)
   # upstream	git@github.com:equinor/xtgeo (push)
   ```

3. Install your forked copy into a local venv:

   ```sh
   python -m venv ~/venv/xtgeo
   source ~/venv/xtgeo/bin/activate
   pip install -U pip
   pip install -e ".[dev,docs]"  # add -vv to see compilation output
   ```

4. Install the test data one directory below your XTGeo directory and run
   the tests to ensure everything works:

   ```sh
   git clone --depth 1 https://github.com/equinor/xtgeo-testdata ../xtgeo-testdata
   pytest -n auto
   ```

5. Create a branch for local development:

   ```sh
   git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

6. When you're done making changes, check that your changes pass ruff and the
   tests:

   ```sh
   ruff check .
   ruff format .
   mypy src/
   pytest tests
   ```

7. If you want to contribute to the C/C++ code base contact the XTGeo authors
   for detailed instructions.

8. Commit your changes (see
   [Writing commit messages](#writing-commit-messages)) and push your
   branch to GitHub:

   ```sh
   git add file1.py file2.py
   git commit -m "ENH: Add some feature"
   git push origin name-of-your-bugfix-or-feature
   ```

9. Submit a pull request through GitHub.

## Building documentation

It is a good idea to check that the documentation builds and displays
properly, particularly if you have made changes to it. You can build and look
at the documentation like so:

```sh
sphinx-build -W -b html docs build/docs/html
```

Then open `build/docs/html/index.html` in your browser.

## Writing commit messages

Commit messages should be clear and follow a few basic rules. Example:

```text
ENH: add functionality X to numpy

This functionality was added due to a user feature request and relates to
changes X, Y, Z which together completed feature ABC. With this change users
can now do `vals = values.foo().bar()`.
```

The first word of the commit message starts with a capitalized acronym or
abbreviation. A list of these is available below. This prefixindicates what
type of commit this is and is followed by a brief description of the change.
This line should strive to be less than or equal to 50 characters.

More explanation and context is often helpful to developers in the future. If
this is the case you should add a blank link with a longer description giving
some of these contextual details. This commit information can be as long as is
needed but the individual lines shouldn't be longer than 72 characters.

See [Chris Beams How to Write a Git Commit Message](https://cbea.ms/git-commit/)
article for more explanation of these guidelines.

Describing the motivation for a change, the nature of a bug for bug fixes or
some details on what an enhancement does are also good to include in a commit message.
Messages should be understandable without looking at the code changes.
A commit message like `FIX: fix another one` is an example of what not to do;
the reader has to go look for context elsewhere.

Standard prefixes to start the commit message with are:

```text
API: an (incompatible) API change (will be rare)
BLD: change related to building xtgeo
BUG: bug fix
CI: related to CI/CD workflows
CLN: code cleanup, maintenance commit (refactoring, typos, PEP, etc.)
DEP: deprecate something, or remove a deprecated object
DOC: documentation, addition, updates
ENH: enhancement, new functionality
FIX: fixes wrt to technical issues, e.g. wrong requirements.txt
PERF: performance or bench-marking
REV: revert an earlier commit
REL: related to releasing xtgeo
TST: addition or modification of tests
```

### Type Hints

XTGeo requires the use of type annotations in all new feature
developments, incorporating Python 3.10's enhanced syntax for type hints.
This facilitates a more concise and readable style.

### Style Guidelines

- For Python versions prior to 3.10, include the following import for
  compatibility:

  ```python
  from __future__ import annotations
  ```

- Use Python's built-in generics (e.g., `list`, `tuple`) directly. This
  approach is preferred over importing types like `List` or `Tuple` from
  the `typing` module.

- Apply the new union type syntax using the pipe (`|`) for clarity and
  simplicity. For example:

  ```python
  primes: list[int | float] = []
  ```

- For optional types, use `None` with the pipe (`|`) instead of `Optional`.
  For instance:

  ```python
  maybe_primes: list[int | None] = []
  ```

Note: These guidelines align with PEP 604 and are preferred for all new code
submissions and when updating existing code.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.

2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring and if it is a
   public method make sure it displays helpful information in the
   documentation.

## Tips

- To run a subset of tests, e.g. only surface tests:

  ```python
  pytest test/test_surfaces
  ```

- scikit-build-core offers some suggestions about building with editable
  installs, see info
  [here](https://scikit-build-core.readthedocs.io/en/latest/configuration.html#editable-installs)


## Working with RMS python

The following is a special recipe when working with RMS' Python version,
and it is targeted to Equinor usage using bash shell in Linux:

```sh
# activate RMS python, e.g. RMS version 13.1.2
source /prog/res/roxapi/aux/roxenvbash 13.1.2
# Make a venv with the libraries included by RMS
python -m venv ~/venv/py38_rms13.1.2 --system-site-packages
source ~/venv/py38_rms13.1.2/bin/activate
unset PYTHONPATH  # to avoid potential issues
python -m pip install -U pip
pip install -e ".[dev]"
pytest
```

Now you have an editable install in your virtual environment that can be ran
in RMS while testing. Hence open rms with ``rms`` command (not ``runrms``).

Inside RMS you can open a Python dialog and run your version of XTGeo. Theoretically,
you could now do changes in your editable install and RMS should see them.
However, RMS will not load libraries updates once loaded, and ``importlib.reload``
will not help very much. One safe alternative is of course to close and
reopen RMS, but that is unpractical and time consuming.

The better alternative is a brute force hack in order to make it work,
see the five lines of code in top of this example:

```python
import sys
sysm = sys.modules.copy()
for k, _ in sysm.items():
    if "xtgeo" in k:
        del sys.modules[k]

import xtgeo

grd = xgeo.grid_from_roxar(project, "Geogrid")
```

This will work if you change python code in XTGeo. If you change C code in XTGeo, then
this hack will not work. The only solution is to close and re-open RMS everytime the
C code is compiled.
