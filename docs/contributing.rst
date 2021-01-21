.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/equinor/xtgeo/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the Git issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the Git issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Yes, xtgeo could always use more documentation, whether as part of the
official xtgeo docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue
at https://github.com/equinor/xtgeo/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.

Get Started!
------------

Ready to contribute? Here's how to set up ``xtgeo`` for local development.

1. Fork the ``xtgeo`` repo on Github equinor to your personal user
2. Clone your fork locally:

.. code-block:: bash

    $ git clone git@github.com:your_name_here/xtgeo
    $ cd xtgeo
    $ git remote add upstream git@github.com:equinor/xtgeo
    $ git remote -v
    origin	git@github.com:your_name_here/xtgeo (fetch)
    origin	git@github.com:your_name_here/xtgeo (push)
    upstream	git@github.com:equinor/xtgeo (fetch)
    upstream	git@github.com:equinor/xtgeo (push)


3. Install your local copy into a virtualenv. Using python 3, this is how you set
up your fork for local development (first time):

.. code-block:: bash

    $ mkdir ~/venv/xtgeo; cd ~/venv/xtgeo
    $ python -m venv .
    $ source bin/activate
    $ cd /your_path_to_git_clone/xtgeo/
    $ pip install -r requirements_dev.txt
    $ python setup.py clean
    $ python setup.py develop

4. Create a branch for local development:

.. code-block:: bash

    $ git checkout -b name-of-your-bugfix-or-feature

Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the tests:

.. code-block:: bash

    $ flake8 <your edited code>
    $ pylint <your edited code>
    $ Use `pytest` for running tests
    $ Run `black` on your python code, then there is no discussions on formatting

To get `flake8`, `pylint` and `black` and just pip install them into your virtualenv.

6. If you want to edit C code, take contact with the author for detailed instructions.


7. Commit your changes (see below) and push your branch to GitHub:

.. code-block:: bash

    $ git add .
    $ git commit -m "AAA: Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the Git website.

Writing commit messages
-----------------------
The following takes effect from year 2021.

Commit messages should be clear and follow a few basic rules. Example:

.. code-block:: text

    ENH: add functionality X to numpy.<submodule>.

The first line of the commit message starts with a capitalized acronym
(options listed below) indicating what type of commit this is.  Then a blank
line, then more text if needed.  Lines shouldn't be longer than 72
characters.  If the commit is related to a ticket, indicate that with
``"See #3456", "Cf. #3344, "See ticket 3456", "Closes #3456"`` or similar.

Read `Chris Beams hints on commit messages <https://chris.beams.io/posts/git-commit/>`_.

Describing the motivation for a change, the nature of a bug for bug fixes or
some details on what an enhancement does are also good to include in a commit message.
Messages should be understandable without looking at the code changes.
A commit message like FIX: fix another one is an example of what not to do;
the reader has to go look for context elsewhere.

Standard acronyms to start the commit message with are:

.. code-block:: text

    API: an (incompatible) API change (will be rare)
    PERF: performance or bench-marking
    BLD: change related to building xtgeo
    BUG: bug fix
    FIX: fixes wrt to technical issues, e.g. wrong requirements.txt
    DEP: deprecate something, or remove a deprecated object
    DOC: documentation, addition, updates
    ENH: enhancement, new functionality
    CLN: code cleanup, maintenance commit (refactoring, typos, PEP, etc.)
    REV: revert an earlier commit
    TST: addition or modification of tests
    REL: related to releasing xtgeo

Type hints
----------

xtgeo strongly encourages (from year 2021) the use of PEP 484 style type hints.
New development should contain type hints and pull requests to annotate existing
code are accepted as well!

Style guidelines
~~~~~~~~~~~~~~~~

Types imports should follow the from typing import ... convention. So rather than

.. code-block:: python

    import typing

    primes: typing.List[int] = []

You should write

.. code-block:: python

    from typing import List, Optional, Union

    primes: List[int] = []

Optional should be used where applicable, so instead of

.. code-block:: python

    maybe_primes: List[Union[int, None]] = []

You should write

.. code-block:: python

    maybe_primes: List[Optional[int]] = []


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in HISTORY.md.


Tips
----

To run a subset of tests, e.g. only surface tests:

.. code:: bash

    $ pytest test/test_surfaces
