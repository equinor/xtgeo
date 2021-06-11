#!/bin/sh
set -e
# Run all examples; this is just "once in  while" as many examples are included
# in the documentation. Will work on Linux and unix.

export SKIP_PLOT=1

for pyfile in examples/*.py; do
    echo "======================================================"
    echo "Running $pyfile"
    echo "======================================================"
    python -W ignore::DeprecationWarning $pyfile
done

echo "Done"
