"""Allow ``python -m qml_compare`` to reach the CLI."""

import sys

from qml_compare.cli import main

if __name__ == "__main__":
    sys.exit(main())
