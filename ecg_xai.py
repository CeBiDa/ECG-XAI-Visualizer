import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ecgproc.deps import check_dependencies

check_dependencies()

from ecgxai.cli import main

if __name__ == "__main__":
    sys.exit(main())
