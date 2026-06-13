"""Allow `python -m pmtdeconv ...` invocation."""
from pmtdeconv.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
