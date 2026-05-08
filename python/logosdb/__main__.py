"""Allow running logosdb modules directly."""

import sys

if len(sys.argv) >= 2 and sys.argv[1] == 'sizing':
    # Remove the subcommand so the sizing module gets standard args
    sys.argv.pop(1)
    from logosdb.sizing import main
    sys.exit(main())
else:
    print("Usage: python -m logosdb.sizing [options]")
    print("       python -m logosdb.sizing --help")
    sys.exit(1)
