"""CLI entry point for ``logosdb-vibe``.

Usage::

    logosdb-vibe index ./src --namespace code
    logosdb-vibe search "retry logic" --namespace code --top-k 5
    logosdb-vibe forget --namespace code --query "old feature"
    logosdb-vibe forget --namespace code --id 42
    logosdb-vibe info [--namespace code]
    logosdb-vibe list
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _mem() -> "VibeMemory":  # noqa: F821 — imported lazily to keep startup fast
    from .vibe import VibeMemory

    return VibeMemory(
        uri=os.getenv("LOGOSDB_PATH", "./.logosdb"),
        api_key=os.getenv("MISTRAL_API_KEY"),
    )


def _print(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="logosdb-vibe",
        description="LogosDB memory layer for Mistral Vibe",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # index
    p_idx = sub.add_parser("index", help="Index a file or directory")
    p_idx.add_argument("path", help="File or directory to index")
    p_idx.add_argument("--namespace", "-n", default="code")
    p_idx.add_argument("--chunk-size", type=int, default=800)

    # search
    p_srch = sub.add_parser("search", help="Semantic search")
    p_srch.add_argument("query")
    p_srch.add_argument("--namespace", "-n", default="code")
    p_srch.add_argument("--top-k", "-k", type=int, default=5)

    # forget
    p_fgt = sub.add_parser("forget", help="Delete entries by ID or query")
    p_fgt.add_argument("--namespace", "-n", default="code")
    grp = p_fgt.add_mutually_exclusive_group(required=True)
    grp.add_argument("--id", type=int, dest="memory_id", metavar="ID")
    grp.add_argument("--query", "-q")
    p_fgt.add_argument("--top-k", "-k", type=int, default=1)

    # info
    p_inf = sub.add_parser("info", help="Database statistics")
    p_inf.add_argument("--namespace", "-n", default=None)

    # list
    sub.add_parser("list", help="List all namespaces")

    args = parser.parse_args(argv)

    try:
        mem = _mem()

        if args.cmd == "index":
            _print(mem.index(args.path, namespace=args.namespace, chunk_size=args.chunk_size))

        elif args.cmd == "search":
            results = mem.search(args.query, namespace=args.namespace, top_k=args.top_k)
            _print(results)

        elif args.cmd == "forget":
            _print(
                mem.forget(
                    namespace=args.namespace,
                    memory_id=args.memory_id,
                    query=args.query,
                    top_k=args.top_k,
                )
            )

        elif args.cmd == "info":
            _print(mem.info(namespace=args.namespace))

        elif args.cmd == "list":
            _print(mem.list_namespaces())

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
