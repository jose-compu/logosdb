"""LogosDB sizing calculator for disk and memory estimates.

Usage:
    python -m logosdb.sizing --rows 1_000_000 --dim 768
    python -m logosdb.sizing --rows 10_000_000 --dim 1024 --m 32 --text-bytes 500

Programmatic usage:
    from logosdb.sizing import estimate_size
    est = estimate_size(n_rows=1_000_000, dim=768)
    print(est.total_disk_gb)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Literal


# Storage constants
HEADER_BYTES = 32  # Vector file header size
BYTES_PER_DTYPE = {
    'float32': 4,
    'float16': 2,
    'int8': 1,
}

# HNSW index constants (from hnswlib documentation)
# Index size ≈ N × M × 9 bytes (average of 8-10 bytes per link)
HNSW_BYTES_PER_LINK = 9  # Average of 8-10 bytes per element per M

# Default values
DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 50
DEFAULT_AVG_TEXT_BYTES = 200  # Average metadata text size per document


@dataclass(frozen=True)
class SizeEstimate:
    """Sizing estimate for a LogosDB deployment."""

    # Input parameters
    n_rows: int
    dim: int
    m: int
    dtype: str
    avg_text_bytes: int

    # Storage sizes (bytes)
    vector_file_bytes: int
    hnsw_index_bytes: int
    metadata_bytes: int

    # Memory sizes (bytes)
    index_ram_bytes: int

    @property
    def total_disk_bytes(self) -> int:
        """Total disk space required."""
        return self.vector_file_bytes + self.hnsw_index_bytes + self.metadata_bytes

    @property
    def vector_file_gb(self) -> float:
        return self.vector_file_bytes / (1024 ** 3)

    @property
    def hnsw_index_gb(self) -> float:
        return self.hnsw_index_bytes / (1024 ** 3)

    @property
    def metadata_gb(self) -> float:
        return self.metadata_bytes / (1024 ** 3)

    @property
    def total_disk_gb(self) -> float:
        return self.total_disk_bytes / (1024 ** 3)

    @property
    def index_ram_gb(self) -> float:
        return self.index_ram_bytes / (1024 ** 3)

    @property
    def query_ram_estimate_mb(self) -> float:
        """Estimate query working set RAM (conservative)."""
        # Typical query touches ~500 vectors + overhead
        vectors_touched = min(500, self.n_rows)
        working_set_bytes = vectors_touched * self.dim * BYTES_PER_DTYPE[self.dtype]
        # Add overhead for HNSW search structures
        overhead = 10 * 1024 * 1024  # 10 MB base overhead
        return (working_set_bytes + overhead) / (1024 ** 2)

    def format_report(self) -> str:
        """Generate a formatted text report."""
        lines = [
            "LogosDB Size Estimate",
            "=====================",
            "",
            "Input:",
            f"  Rows: {self.n_rows:,}",
            f"  Dimensions: {self.dim:,}",
            f"  HNSW M: {self.m}",
            f"  HNSW ef_construction: {DEFAULT_EF_CONSTRUCTION}",
            f"  HNSW ef_search: {DEFAULT_EF_SEARCH}",
            f"  Precision: {self.dtype}",
            f"  Avg text size: {self.avg_text_bytes} bytes/doc",
            "",
            "Storage:",
            f"  Vector file:     {self.vector_file_gb:>8.2f} GB  ({self.vector_file_bytes:,} bytes)",
            f"  HNSW index:      {self.hnsw_index_gb:>8.2f} GB  ({self.hnsw_index_bytes:,} bytes)",
            f"  Metadata (est):  {self.metadata_gb:>8.2f} GB  (~{self.avg_text_bytes} bytes/doc)",
            f"  {'─' * 28}",
            f"  Total disk:      {self.total_disk_gb:>8.2f} GB",
            "",
            "Memory (RSS):",
            f"  Index RAM (required): {self.index_ram_gb:.2f} GB",
            f"  Query RAM (typical):  <{self.query_ram_estimate_mb:.0f} MB",
            "",
            "Notes:",
            "  • Vector file uses mmap - only touched pages load into RAM",
            "  • HNSW index must reside in RAM for queries",
            "  • Metadata is read on-demand; not fully cached",
            "  • Query RAM depends on working set size and access patterns",
        ]
        return '\n'.join(lines)


def estimate_size(
    n_rows: int,
    dim: int,
    m: int = DEFAULT_M,
    dtype: Literal['float32', 'float16', 'int8'] = 'float32',
    avg_text_bytes: int = DEFAULT_AVG_TEXT_BYTES,
) -> SizeEstimate:
    """Calculate size estimates for a LogosDB deployment.

    Args:
        n_rows: Number of vectors (rows)
        dim: Vector dimensionality
        m: HNSW M parameter (graph connectivity, default 16)
        dtype: Vector precision ('float32', 'float16', 'int8')
        avg_text_bytes: Average metadata text size per document

    Returns:
        SizeEstimate with detailed breakdown

    Example:
        >>> est = estimate_size(1_000_000, 768)
        >>> print(f"Disk: {est.total_disk_gb:.2f} GB")
        Disk: 3.12 GB
    """
    if dtype not in BYTES_PER_DTYPE:
        raise ValueError(f"dtype must be one of {list(BYTES_PER_DTYPE.keys())}")

    if n_rows < 0 or dim < 0:
        raise ValueError("n_rows and dim must be non-negative")

    # Vector file: header + N × dim × bytes_per_dim
    bytes_per_dim = BYTES_PER_DTYPE[dtype]
    vector_file_bytes = HEADER_BYTES + (n_rows * dim * bytes_per_dim)

    # HNSW index: N × M × bytes_per_link
    # Per hnswlib docs: ~8-10 bytes per element per M
    hnsw_index_bytes = n_rows * m * HNSW_BYTES_PER_LINK

    # Metadata: JSONL format, variable size
    # Estimate: avg_text_bytes × 1.2 for JSON overhead + timestamp
    json_overhead = 1.2
    metadata_bytes = int(n_rows * avg_text_bytes * json_overhead)

    # Index RAM = HNSW index size (must be in memory)
    index_ram_bytes = hnsw_index_bytes

    return SizeEstimate(
        n_rows=n_rows,
        dim=dim,
        m=m,
        dtype=dtype,
        avg_text_bytes=avg_text_bytes,
        vector_file_bytes=vector_file_bytes,
        hnsw_index_bytes=hnsw_index_bytes,
        metadata_bytes=metadata_bytes,
        index_ram_bytes=index_ram_bytes,
    )


def generate_comparison_table(
    rows_list: list[int] = None,
    dims: list[int] = None,
    m: int = DEFAULT_M,
) -> str:
    """Generate a markdown sizing table."""
    if rows_list is None:
        rows_list = [100_000, 1_000_000, 10_000_000]
    if dims is None:
        dims = [384, 768, 1024, 4096]

    lines = [
        "# LogosDB Sizing Reference (float32)",
        "",
        f"HNSW M={m}, default ef parameters",
        "",
        "| Vectors | Dim | Vector File | HNSW Index | Total Disk | Index RAM |",
        "|---------|-----|-------------|------------|------------|-----------|",
    ]

    for n_rows in rows_list:
        for dim in dims:
            est = estimate_size(n_rows, dim, m=m)
            lines.append(
                f"| {n_rows:,} | {dim} | "
                f"{est.vector_file_gb:.1f} GB | "
                f"{est.hnsw_index_gb:.2f} GB | "
                f"{est.total_disk_gb:.1f} GB | "
                f"{est.index_ram_gb:.2f} GB |"
            )

    return '\n'.join(lines)


def main(args: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LogosDB size estimator for disk and memory planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m logosdb.sizing --rows 1_000_000 --dim 768
  python -m logosdb.sizing --rows 10_000_000 --dim 1024 --m 32
  python -m logosdb.sizing --rows 100_000 --dim 4096 --dtype float16
  python -m logosdb.sizing --generate-table > sizing_table.md
        """,
    )

    parser.add_argument(
        '--rows', '-n',
        type=lambda x: int(x.replace('_', '')),
        help='Number of vectors (supports underscores: 1_000_000)',
    )
    parser.add_argument(
        '--dim', '-d',
        type=int,
        help='Vector dimensionality',
    )
    parser.add_argument(
        '--m', '-m',
        type=int,
        default=DEFAULT_M,
        help=f'HNSW M parameter (default: {DEFAULT_M})',
    )
    parser.add_argument(
        '--dtype',
        choices=['float32', 'float16', 'int8'],
        default='float32',
        help='Vector precision (default: float32)',
    )
    parser.add_argument(
        '--text-bytes', '-t',
        type=int,
        default=DEFAULT_AVG_TEXT_BYTES,
        help=f'Average metadata text size per doc (default: {DEFAULT_AVG_TEXT_BYTES})',
    )
    parser.add_argument(
        '--generate-table',
        action='store_true',
        help='Generate a markdown comparison table',
    )

    parsed = parser.parse_args(args)

    if parsed.generate_table:
        print(generate_comparison_table())
        return 0

    if parsed.rows is None or parsed.dim is None:
        parser.error('--rows and --dim are required (or use --generate-table)')

    est = estimate_size(
        n_rows=parsed.rows,
        dim=parsed.dim,
        m=parsed.m,
        dtype=parsed.dtype,
        avg_text_bytes=parsed.text_bytes,
    )

    print(est.format_report())
    return 0


if __name__ == '__main__':
    sys.exit(main())
