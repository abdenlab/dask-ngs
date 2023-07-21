from __future__ import annotations

from io import BytesIO

import bioframe
import dask
import dask.dataframe as dd
import oxbow as ox
import pandas as pd
import pyarrow.ipc

__version__ = "0.1.0"

__all__ = ("__version__", "read_bam")


def _read_bam_query_from_path(
    path: str, chrom: str, start: int, end: int
) -> pd.DataFrame:
    stream = BytesIO(ox.read_bam(path, f"{chrom}:{start}-{end}"))
    ipc = pyarrow.ipc.open_file(stream)
    return ipc.read_pandas()


def read_bam(path: str, chunksize: int = 10_000_000) -> dd.DataFrame:
    """
    Map an indexed BAM file to a Dask DataFrame.

    Parameters
    ----------
    path : str
        Path to the BAM file.
    chunksize : int, optional [default=10_000_000]
        Chunk size, currently in base pair coordinates.

    Returns
    -------
    dask.dataframe.DataFrame
        A Dask DataFrame with the BAM file contents.
    """
    chromsizes = bioframe.fetch_chromsizes("hg38")
    chunk_spans = bioframe.binnify(chromsizes, chunksize)
    chunks = [
        dask.delayed(_read_bam_query_from_path)(path, chrom, start + 1, end)
        for chrom, start, end in chunk_spans.to_numpy()
    ]
    return dd.from_delayed(chunks)
