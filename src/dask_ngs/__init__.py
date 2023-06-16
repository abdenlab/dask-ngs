from __future__ import annotations

from io import BytesIO
import bioframe
import dask.dataframe as dd
import dask
import pandas as pd
from pyarrow.ipc import open_file as read_ipc

import oxbow as ox

__version__ = "0.1.0"

__all__ = ("__version__",)


def _read_bam_query_from_path(
    path: str, chrom: str, start: int, end: int
) -> pd.DataFrame:
    stream = BytesIO(ox.read_bam(path, f"{chrom}:{start}-{end}"))
    ipc = read_ipc(stream)
    return ipc.to_pandas()


def read_bam(path: str, chunksize: int = 10_000_000) -> dd.DataFrame:
    chromsizes = bioframe.fetch_chromsizes("hg38")
    chunk_spans = bioframe.binnify(chromsizes, chunksize)
    chunks = [
        dask.delayed(_read_bam_query_from_path)(path, chrom, start, end)
        for chrom, start, end in chunk_spans.to_numpy()
    ]
    return dd.from_delayed(chunks)
