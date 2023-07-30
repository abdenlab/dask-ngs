from __future__ import annotations

from io import BytesIO
from pathlib import Path

import dask
import dask.dataframe as dd
import oxbow as ox
import pandas as pd
import pyarrow.ipc

__version__ = "0.1.0"

__all__ = ("__version__", "read_bam", "read_vcf", "read_bcf")


def _read_bam_vpos_from_path(
    path: str, vpos_lo: tuple[int, int], vpos_hi: tuple[int, int]
) -> pd.DataFrame:
    stream = BytesIO(ox.read_bam_vpos(path, vpos_lo, vpos_hi))
    ipc = pyarrow.ipc.open_file(stream)
    return ipc.read_pandas()


def _read_vcf_vpos_from_path(
    path: str, vpos_lo: tuple[int, int], vpos_hi: tuple[int, int]
) -> pd.DataFrame:
    stream = BytesIO(ox.read_vcf_vpos(path, vpos_lo, vpos_hi))
    ipc = pyarrow.ipc.open_file(stream)
    return ipc.read_pandas()


def _read_bcf_vpos_from_path(
    path: str, vpos_lo: tuple[int, int], vpos_hi: tuple[int, int]
) -> pd.DataFrame:
    stream = BytesIO(ox.read_bcf_vpos(path, vpos_lo, vpos_hi))
    ipc = pyarrow.ipc.open_file(stream)
    return ipc.read_pandas()


def read_bam(
    path: str | Path, chunksize: int = 10_000_000, index: str | Path | None = None
) -> dd.DataFrame:
    """
    Map an indexed BAM file to a Dask DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the BAM file.
    chunksize : int, optional [default=10_000_000]
        Approximate partition size, in compressed bytes.
    index : str or Path, optional
        Path to the index file. If not provided, the index file is assumed to
        be at the same location as the BAM file, with the same name but with
        the additional .bai or .csi extension.

    Returns
    -------
    dask.dataframe.DataFrame
    """
    path = Path(path)
    if index is None:
        bai_index = path.with_suffix(path.suffix + ".bai")
        csi_index = path.with_suffix(path.suffix + ".csi")
        if bai_index.exists():
            index = str(bai_index)
        elif csi_index.exists():
            index = str(csi_index)
        else:
            msg = "Index .bai or .csi file not found."
            raise FileNotFoundError(msg)

    vpos = ox.partition_from_index_file(index, chunksize)
    chunks = [
        dask.delayed(_read_bam_vpos_from_path)(
            str(path), tuple(vpos[i]), tuple(vpos[i + 1])
        )
        for i in range(len(vpos) - 1)
    ]

    return dd.from_delayed(chunks)


def read_vcf(
    path: str | Path, chunksize: int = 10_000_000, index: str | Path | None = None
) -> dd.DataFrame:
    """
    Map an indexed, bgzf-compressed VCF.gz file to a Dask DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the VCF.gz file.
    chunksize : int, optional [default=10_000_000]
        Approximate partition size, in compressed bytes.
    index : str or Path, optional
        Path to the index file. If not provided, the index file is assumed to
        be at the same location as the VCF.gz file, with the same name but with
        the additional .tbi or .csi extension.

    Returns
    -------
    dask.dataframe.DataFrame
    """
    path = Path(path)
    if index is None:
        tbi_index = path.with_suffix(path.suffix + ".tbi")
        csi_index = path.with_suffix(path.suffix + ".csi")
        if tbi_index.exists():
            index = str(tbi_index)
        elif csi_index.exists():
            index = str(csi_index)
        else:
            msg = "Index .tbi or .csi file not found."
            raise FileNotFoundError(msg)

    vpos = ox.partition_from_index_file(index, chunksize)
    chunks = [
        dask.delayed(_read_vcf_vpos_from_path)(
            str(path), tuple(vpos[i]), tuple(vpos[i + 1])
        )
        for i in range(len(vpos) - 1)
    ]

    return dd.from_delayed(chunks)


def read_bcf(
    path: str | Path, chunksize: int = 10_000_000, index: str | Path | None = None
) -> dd.DataFrame:
    """
    Map an indexed BCF file to a Dask DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the BCF file.
    chunksize : int, optional [default=10_000_000]
        Approximate partition size, in compressed bytes.
    index : str or Path, optional
        Path to the index file. If not provided, the index file is assumed to
        be at the same location as the BCF file, with the same name but with
        the additional .csi extension.

    Returns
    -------
    dask.dataframe.DataFrame
    """
    path = Path(path)
    if index is None:
        csi_index = path.with_suffix(path.suffix + ".csi")
        if csi_index.exists():
            index = str(csi_index)
        else:
            msg = "Index .csi file not found."
            raise FileNotFoundError(msg)

    vpos = ox.partition_from_index_file(index, chunksize)
    chunks = [
        dask.delayed(_read_bcf_vpos_from_path)(
            str(path), tuple(vpos[i]), tuple(vpos[i + 1])
        )
        for i in range(len(vpos) - 1)
    ]

    return dd.from_delayed(chunks)
