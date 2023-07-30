from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BAI_MIN_SHIFT = 14
BAI_DEPTH = 5
COMPRESSED_POSITION_SHIFT = 16
UNCOMPRESSED_POSITION_MASK = 0xFFFF
BLOCKSIZE = 65536
BAM_PSEUDO_BIN = 37450


def read_bai(path: str):
    """
    https://samtools.github.io/hts-specs/SAMv1.pdf
    """
    int_kwargs = {"byteorder": "little", "signed": False}
    with Path(path).open("rb") as f:
        # read the 4-byte magic number
        f.read(4)

        # read the number of reference sequences
        n_ref = int.from_bytes(f.read(4), **int_kwargs)

        # read the reference sequence indices
        references = []
        for i in range(n_ref):
            ref = {"ref_id": i}

            # The "Bin Index"
            chunks = []
            n_bin = int.from_bytes(f.read(4), **int_kwargs)
            for _ in range(n_bin):
                # bin number
                bin_id = int.from_bytes(f.read(4), **int_kwargs)

                if bin_id == BAM_PSEUDO_BIN:
                    # This is an entry that describes the "pseudo-bin" for the
                    # reference, using the same byte layout as normal bins but
                    # interpreted differently.
                    # The ref beg/ref end fields locate the first and last reads on
                    # this reference sequence, whether they are mapped or placed
                    # unmapped. Thus they are equal to the minimum chunk beg and
                    # maximum chunk end respectively.
                    n_chunk = int.from_bytes(f.read(4), **int_kwargs)  # always 2
                    # Not really a chunk
                    vpos = int.from_bytes(f.read(8), **int_kwargs)
                    ref["ref_beg.cpos"] = vpos >> COMPRESSED_POSITION_SHIFT
                    ref["ref_beg.upos"] = vpos & UNCOMPRESSED_POSITION_MASK
                    vpos = int.from_bytes(f.read(8), **int_kwargs)
                    ref["ref_end.cpos"] = vpos >> COMPRESSED_POSITION_SHIFT
                    ref["ref_end.upos"] = vpos & UNCOMPRESSED_POSITION_MASK
                    # Not really a chunk
                    ref["n_mapped"] = int.from_bytes(f.read(8), **int_kwargs)
                    ref["n_unmapped"] = int.from_bytes(f.read(8), **int_kwargs)
                    continue

                n_chunk = int.from_bytes(f.read(4), **int_kwargs)
                for _ in range(n_chunk):
                    vpos = int.from_bytes(f.read(8), **int_kwargs)
                    chunk_beg_cpos = vpos >> COMPRESSED_POSITION_SHIFT
                    chunk_beg_upos = vpos & UNCOMPRESSED_POSITION_MASK
                    vpos = int.from_bytes(f.read(8), **int_kwargs)
                    chunk_end_cpos = vpos >> COMPRESSED_POSITION_SHIFT
                    chunk_end_upos = vpos & UNCOMPRESSED_POSITION_MASK

                    chunks.append(
                        (
                            bin_id,
                            chunk_beg_cpos,
                            chunk_beg_upos,
                            chunk_end_cpos,
                            chunk_end_upos,
                        )
                    )

                ref["bins"] = chunks

            # The "Linear Index"
            ref["ioffsets"] = []
            n_intv = int.from_bytes(f.read(4), **int_kwargs)
            for _ in range(n_intv):
                vpos = int.from_bytes(f.read(8), **int_kwargs)
                ioffset_cpos = vpos >> COMPRESSED_POSITION_SHIFT
                ioffset_upos = vpos & UNCOMPRESSED_POSITION_MASK
                ref["ioffsets"].append((ioffset_cpos, ioffset_upos))

            references.append(ref)

            # Check for the optional n_no_coor at the end of the file
            try:
                n_no_coor = int.from_bytes(f.read(8), **int_kwargs)
            except Exception:
                n_no_coor = None

            for ref in references:
                if "bins" not in ref:
                    continue

                ref["bins"] = pd.DataFrame(
                    ref["bins"],
                    columns=[
                        "bin_id",
                        "chunk_beg.cpos",
                        "chunk_beg.upos",
                        "chunk_end.cpos",
                        "chunk_end.upos",
                    ],
                )
                ref["ioffsets"] = pd.DataFrame(
                    ref["ioffsets"], columns=["ioffset.cpos", "ioffset.upos"]
                )

    return references, n_no_coor


def _cumsum_assign_chunks(arr: np.array, thresh: int) -> tuple[np.array, np.array]:
    """
    Loops through a given array of integers, cumulatively summing the values.
    The rows are labeled with a `chunk_id`, starting at 0.

    When the cumulative sum exceeds the threshold, the chunk_id is incremented,
    and the next rows are binned into the next chunk until again the threshold
    is reached. The cumulative sum of that chunk is also recorded as `size`.
    Returns a tuple of the cumulative sum array and the chunk_id array.

    Parameters
    ----------
    arr : numpy array
        The array of byte offsets to chunk
    thresh : int
        The size of chunks in bytes

    Returns
    -------
    array of cumulative byte sums
    array of chunk_ids assigned to each row
    """
    sum = 0
    chunkid = 0
    cumsums = np.zeros_like(arr)
    chunk_ids = np.zeros_like(arr)
    for i in range(len(arr)):
        sum += arr[i]
        if sum > thresh:
            sum = 0
            chunkid += 1
        cumsums[i] = sum
        chunk_ids[i] = chunkid
    return cumsums, chunk_ids


def map_offsets_to_chunks(offsets: pd.DataFrame, chunksize_bytes: int) -> pd.DataFrame:
    """Given a dataframe of offset positions, calculate the difference
    between each byte offset.

    Group those differences into chunks of size `chunksize_bytes`.

    Returns
    -------
    A Pandas dataframe with additional columns:
    chunk_id : int
        The chunk index that row was assigned
    size : int
        The cumulative size of that chunk
    """

    # calculate the difference in byte positions from the prior row
    # i.e. current row - previous
    offsets["ioffset.cpos.diff"] = offsets["ioffset.cpos"].diff().fillna(0).astype(int)

    # group the offsets so
    # this produces a dataframe that looks like this:
    # ioffset.cpos | ioffset.upos |	ioffset.cpos.diff
    #        38660 |            0 |	                0
    #       157643 |	    61968 |            118983
    #       456717 |	    19251 |            299074
    # this represents how far to read each compressed array
    # e.g. 38660 + 118983 = 157643
    offsets_uniq = (
        offsets.groupby("ioffset.cpos")
        .agg({"ioffset.upos": "first", "ioffset.cpos.diff": "first"})
        .reset_index()
    )

    cumsums, chunk_ids = _cumsum_assign_chunks(
        offsets_uniq["ioffset.cpos.diff"].to_numpy(), chunksize_bytes
    )
    offsets_uniq["chunk_id"] = chunk_ids
    offsets_uniq["size"] = cumsums

    return offsets_uniq


def consolidate_chunks(offsets_uniq: pd.DataFrame) -> pd.DataFrame:
    """Group the data by `chunk_id`, keeping the first compressed byte value
    (`ioffset.cpos`) and the first uncompressed byte value of that stream
    (`ioffset.upos`).

    Take the last `size` value which tells you how many compressed bytes to read.

    Returns
    -------
    A Pandas dataframe grouped by `chunk_id`

    Notes
    -----
    Now you can decompress the data starting from `ioffset.cpos` and read `size` bytes.
    `ioffsets.upos` tells you which byte to read first from the uncompressed data.
    """
    return offsets_uniq.groupby("chunk_id").agg(
        {"ioffset.cpos": "first", "ioffset.upos": "first", "size": "last"}
    )
