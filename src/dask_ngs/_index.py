import pathlib
import io
import os

import pysam
import polars as pl
import pyarrow as pa
import pandas as pd
import bioframe

import dask_ngs
from io import BytesIO

import bioframe
import dask
import dask.dataframe as dd
import oxbow as ox
import pandas as pd
import pyarrow.ipc
from gzip import GzipFile
from io import BytesIO

import numpy as np
import pandas as pd


# Reads BAM data from Oxbox using a query:
# Path, chromosome index, start bytes, end bytes
def _read_bam_query_from_path(
    path: str, chrom: str, start: int, end: int
) -> pd.DataFrame:
    stream = BytesIO(ox.read_bam(path, f"{chrom}:{start}-{end}"))
    ipc = pyarrow.ipc.open_file(stream)
    return ipc.read_pandas()

BAI_MIN_SHIFT = 14
BAI_DEPTH = 5
COMPRESSED_POSITION_SHIFT = 16
UNCOMPRESSED_POSITION_MASK = 0xffff
BLOCKSIZE = 65536

def read_bai(path):
    """
    https://samtools.github.io/hts-specs/SAMv1.pdf
    """
    int_kwargs = {'byteorder': 'little', 'signed': False}
    f = open(path, "rb")
    # read the 4-byte magic number
    magic = f.read(4)

    # read the number of reference sequences
    n_ref = int.from_bytes(f.read(4), **int_kwargs)

    # read the reference sequence indices
    references = []
    for i in range(n_ref):
        ref = {'ref_id': i}

        # The "Bin Index"
        chunks = []
        n_bin = int.from_bytes(f.read(4), **int_kwargs)
        for _ in range(n_bin):
            # bin number
            bin_id = int.from_bytes(f.read(4), **int_kwargs)

            if bin_id == 37450:
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

                chunks.append((bin_id, chunk_beg_cpos, chunk_beg_upos, chunk_end_cpos, chunk_end_upos))   

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
        except:
            n_no_coor = None

        for ref in references:
            if not 'bins' in ref:
                continue

            ref["bins"] = pd.DataFrame(
                ref["bins"],
                columns=["bin_id", "chunk_beg.cpos", "chunk_beg.upos", "chunk_end.cpos", "chunk_end.upos"]
            )
            ref["ioffsets"] = pd.DataFrame(
                ref["ioffsets"],
                columns=["ioffset.cpos", "ioffset.upos"]
            )

    return references, n_no_coor  

# Loops through a given array of integers, cumulatively summing the values.
# The rows are labeled with a `chunk_id`, starting at 0.
# When the cumulative sum exceeds the threshold, the chunk_id is incremented,
# and the next rows are binned into the next chunk until again the threshold
# is reached. The cumulative sum of that chunk is also recorded as `size`.
# Returns a tuple of the cumulative sum array and the chunk_id array.
def cumsum_label_chunks(arr, thresh: int):
    
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


if __name__ == '__main__':

    os.chdir('./src/dask_ngs')

    # read from the index file to get the data structure
    bai, n_no_coor = read_bai('example.bam.bai')
    # select the data that defines the byte range offsets in the file
    offsets = bai[0]["ioffsets"]
    # calculate the difference in byte positions from the prior row
    # i.e. current row - previous
    offsets["ioffset.cpos.diff"] = offsets['ioffset.cpos'].diff().fillna(0).astype(int)

    offsets_uniq = offsets.groupby("ioffset.cpos").agg({
        "ioffset.upos": "first",
        "ioffset.cpos.diff": "first"
    }).reset_index()

    # note that the chunksize in this example is 1MB, not 100MB as is recommended for dask
    cumsums, chunk_ids = cumsum_label_chunks(offsets_uniq["ioffset.cpos.diff"].to_numpy(), 1000000)
    offsets_uniq["chunk_id"] = chunk_ids
    offsets_uniq["size"] = cumsums

    # Group the data by chunk_id, 
    # keeping the first compressed byte value (`ioffset.cpos`)
    # and the first uncompressed byte value of that stream (`ioffset.upos`).
    # Take the last size value which tells you how many compressed bytes to read.
    # Now you can decompress the data starting from `ioffset.cpos` and read `size` bytes.
    # `ioffsets.upos` tells you which byte to read first from the uncompressed data.
    offsets_uniq.groupby("chunk_id").agg({
        "ioffset.cpos": "first",
        "ioffset.upos": "first",
        "size": "last"
    })

    offsets_uniq