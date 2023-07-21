from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dask_ngs import _index


@pytest.fixture()
def example_bai():
    return _index.read_bai(str(Path(__file__).parent / "fixtures" / "example.bam.bai"))

@pytest.fixture()
def offsets(example_bai):
    bai, n_no_coor = example_bai
    # TODO: Not sure if multiple BAI indexes should be tested
    return bai[0]["ioffsets"]

def test_chunk_offsets(offsets):
    chunksize = 1_000_000
    offsets_uniq = _index.chunk_offsets(offsets, chunksize)
    
    cumsum = 0
    for i in range(len(offsets_uniq)-1):
        prev = offsets_uniq.iloc[i]
        next = offsets_uniq.iloc[i+1]
        # validate differences
        assert next['ioffset.cpos'] - prev['ioffset.cpos'] == next['ioffset.cpos.diff']
        cumsum += next['ioffset.cpos.diff']
        if cumsum > chunksize:
            cumsum = 0
        # validate chunk sizes
        assert next['size'] == cumsum
        assert next['size'] <= chunksize
    return offsets_uniq

#def test_chunk_groups(test_chunk_offsets):
#    offset_groups = _index.group_chunks(test_chunk_offsets)