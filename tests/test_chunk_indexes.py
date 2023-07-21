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

def test_chunk(offsets):
    offsets_uniq = _index.chunk_offsets(offsets, 1_000_000)
    assert True