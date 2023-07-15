from __future__ import annotations
from pathlib import Path
import dask_ngs
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def example_bam():
    return dask_ngs.read_bam(str((Path(__file__).parent / "fixtures" / "example.bam")))


def test_read_bam_pnext_value(example_bam):
    # Verifies that the pnext value was read correctly from the BAM file
    few_pnext_values = example_bam['pnext'].head()
    assert few_pnext_values[2] == 82982

def test_read_correct_data_types_from_bam(example_bam):
    # Compares data types of read values against expected
    assert example_bam.dtypes["qname"] == np.dtype("O")
    assert example_bam.dtypes["flag"] == np.dtype("uint16")
    assert isinstance(example_bam.dtypes["rname"], pd.core.dtypes.dtypes.CategoricalDtype)
    assert example_bam.dtypes["pos"] == np.dtype("int32")
    assert example_bam.dtypes["mapq"] == np.dtype("uint8")
    assert example_bam.dtypes["cigar"] == np.dtype("O")
    assert isinstance(example_bam.dtypes["rnext"], pd.core.dtypes.dtypes.CategoricalDtype)
    assert example_bam.dtypes["pnext"] == np.dtype("int32")
    assert example_bam.dtypes["tlen"] == np.dtype("int32")
    assert example_bam.dtypes["seq"] == np.dtype("O")
    assert example_bam.dtypes["qual"] == np.dtype("O")
    assert example_bam.dtypes["end"] == np.dtype("int32")
