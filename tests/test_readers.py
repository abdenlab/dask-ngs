from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pandas.api.types import is_categorical_dtype, is_string_dtype

import dask_ngs


@pytest.fixture()
def example_bam():
    return dask_ngs.read_bam(str(Path(__file__).parent / "fixtures" / "example.bam"))


def test_read_bam_pnext_value(example_bam):
    # Verifies that the pnext value was read correctly from the BAM file
    few_pnext_values = example_bam["pnext"].head()
    pnext_val = 82982
    assert few_pnext_values[2] == pnext_val


def test_read_correct_data_types_from_bam(example_bam):
    # Compares data types of read values against expected
    assert is_string_dtype(example_bam.dtypes["qname"])
    assert example_bam.dtypes["flag"] == np.dtype("uint16")
    assert is_categorical_dtype(example_bam.dtypes["rname"])
    assert example_bam.dtypes["pos"] == np.dtype("int32")
    assert example_bam.dtypes["mapq"] == np.dtype("uint8")
    assert is_string_dtype(example_bam.dtypes["cigar"])
    assert is_categorical_dtype(example_bam.dtypes["rnext"])
    assert example_bam.dtypes["pnext"] == np.dtype("int32")
    assert example_bam.dtypes["tlen"] == np.dtype("int32")
    assert is_string_dtype(example_bam.dtypes["seq"])
    assert is_string_dtype(example_bam.dtypes["qual"])
    assert example_bam.dtypes["end"] == np.dtype("int32")
