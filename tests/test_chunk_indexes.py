from __future__ import annotations

from pathlib import Path

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


@pytest.mark.parametrize(
    ("offsets", "chunksize"),
    [("offsets", 500_000), ("offsets", 1_000_000)],
    indirect=["offsets"],
)
def test_map_offsets_to_chunks(offsets, chunksize):
    offsets_uniq = _index.map_offsets_to_chunks(offsets, chunksize)

    cumsum = 0
    for i in range(len(offsets_uniq) - 1):
        prev = offsets_uniq.iloc[i]
        next = offsets_uniq.iloc[i + 1]
        # validate differences
        assert next["ioffset.cpos"] - prev["ioffset.cpos"] == next["ioffset.cpos.diff"]
        cumsum += next["ioffset.cpos.diff"]
        if cumsum > chunksize:
            cumsum = 0
        # validate chunk sizes
        assert next["size"] == cumsum
        assert next["size"] <= chunksize


@pytest.mark.parametrize(
    ("offsets", "chunksize"),
    [("offsets", 500_000), ("offsets", 1_000_000)],
    indirect=["offsets"],
)
def test_consolidate_chunks(offsets, chunksize):
    offsets_uniq = _index.map_offsets_to_chunks(offsets, chunksize)
    offset_groups = _index.consolidate_chunks(offsets_uniq)

    last_cpos = offsets_uniq.groupby("chunk_id").agg(
        {
            "ioffset.cpos": "last",
        }
    )

    # validate that the final edge of the chunk `last_cpos`
    # matches the start of the chunk (`ioffset.cpos`) + its `size`
    for i in range(len(offset_groups)):
        g = offset_groups.iloc[i]
        assert g["ioffset.cpos"] + g["size"] == last_cpos.iloc[i]["ioffset.cpos"]
