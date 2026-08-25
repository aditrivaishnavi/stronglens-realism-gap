from __future__ import annotations

import pandas as pd
import pytest

from dhs.data_splits import SplitOverlapError, check_frames


def test_development_rows_are_checked_against_test_ids(tmp_path) -> None:
    test_path = tmp_path / "test.parquet"
    pd.DataFrame({"galaxy_id": ["test-object"]}).to_parquet(test_path)

    clear = pd.DataFrame({"galaxy_id": ["development-object"]})
    result = check_frames(
        {"training": clear},
        purpose="model fitting",
        test_manifest=test_path,
    )
    assert result["test_rows_selected"] == 0

    overlap = pd.DataFrame({"galaxy_id": ["test-object"]})
    with pytest.raises(SplitOverlapError, match="1 selected rows"):
        check_frames(
            {"training": overlap},
            purpose="model fitting",
            test_manifest=test_path,
        )
