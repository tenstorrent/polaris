# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tools.run_rtl_neosim_correlation import Utils, TestStatusUtils, Run, RunConfig

def test_align_text_basic():
    assert Utils.align_text("a", ("left", 3)) == "a  "
    assert Utils.align_text("a", ("right", 3)) == "  a"
    assert Utils.align_text("abc", None) == "abc"


def test_sort_table_by_columns_order_with_none_and_desc():
    tbl = [
        [1, 3],
        [1, None],
        [2, 1],
        [2, None],
    ]

    asc = Utils.sort_table_by_columns_order([row[:] for row in tbl], [0, 1], [False, False])
    assert asc == [[1, 3], [1, None], [2, 1], [2, None]]

    desc = Utils.sort_table_by_columns_order([row[:] for row in tbl], [0, 1], [True, True])
    assert desc == [[2, None], [2, 1], [1, None], [1, 3]]


def test_failure_bins_index_roundtrip():
    bins = TestStatusUtils.get_failure_bins_as_str()
    for i, b in enumerate(bins):
        assert TestStatusUtils.get_failure_bin_index(b) == i
    assert TestStatusUtils.get_failure_bin_index("not present") == len(bins)


def test_runconfig_append_run_ensures_unique_names():
    rc = RunConfig()
    rc.append_run(Run(name="dup"))
    rc.append_run(Run(name="dup"))
    rc.append_run(Run(name="dup"))
    names = [r.name for r in rc.runs]
    assert names[0] == "dup"
    assert names[1] == "dup_1"
    assert names[2] == "dup_2"


def test_utils_file_and_dir_helpers(tmp_path: Path):
    # Directory structure
    (tmp_path / "prefix_foo").mkdir()
    (tmp_path / "bar_suffix").mkdir()
    (tmp_path / "exact_name").mkdir()
    (tmp_path / "prefix_foo" / "sub").mkdir()

    # Files with same name in different places
    fname = "target.txt"
    (tmp_path / "prefix_foo" / fname).write_text("a")
    (tmp_path / "bar_suffix" / fname).write_text("b")

    files = Utils.get_files_in_dir_with_name(fname, str(tmp_path))
    assert len(files) == 2
    assert all(p.endswith(f"/{fname}") for p in files)
    assert Utils.has_only_one_copy_of_file_in_dir(fname, str(tmp_path)) is False

    # Single copy case
    only_dir = tmp_path / "only_dir"
    only_dir.mkdir()
    (only_dir / fname).write_text("x")
    assert Utils.has_only_one_copy_of_file_in_dir(fname, str(only_dir)) is True

    # Prefix, suffix, and exact name directory queries
    pref = Utils.get_dirs_with_prefix("prefix_", str(tmp_path))
    suf = Utils.get_dirs_with_suffix("_suffix", str(tmp_path))
    exact = Utils.get_dirs_with_name("exact_name", str(tmp_path))
    assert any(Path(p).name.startswith("prefix_") for p in pref)
    assert any(Path(p).name.endswith("_suffix") for p in suf)
    assert any(Path(p).name == "exact_name" for p in exact)

