#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for reference data validation and integrity.

This module validates that reference data in data/metal/inf/<TAG>/ is
properly structured, complete, and consistent across tags.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest
import yaml

from tools.ttsi_corr.ttsi_corr_utils import TTSI_REF_DEFAULT_TAG, TTSI_REF_VALID_TAGS

# Path to reference data directory
REFERENCE_DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'metal' / 'inf'


# Required fields in each benchmark entry
REQUIRED_FIELDS = {
    'batch', 'gpu', 'hardware', 'id', 'model', 'precision', 'input_dtype'
}

# Optional but recommended fields
RECOMMENDED_FIELDS = {'release'}

# Valid metric fields (at least one should be non-null)
METRIC_FIELDS = {
    'images_per_sec', 'fps', 'sentences_per_sec',
    'tokens_per_sec', 'tokens_per_sec_per_user', 'sec_per_image'
}


def load_tag_data(tag: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all YAML files for a given tag.

    Args:
        tag: Tag identifier (e.g., '03nov25')

    Returns:
        Dict mapping filename to list of benchmark entries
    """
    tag_dir = REFERENCE_DATA_DIR / tag
    if not tag_dir.exists():
        pytest.skip(f'Tag directory {tag} does not exist')

    data = {}
    for yaml_file in tag_dir.glob('tensix_md_perf_metrics_*.yaml'):
        with open(yaml_file, 'r') as f:
            content = yaml.safe_load(f)
            if content:  # Skip empty files
                data[yaml_file.name] = content

    return data


def load_metadata(tag: str) -> Dict[str, Any]:
    """
    Load metadata for a given tag.

    Args:
        tag: Tag identifier

    Returns:
        Metadata dictionary
    """
    metadata_file = REFERENCE_DATA_DIR / tag / '_metadata.yaml'
    if not metadata_file.exists():
        return {}

    with open(metadata_file, 'r') as f:
        return yaml.safe_load(f) or {}


class TestReferenceDataStructure:
    """Test the structure and integrity of reference data."""

    @pytest.mark.parametrize('tag', TTSI_REF_VALID_TAGS)
    def test_tag_directory_exists(self, tag: str) -> None:
        """Verify that tag directories exist for all valid tags."""
        tag_dir = REFERENCE_DATA_DIR / tag
        assert tag_dir.exists(), f'Tag directory {tag} does not exist'
        assert tag_dir.is_dir(), f'{tag} is not a directory'

    @pytest.mark.parametrize('tag', TTSI_REF_VALID_TAGS)
    def test_metadata_file_exists(self, tag: str) -> None:
        """Verify that each tag has a metadata file."""
        metadata_file = REFERENCE_DATA_DIR / tag / '_metadata.yaml'
        assert metadata_file.exists(), f'Metadata file missing for tag {tag}'

    @pytest.mark.parametrize('tag', TTSI_REF_VALID_TAGS)
    def test_metadata_has_required_fields(self, tag: str) -> None:
        """Verify metadata has required fields."""
        metadata = load_metadata(tag)
        required = {'tag', 'data_source', 'input_url', 'parsed_date'}
        missing = required - set(metadata.keys())
        assert not missing, f'Metadata for {tag} missing fields: {missing}'

    @pytest.mark.parametrize('tag', TTSI_REF_VALID_TAGS)
    def test_metadata_tag_matches_directory(self, tag: str) -> None:
        """Verify metadata tag field matches directory name."""
        metadata = load_metadata(tag)
        assert metadata.get('tag') == tag, \
            f'Metadata tag {metadata.get("tag")} does not match directory {tag}'

    @pytest.mark.parametrize('tag', TTSI_REF_VALID_TAGS)
    def test_yaml_files_are_valid(self, tag: str) -> None:
        """Verify all YAML files can be parsed."""
        tag_dir = REFERENCE_DATA_DIR / tag
        yaml_files = list(tag_dir.glob('tensix_md_perf_metrics_*.yaml'))

        assert len(yaml_files) > 0, f'No YAML files found for tag {tag}'

        for yaml_file in yaml_files:
            with open(yaml_file, 'r') as f:
                try:
                    content = yaml.safe_load(f)
                    assert content is None or isinstance(content, list), \
                        f'{yaml_file.name} should contain a list of entries'
                except yaml.YAMLError as e:
                    pytest.fail(f'Invalid YAML in {yaml_file.name}: {e}')


class TestBenchmarkEntries:
    """Test individual benchmark entries for completeness and validity."""

    @pytest.mark.parametrize('tag', TTSI_REF_VALID_TAGS)
    def test_all_entries_have_required_fields(self, tag: str) -> None:
        """Verify all benchmark entries have required fields."""
        data = load_tag_data(tag)
        errors = []

        for filename, entries in data.items():
            for idx, entry in enumerate(entries):
                missing = REQUIRED_FIELDS - set(entry.keys())
                if missing:
                    errors.append(
                        f'{filename}[{idx}] ({entry.get("model", "unknown")}): '
                        f'missing {missing}'
                    )

        assert not errors, 'Entries with missing required fields:\n' + '\n'.join(errors)

    @pytest.mark.parametrize('tag', TTSI_REF_VALID_TAGS)
    def test_all_entries_have_metrics(self, tag: str) -> None:
        """Verify all entries have at least one performance metric."""
        data = load_tag_data(tag)
        errors = []

        for filename, entries in data.items():
            for idx, entry in enumerate(entries):
                has_metric = any(
                    entry.get(field) is not None
                    for field in METRIC_FIELDS
                )
                if not has_metric:
                    errors.append(
                        f'{filename}[{idx}] ({entry.get("model", "unknown")}): '
                        f'no performance metrics'
                    )

        assert not errors, 'Entries without metrics:\n' + '\n'.join(errors)

    @pytest.mark.parametrize('tag', TTSI_REF_VALID_TAGS)
    def test_gpu_field_is_tensix(self, tag: str) -> None:
        """Verify all entries have gpu='Tensix'."""
        data = load_tag_data(tag)
        errors = []

        for filename, entries in data.items():
            for idx, entry in enumerate(entries):
                if entry.get('gpu') != 'Tensix':
                    errors.append(
                        f'{filename}[{idx}] ({entry.get("model", "unknown")}): '
                        f'gpu={entry.get("gpu")}'
                    )

        assert not errors, 'Entries with incorrect gpu field:\n' + '\n'.join(errors)

    @pytest.mark.parametrize('tag', TTSI_REF_VALID_TAGS)
    def test_hardware_field_format(self, tag: str) -> None:
        """Verify hardware field follows expected format."""
        data = load_tag_data(tag)
        errors = []

        # Expected format: "device (architecture)" or "device"
        # Allow spaces in device name (e.g., "2 x p150 (Blackhole)", "QuietBox  (Wormhole)")
        # Require at least one alphanumeric character
        pattern = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9\s]*\([^)]+\)$|^[a-zA-Z0-9][a-zA-Z0-9\s]*$')

        for filename, entries in data.items():
            for idx, entry in enumerate(entries):
                hardware = entry.get('hardware', '')
                if not pattern.match(hardware):
                    errors.append(
                        f'{filename}[{idx}] ({entry.get("model", "unknown")}): '
                        f'hardware="{hardware}" does not match expected format'
                    )

        assert not errors, 'Entries with invalid hardware format:\n' + '\n'.join(errors)

    @pytest.mark.parametrize('tag', TTSI_REF_VALID_TAGS)
    def test_batch_is_positive(self, tag: str) -> None:
        """Verify batch size is positive integer."""
        data = load_tag_data(tag)
        errors = []

        for filename, entries in data.items():
            for idx, entry in enumerate(entries):
                batch = entry.get('batch')
                if not isinstance(batch, int) or batch <= 0:
                    errors.append(
                        f'{filename}[{idx}] ({entry.get("model", "unknown")}): '
                        f'batch={batch} is not a positive integer'
                    )

        assert not errors, 'Entries with invalid batch size:\n' + '\n'.join(errors)


class TestReleaseFieldValidation:
    """Test release field validation with warnings."""

    @pytest.mark.parametrize('tag', TTSI_REF_VALID_TAGS)
    def test_release_field_populated(self, tag: str) -> None:
        """Warn if entries have null release field."""
        data = load_tag_data(tag)
        warnings = []

        for filename, entries in data.items():
            for idx, entry in enumerate(entries):
                if entry.get('release') is None:
                    warnings.append(
                        f'{filename}[{idx}] ({entry.get("model", "unknown")}): '
                        f'release is null'
                    )

        if warnings:
            # Use pytest.warns style - just print warning
            warning_msg = (
                f'\n⚠️  WARNING: {len(warnings)} entries in tag "{tag}" '
                f'have null release field:\n' + '\n'.join(warnings[:10])
            )
            if len(warnings) > 10:
                warning_msg += f'\n... and {len(warnings) - 10} more'
            print(warning_msg)


class TestTagComparison:
    """Test comparisons between different tags."""

    def test_03nov25_vs_15oct25_changes(self) -> None:
        """Document and validate expected differences between 03nov25 and 15oct25."""
        if '03nov25' not in TTSI_REF_VALID_TAGS or '15oct25' not in TTSI_REF_VALID_TAGS:
            pytest.skip('Both 03nov25 and 15oct25 tags required for comparison')

        data_new = load_tag_data('03nov25')
        data_old = load_tag_data('15oct25')

        # Count total entries
        count_new = sum(len(entries) for entries in data_new.values())
        count_old = sum(len(entries) for entries in data_old.values())

        # Expected: 6 new benchmarks added
        assert count_new >= count_old, \
            f'Expected new tag to have more entries, got {count_new} vs {count_old}'

        diff = count_new - count_old
        print('\n📊 Tag comparison (03nov25 vs 15oct25):')
        print(f'   Total benchmarks: {count_new} (new) vs {count_old} (old)')
        print(f'   Difference: +{diff} benchmarks')

        # Check for new hardware platforms
        def extract_hardware(data: Dict[str, List[Dict[str, Any]]]) -> Set[str]:
            hardware = set()
            for entries in data.values():
                for entry in entries:
                    hw = entry.get('hardware', '')
                    # Extract device name
                    device = hw.split('(')[0].strip()
                    hardware.add(device)
            return hardware

        hw_new = extract_hardware(data_new)
        hw_old = extract_hardware(data_old)
        new_platforms = hw_new - hw_old

        if new_platforms:
            print(f'   New hardware platforms: {", ".join(sorted(new_platforms))}')

        # Expected to have p150 (Blackhole)
        assert 'p150' in hw_new, 'Expected p150 (Blackhole) in new data'

    def test_default_tag_is_first_in_valid_tags(self) -> None:
        """Verify default tag is first in TTSI_REF_VALID_TAGS."""
        assert TTSI_REF_VALID_TAGS[0] == TTSI_REF_DEFAULT_TAG, \
            f'Default tag {TTSI_REF_DEFAULT_TAG} should be first in TTSI_REF_VALID_TAGS'


class TestTagNamingStandard:
    """Test tag naming conventions."""

    def test_tag_format_is_consistent(self) -> None:
        """Verify all tags follow consistent naming format."""
        # Expected format: DDmmmYY (e.g., 03nov25, 15oct25)
        # Allow single digit day as well (e.g., 3nov25)
        pattern = re.compile(r'^\d{1,2}[a-z]{3}\d{2}$')

        errors = []
        for tag in TTSI_REF_VALID_TAGS:
            if not pattern.match(tag):
                errors.append(
                    f'Tag "{tag}" does not match expected format DDmmmYY '
                    f'(e.g., 03nov25, 3nov25)'
                )

        assert not errors, 'Invalid tag formats:\n' + '\n'.join(errors)

    def test_tag_month_abbreviations(self) -> None:
        """Verify tags use valid month abbreviations."""
        valid_months = {
            'jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        }

        errors = []
        for tag in TTSI_REF_VALID_TAGS:
            # Extract month part (characters after digits)
            match = re.search(r'\d+([a-z]{3})\d+', tag)
            if match:
                month = match.group(1)
                if month not in valid_months:
                    errors.append(
                        f'Tag "{tag}" has invalid month abbreviation "{month}"'
                    )

        assert not errors, 'Invalid month abbreviations:\n' + '\n'.join(errors)


class TestDataProvenance:
    """Test data provenance and metadata completeness."""

    @pytest.mark.parametrize('tag', TTSI_REF_VALID_TAGS)
    def test_metadata_has_source_url(self, tag: str) -> None:
        """Verify metadata includes source URL."""
        metadata = load_metadata(tag)
        assert 'input_url' in metadata, f'Metadata for {tag} missing input_url'
        assert metadata['input_url'], f'Metadata for {tag} has empty input_url'

    @pytest.mark.parametrize('tag', TTSI_REF_VALID_TAGS)
    def test_metadata_has_parsed_date(self, tag: str) -> None:
        """Verify metadata includes parse date."""
        metadata = load_metadata(tag)
        assert 'parsed_date' in metadata, f'Metadata for {tag} missing parsed_date'

        # Verify date format (ISO 8601)
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+$')
        parsed_date = str(metadata['parsed_date'])
        assert date_pattern.match(parsed_date), \
            f'Metadata for {tag} has invalid parsed_date format: {parsed_date}'

    @pytest.mark.parametrize('tag', TTSI_REF_VALID_TAGS)
    def test_metadata_recommends_commit_hash(self, tag: str) -> None:
        """Recommend including git commit hash in metadata."""
        metadata = load_metadata(tag)

        if 'source_commit' not in metadata:
            print(
                f'\n💡 RECOMMENDATION: Add "source_commit" field to metadata '
                f'for tag "{tag}" to track exact source version'
            )

