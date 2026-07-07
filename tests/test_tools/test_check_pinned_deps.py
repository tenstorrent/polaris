# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

from tools.check_pinned_deps import _is_pinned, find_unpinned


@pytest.mark.parametrize("spec,expected", [
    # exact conda pins
    ("numpy=2.2.3", True),
    ("python=3.13.2", True),
    ("numpy=2.2.3=py313_0", True),        # exact pin with build string
    # exact pip pins
    ("pytest==8.3.4", True),
    ("types-requests==2.33.0.20260518", True),
    ("pkg==1!2.0", True),                 # PEP 440 epoch: '!' is not the '!=' range op
    # URL / VCS / PEP 508 direct references carry no exact version specifier and
    # are not expected in the env files -> flagged as unpinned for human review
    ("mypkg @ https://example.com/mypkg-1.0.whl", False),
    ("git+ssh://git@github.com/org/repo.git", False),
    # exact pip pin carrying an environment marker: the marker's '<' constrains
    # the environment, not the package version, so this stays pinned
    ('pkg==1.2.3; python_version<"3.12"', True),
    # unpinned — no version at all
    ("gitpython", False),
    ("ruff", False),
    # range / inequality operators still allow drift -> not an exact pin
    ("ruff>=0.15", False),
    ("numpy<3", False),
    ("pkg!=1.0", False),
    ("pkg~=1.2", False),
    # wildcard conda pin still allows patch drift -> not an exact pin
    ("pkg=1.2.*", False),
    # malformed pins with no version part (typos) -> not pinned
    ("pkg=", False),
    ("pkg==", False),
])
def test_is_pinned_accepts_only_exact_pins(spec, expected):
    assert _is_pinned(spec) is expected


def _write(tmp_path, text):
    p = tmp_path / "env.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_find_unpinned_flags_conda_and_pip_entries(tmp_path):
    env = _write(tmp_path, """
name: sample
channels:
  - conda-forge
dependencies:
  - numpy=2.2.3
  - gitpython
  - ruff>=0.15
  - pip:
    - pytest==8.3.4
    - types-requests
""")
    # channels are ignored; unpinned conda ('gitpython'), a range spec ('ruff>=0.15'),
    # and unpinned pip ('types-requests') are all flagged.
    assert find_unpinned(env) == ["gitpython", "ruff>=0.15", "pip: types-requests"]


def test_find_unpinned_empty_on_fully_pinned(tmp_path):
    env = _write(tmp_path, """
dependencies:
  - numpy=2.2.3
  - pip:
    - pytest==8.3.4
""")
    assert find_unpinned(env) == []


def test_find_unpinned_handles_empty_yaml(tmp_path):
    # yaml.safe_load returns None for an empty document; must not raise.
    env = _write(tmp_path, "")
    assert find_unpinned(env) == []


def test_find_unpinned_handles_empty_pip_block(tmp_path):
    # '- pip:' with no items yields {'pip': None}; must not raise a TypeError.
    env = _write(tmp_path, """
dependencies:
  - numpy=2.2.3
  - pip:
""")
    assert find_unpinned(env) == []
