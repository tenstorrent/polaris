# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The resnet50 HW runners must call canonical's ``create_test_infra`` correctly.

The HW branch cannot be imported off-device (`ttnn` and the tt-metal `models.*`
packages are unavailable), so a signature mismatch there survives every local
check and only surfaces on booked hardware. That is exactly what happened:

    TypeError: create_test_infra() missing 3 required positional arguments:
               'act_dtype', 'weight_dtype', and 'math_fidelity'

Canonical takes those three as REQUIRED positionals
(``ttnn_resnet/tests/common/resnet50_test_infra.py:342-352``), while the Polaris
infra defaults them — so the call worked in simulation and failed on silicon.

This reads canonical's signature out of the vendored tt-metal source with `ast`
and checks the runners' call site against it, with no imports on either side.
It also guards the reverse direction: if upstream changes the signature, this
fails here rather than on an IRD box.
"""

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CANON_DIR = (
    _REPO / 'tt-metal/models/demos/vision/classification/resnet50'
    / 'ttnn_resnet/tests/common'
)
_CANONICAL = _CANON_DIR / 'resnet50_test_infra.py'
_CANONICAL_PERFORMANT = _CANON_DIR / 'resnet50_performant.py'
_HW_INFRA = _REPO / 'workloads/ttnn/resnet50/common/resnet50_test_infra_hw.py'
_RUNNERS = {
    'wh': _REPO / 'workloads/ttnn/resnet50/wh/test_resnet50_device_perf_wh.py',
    'bh': _REPO / 'workloads/ttnn/resnet50/bh/test_resnet50_device_perf_bh.py',
}

pytestmark = pytest.mark.skipif(
    not _CANONICAL.exists(),
    reason='vendored tt-metal checkout not present',
)


def _params_of(path, func_name):
    """(required_positional_names, all_param_names) of a top-level function."""
    tree = ast.parse(Path(path).read_text())
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == func_name
    )
    args = fn.args.args
    n_defaults = len(fn.args.defaults)
    required = [a.arg for a in args[:len(args) - n_defaults]]
    return required, [a.arg for a in args] + [a.arg for a in fn.args.kwonlyargs]


def _canonical_params():
    """(required, all) of canonical create_test_infra."""
    return _params_of(_CANONICAL, 'create_test_infra')


def _call_in(path, enclosing_func, callee):
    tree = ast.parse(Path(path).read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == enclosing_func
    )
    return next(
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, (ast.Name, ast.Attribute))
        and (n.func.id if isinstance(n.func, ast.Name) else n.func.attr) == callee
    )


def _runner_call(path):
    """The `create_test_infra(...)` call inside run_resnet50_e2e."""
    tree = ast.parse(path.read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == 'run_resnet50_e2e'
    )
    return next(
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == 'create_test_infra'
    )


@pytest.mark.unit
def test_canonical_still_requires_the_three_dtypes() -> None:
    """Pins the upstream contract this test is written against."""
    required, _ = _canonical_params()
    assert required == [
        'device', 'batch_size', 'act_dtype', 'weight_dtype', 'math_fidelity',
    ], f'canonical create_test_infra signature changed: required={required}'


@pytest.mark.unit
def test_hw_path_supplies_every_required_argument_of_run_resnet50_inference() -> None:
    """The HW path's real contract: run_ported_resnet50_inference -> canonical.

    The runners no longer call ``create_test_infra`` on the HW path — canonical's
    ``run_resnet50_inference`` builds the infra itself, because it also owns the
    input path (``models.tt_cnn.tt.pipeline`` does the host->device transfer).
    Hand-rolling that produced
    ``TT_FATAL: Device Operations expect device tensors as inputs``.
    """
    required, allowed = _params_of(_CANONICAL_PERFORMANT, 'run_resnet50_inference')
    assert required == [
        'device', 'device_batch_size', 'act_dtype', 'weight_dtype',
        'math_fidelity', 'model_location_generator',
    ], f'canonical run_resnet50_inference signature changed: required={required}'

    call = _call_in(_HW_INFRA, 'run_ported_resnet50_inference', 'run_resnet50_inference')
    supplied = list(required[:len(call.args)])
    supplied += [kw.arg for kw in call.keywords if kw.arg]

    missing = [p for p in required if p not in supplied]
    assert not missing, f'run_ported_resnet50_inference omits {missing}'

    unknown = [kw.arg for kw in call.keywords if kw.arg and kw.arg not in allowed]
    assert not unknown, f'unknown kwarg(s) passed to run_resnet50_inference: {unknown}'


@pytest.mark.unit
@pytest.mark.parametrize('arch', sorted(_RUNNERS))
def test_polaris_path_supplies_every_required_argument(arch) -> None:
    """The shim branch still calls create_test_infra directly.

    Its signature mirrors canonical's, so keeping the call conformant means the
    two paths stay swappable.
    """
    required, allowed = _canonical_params()
    call = _runner_call(_RUNNERS[arch])

    supplied = list(required[:len(call.args)])           # positionally
    supplied += [kw.arg for kw in call.keywords if kw.arg]  # by keyword

    missing = [p for p in required if p not in supplied]
    assert not missing, (
        f'{arch} runner omits {missing} from create_test_infra. Canonical takes these '
        'as required positionals, so this raises TypeError on hardware while the '
        'Polaris infra (which defaults them) accepts it silently.'
    )

    unknown = [kw.arg for kw in call.keywords if kw.arg and kw.arg not in allowed]
    assert not unknown, f'{arch} runner passes unknown kwarg(s) {unknown}'


@pytest.mark.unit
@pytest.mark.parametrize('arch', sorted(_RUNNERS))
def test_dtypes_are_the_canonical_operating_point(arch) -> None:
    """bfloat8_b / bfloat8_b / LoFi — not bfloat16, and not a bare int fidelity.

    The superseded fork defaulted to bfloat16 and `math_fidelity=1`, which doubles
    modelled activation and weight bytes and is the dominant error term for a
    bandwidth-bound CNN.
    """
    src = _RUNNERS[arch].read_text()
    for name, expected in (
        ('ACT_DTYPE', 'ttnn.bfloat8_b'),
        ('WEIGHT_DTYPE', 'ttnn.bfloat8_b'),
        ('MATH_FIDELITY', 'ttnn.MathFidelity.LoFi'),
    ):
        assert f'{name} = {expected}' in src, (
            f'{arch}: expected `{name} = {expected}` — the canonical device-perf '
            'operating point from tests/test_perf_device_resnet50.py:16'
        )
