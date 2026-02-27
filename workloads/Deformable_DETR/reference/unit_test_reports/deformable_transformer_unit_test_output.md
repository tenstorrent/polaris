
═════════════════════════════════════════════════════════════════
DEFORMABLE TRANSFORMER UNIT TEST SUITE - PyTorch vs TTSim
═════════════════════════════════════════════════════════════════

============================= test session starts ==============================
platform linux -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3.13
cachedir: .pytest_cache
rootdir: /home/aughag/Videos/TensTorrent/polaris
configfile: pyproject.toml
collecting ... collected 5 items

workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_encoder_layer FAILED
workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_encoder FAILED
workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_decoder_layer FAILED
workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_decoder FAILED
workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_full_transformer FAILED

=================================== FAILURES ===================================
______________________________ test_encoder_layer ______________________________
workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py:781: in test_encoder_layer
    _run_encoder_layer_test(batch, seq, dm, dffn, nl, nh, np_, ss, lsi, dt, tno)
workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py:734: in _run_encoder_layer_test
    tt_out = tt_layer(src_sim, pos_sim, ref_sim, ss_sim, lsi_sim)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
workloads/Deformable_DETR/models/deformable_transformer_ttsim.py:236: in __call__
    src2 = self.forward_ffn(src)
           ^^^^^^^^^^^^^^^^^^^^^
workloads/Deformable_DETR/models/deformable_transformer_ttsim.py:197: in forward_ffn
    src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
                                                      ^^^^^^^^^^^^^^^^^
ttsim/front/functional/sim_nn.py:264: in __call__
    Y = self.matmul(x, W_T)
        ^^^^^^^^^^^^^^^^^^^
ttsim/front/functional/op.py:158: in __call__
    self.perf_stats = self.sim_op.get_perf_counts(xinput,[self.otensor])
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ttsim/ops/op.py:86: in get_perf_counts
    shape_inf_func(inT, outT, self, **kwargs)
ttsim/ops/desc/math.py:521: in matmul_shape_inf
    oTList[0].data = compute_matmul(iTList, op)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
ttsim/ops/desc/data_compute.py:593: in compute_matmul
    return np.matmul(A, B)
           ^^^^^^^^^^^^^^^
E   ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 128 is different from 64)
_________________________________ test_encoder _________________________________
workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py:1090: in test_encoder
    tt_out = enc_tt(src_sim, ss_sim, lsi_sim, vr_sim, pos_sim)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
workloads/Deformable_DETR/models/deformable_transformer_ttsim.py:341: in __call__
    output = layer(
workloads/Deformable_DETR/models/deformable_transformer_ttsim.py:236: in __call__
    src2 = self.forward_ffn(src)
           ^^^^^^^^^^^^^^^^^^^^^
workloads/Deformable_DETR/models/deformable_transformer_ttsim.py:197: in forward_ffn
    src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
                                                      ^^^^^^^^^^^^^^^^^
ttsim/front/functional/sim_nn.py:264: in __call__
    Y = self.matmul(x, W_T)
        ^^^^^^^^^^^^^^^^^^^
ttsim/front/functional/op.py:158: in __call__
    self.perf_stats = self.sim_op.get_perf_counts(xinput,[self.otensor])
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ttsim/ops/op.py:86: in get_perf_counts
    shape_inf_func(inT, outT, self, **kwargs)
ttsim/ops/desc/math.py:521: in matmul_shape_inf
    oTList[0].data = compute_matmul(iTList, op)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
ttsim/ops/desc/data_compute.py:593: in compute_matmul
    return np.matmul(A, B)
           ^^^^^^^^^^^^^^^
E   ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 128 is different from 64)
______________________________ test_decoder_layer ______________________________
workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py:1427: in test_decoder_layer
    tt_out = tt_layer(tgt_sim, qpos_sim, ref_sim, src_sim, ss_sim, lsi_sim)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
workloads/Deformable_DETR/models/deformable_transformer_ttsim.py:470: in __call__
    tgt = self.forward_ffn(tgt)
          ^^^^^^^^^^^^^^^^^^^^^
workloads/Deformable_DETR/models/deformable_transformer_ttsim.py:419: in forward_ffn
    tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
                                                      ^^^^^^^^^^^^^^^^^
ttsim/front/functional/sim_nn.py:264: in __call__
    Y = self.matmul(x, W_T)
        ^^^^^^^^^^^^^^^^^^^
ttsim/front/functional/op.py:158: in __call__
    self.perf_stats = self.sim_op.get_perf_counts(xinput,[self.otensor])
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ttsim/ops/op.py:86: in get_perf_counts
    shape_inf_func(inT, outT, self, **kwargs)
ttsim/ops/desc/math.py:521: in matmul_shape_inf
    oTList[0].data = compute_matmul(iTList, op)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
ttsim/ops/desc/data_compute.py:593: in compute_matmul
    return np.matmul(A, B)
           ^^^^^^^^^^^^^^^
E   ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 128 is different from 64)
_________________________________ test_decoder _________________________________
workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py:1775: in test_decoder
    tt_out, _ = dec_tt(tgt_sim, ref_sim, src_sim, ss_sim, lsi_sim, vr_sim, qpos_sim)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
workloads/Deformable_DETR/models/deformable_transformer_ttsim.py:592: in __call__
    output = layer(
workloads/Deformable_DETR/models/deformable_transformer_ttsim.py:470: in __call__
    tgt = self.forward_ffn(tgt)
          ^^^^^^^^^^^^^^^^^^^^^
workloads/Deformable_DETR/models/deformable_transformer_ttsim.py:419: in forward_ffn
    tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
                                                      ^^^^^^^^^^^^^^^^^
ttsim/front/functional/sim_nn.py:264: in __call__
    Y = self.matmul(x, W_T)
        ^^^^^^^^^^^^^^^^^^^
ttsim/front/functional/op.py:158: in __call__
    self.perf_stats = self.sim_op.get_perf_counts(xinput,[self.otensor])
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ttsim/ops/op.py:86: in get_perf_counts
    shape_inf_func(inT, outT, self, **kwargs)
ttsim/ops/desc/math.py:521: in matmul_shape_inf
    oTList[0].data = compute_matmul(iTList, op)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
ttsim/ops/desc/data_compute.py:593: in compute_matmul
    return np.matmul(A, B)
           ^^^^^^^^^^^^^^^
E   ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 128 is different from 64)
____________________________ test_full_transformer _____________________________
workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py:2115: in test_full_transformer
    hs_tt, _, _, _, _ = tt_xfmr(srcs_sim, masks_sim, pos_sim, qe_sim)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
workloads/Deformable_DETR/models/deformable_transformer_ttsim.py:944: in __call__
    memory = self.encoder(
workloads/Deformable_DETR/models/deformable_transformer_ttsim.py:341: in __call__
    output = layer(
workloads/Deformable_DETR/models/deformable_transformer_ttsim.py:236: in __call__
    src2 = self.forward_ffn(src)
           ^^^^^^^^^^^^^^^^^^^^^
workloads/Deformable_DETR/models/deformable_transformer_ttsim.py:197: in forward_ffn
    src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
                                                      ^^^^^^^^^^^^^^^^^
ttsim/front/functional/sim_nn.py:264: in __call__
    Y = self.matmul(x, W_T)
        ^^^^^^^^^^^^^^^^^^^
ttsim/front/functional/op.py:158: in __call__
    self.perf_stats = self.sim_op.get_perf_counts(xinput,[self.otensor])
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ttsim/ops/op.py:86: in get_perf_counts
    shape_inf_func(inT, outT, self, **kwargs)
ttsim/ops/desc/math.py:521: in matmul_shape_inf
    oTList[0].data = compute_matmul(iTList, op)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
ttsim/ops/desc/data_compute.py:593: in compute_matmul
    return np.matmul(A, B)
           ^^^^^^^^^^^^^^^
E   ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 128 is different from 64)
=============================== warnings summary ===============================
workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_encoder
  /home/aughag/.local/lib/python3.13/site-packages/torch/functional.py:505: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:4381.)
    return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================== slowest durations ===============================
0.02s call     workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_full_transformer
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_encoder
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_encoder_layer
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_decoder
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_decoder_layer

(10 durations < 0.005s hidden.  Use -vv to show these durations.)
=========================== short test summary info ============================
FAILED workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_encoder_layer
FAILED workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_encoder
FAILED workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_decoder_layer
FAILED workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_decoder
FAILED workloads/Deformable_DETR/reference/unit_tests/test_deformable_transformer_unit.py::test_full_transformer
========================= 5 failed, 1 warning in 0.49s =========================

═════════════════════════════════════════════════════════════════
SUMMARY
═════════════════════════════════════════════════════════════════
MODULE                          SHAPE       NUMERICAL   TOTAL
─────────────────────────────────────────────────────────────────
TOTAL                           0/0           N/A         [92m✓ PASS[0m
═════════════════════════════════════════════════════════════════
