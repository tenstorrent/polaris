#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
import numpy as np

causal_conv1d_fn = None

class RMSNormGated(SimNN.Module):
    def __init__(self, objname: str, hidden_size, eps=1e-5, group_size=None, norm_before_gate=True, device=None, dtype=None):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.name = objname
        self.eps = eps
        self.weight = F._from_shape(f'{self.name}_weight', shape=[hidden_size])
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def __call__(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        pass

# CPU implementation of mamba_chunk_scan_combined
def mamba_chunk_scan_combined(
    x,       # (B, L, H, P)
    dt,      # (B, L, H)
    A,       # (H,)
    B_,      # (B, L, G, N)
    C_,      # (B, L, G, N)
    ngroups: int,
    chunk_size=256,
    D=None,
    z=None, seq_idx=None,
    initial_states=None,
    module=None
):
    Bsz, L, H, P = x.shape
    N = B_.shape[-1]
    heads_per_group = H // ngroups
    # head2group = torch.arange(H) // heads_per_group  # (H,)

    # State: (B, H, P, N)
    if initial_states is not None:
        state = initial_states.clone()
    else:
        state = F._from_shape('state', shape=[Bsz, H, P, N])

    y = F._from_shape('y', shape=x.shape)
    y.set_module(module)

    # Expand A so it broadcasts with state: (H,) -> (1,H,1,N)
    A_exp = A.view(1, H, 1, 1)

    for l in range(L):
        # decay: (B, H, 1, 1)
        exop = F.exp(f'exop_{l}')
        exop.set_module(module)
        decay = exop(dt[:, l, :].view(Bsz, H, 1, 1) * A_exp)  # broadcasts over P,N

        # B_t, C_t: map head indices to group indices → (B, H, N)
        B_t = B_[:, l, :, :]  # (B, H, N)  ### should have used B_[:, l, head2group, :]
        C_t = C_[:, l, :, :]  # (B, H, N)

        # state update: broadcast B_t over P to match state/x
        # x[:, l, :, :] is (B,H,P) → unsqueeze(-1) -> (B,H,P,1)
        state = decay * state + B_t.unsqueeze(2) * x[:, l, :, :].unsqueeze(-1)  # (B,H,P,N)

        # output: multiply C_t over N, sum over N → (B,H,P)
        sumop = F.Sum(f'sumop_{l}', dim=-1)
        sumop.set_module(module)
        temp = sumop(C_t.unsqueeze(2) * state)
        assignop = F.Assign(f'assignop_{l}', slice=(slice(None), l, slice(None), slice(None)))
        assignop.set_module(module)
        module._op_hndls[assignop.name] = assignop
        module._tensors[y.name] = y
        y = assignop(y, temp)
        if D is not None:
            temp = D.view(1, H, 1) * x[:, l, :, :]
            addop = F.Add(f'addop_{l}')
            addop.set_module(module)
            module._op_hndls[addop.name] = addop
            temp = addop(temp, y[:, l, :, :]) # type: ignore
            assignop2 = F.Assign(f'assignop2_{l}', slice=(slice(None), l, slice(None), slice(None)))
            assignop2.set_module(module)
            y = assignop2(y, temp)

    return y

# CPU implementation of mamba_split_conv1d_scan_combined
def mamba_split_conv1d_scan_combined(
    zxbcdt: SimNN.SimTensor,       # (B, L, d_proj)
    conv1d_weight: SimNN.SimTensor,# (d_conv, kernel_size)
    conv1d_bias: SimNN.SimTensor,  # (d_conv,)
    dt_bias: SimNN.SimTensor,      # (nheads,)
    A: SimNN.SimTensor,            # (nheads,)
    D: SimNN.SimTensor = None,     # type: ignore
    d_inner: int = None,        # type: ignore[assignment]
    d_state: int = None,        # type: ignore[assignment]
    headdim: int = None,        # type: ignore[assignment]
    ngroups: int = 1,
    chunk_size: int = 256,
    seq_idx=None,
    activation: str = "swish",
    rmsnorm_weight: SimNN.SimTensor = None, # type: ignore[assignment]
    rmsnorm_eps: float = 1e-5,
    outproj_weight: SimNN.SimTensor = None, # type: ignore[assignment]
    outproj_bias: SimNN.SimTensor = None, # type: ignore[assignment]
    norm_before_gate: bool = False,
    initial_states: SimNN.SimTensor = None, # type: ignore[assignment]
    dt_limit = (0.0, float("inf")),
    module=None
):
    Bsz, L, d_proj = zxbcdt.shape
    nheads = A.shape[0]
    assert dt_bias.shape[0] == nheads, "dt_bias mismatch"

    # Infer dims if not given
    if d_inner is None or d_state is None or headdim is None:
        raise ValueError("Must pass d_inner, d_state, headdim for Mamba2 CPU fallback")

    # Validate shape
    expected_d_proj = 2*d_inner + 2*ngroups*d_state + nheads
    assert d_proj == expected_d_proj, \
        f"d_proj mismatch: got {d_proj}, expected {expected_d_proj}"

    # ---- Split ----
    splitop = F.Split('split_zxbcdt', axis=2, count=5)
    splitop.set_module(module)
    module._op_hndls[splitop.name] = splitop
    split_tensor = F._from_data('split_tensor', data=np.array([d_inner, d_inner, ngroups*d_state, ngroups*d_state, nheads]))
    z, x, B_part, C_part, dt = splitop(zxbcdt, split_tensor)

    # ---- Process dt ----
    softplusop = F.softplus('softplus')
    clipop = F.Clip('mamba_split_conv1d_scan_combined_clip', min=dt_limit[0], max=dt_limit[1])
    clipop.set_module(module)
    softplusop.set_module(module)
    module._op_hndls[clipop.name] = clipop
    module._op_hndls[softplusop.name] = softplusop
    dt = clipop(softplusop(dt + dt_bias))  # (B, L, nheads)

    # ---- Depthwise conv on x, B, C combined ----
    catop = F.ConcatX('catop', axis=-1)
    catop.set_module(module)
    module._op_hndls[catop.name] = catop
    xBC = catop(x, B_part, C_part)  # (B, L, d_inner + 2*ngroups*d_state)
    xBC_conv_in = xBC.transpose(1, 2)  # (B, Cin, L)
    weight = conv1d_weight.unsqueeze(1)  # type: ignore[attr-defined]
    conv1dOp = F.conv1d('conv1d', pads=[conv1d_weight.shape[-1] - 1, conv1d_weight.shape[-1] - 1],
                        group=xBC_conv_in.shape[1])
    conv1dOp.set_module(module)
    module._op_hndls[conv1dOp.name] = conv1dOp
    xBC_conv = conv1dOp(
        xBC_conv_in,
        weight,
        conv1d_bias,
    ).transpose(1, 2)[:, :L, :]  # back to (B, L, Cin) and trim to L

    # ---- Activation ----
    if activation in ("swish", "silu"):
        sigmoidop = F.Sigmoid('mamba_split_conv1d_scan_combined_sigmoid')
        module._op_hndls[sigmoidop.name] = sigmoidop
        xBC_conv = xBC_conv * sigmoidop(xBC_conv)
    else:
        raise NotImplementedError(f"Activation {activation} not supported")

    # ---- Split back into x, B, C after conv ----
    convsplitop = F.Split('convsplitop', axis=2, count=3)
    convsplittensor = F._from_data('convsplittensor', data=np.array([d_inner, ngroups*d_state, ngroups*d_state]))
    convsplitop.set_module(module)
    x_conv, B_conv, C_conv = convsplitop(xBC_conv, convsplittensor)

    # ---- Reshapex for scan ----
    x_scan = x_conv.reshape(Bsz, L, nheads, headdim)
    B_scan = B_conv.reshape(Bsz, L, ngroups, d_state)
    C_scan = C_conv.reshape(Bsz, L, ngroups, d_state)

    # ---- Scan ----
    y_scan = mamba_chunk_scan_combined(
        x_scan,
        dt,
        A,
        B_scan,
        C_scan,
        ngroups = ngroups,
        chunk_size=chunk_size,
        D=D,
        z=None,
        seq_idx=seq_idx,
        initial_states=initial_states,
        module=module
    )  # (B, L, H, headdim)

    y_flat = y_scan.reshape(Bsz, L, nheads * headdim)

    # ---- RMSNorm + gate ----
    if rmsnorm_weight is not None:
        meanop = F.Mean('mean', dim=-1)
        meanop.set_module(module)
        module._op_hndls[meanop.name] = meanop
        mu = meanop(y_flat * y_flat).unsqueeze(-1) ## y_flat.pow(2) substituted with mul ## unsqueeze for keepdim=True
        rmsnorm_eps_tensor = F._from_shape('rmsnorm_eps', shape=mu.shape)
        sqrtop = F.Sqrt(f'_sqrtop')
        reciprocalop = F.Reciprocal(f'_reciprocalop')
        sqrtop.set_module(module)
        reciprocalop.set_module(module)
        module._op_hndls[sqrtop.name] = sqrtop
        module._op_hndls[reciprocalop.name] = reciprocalop
        y_flat = y_flat * reciprocalop(sqrtop(mu + rmsnorm_eps_tensor)) * rmsnorm_weight

    sigmoidop = F.Sigmoid('sigmoid')
    sigmoidop.set_module(module)
    module._op_hndls[sigmoidop.name] = sigmoidop
    y_flat = y_flat * sigmoidop(z)  # gate

    # ---- Out projection ----
    nrows, ncols = outproj_weight.shape
    linearop = F.Linear('outproj', ncols, nrows, module=module)
    linearop.set_module(module)
    module._op_hndls[linearop.name] = linearop
    out = linearop(y_flat) #, outproj_weight, outproj_bias)

    return out


class Mamba2Simple(SimNN.Module):
    def __init__(
        self,
        objname,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        batch_size=1,
        seq_len=16,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.name = objname
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = F.Linear(f'{self.name}_in_proj', self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        out_channels = conv_dim
        in_channels = conv_dim
        groups = conv_dim
        kernel_size = d_conv
        self.conv1d_weight = F._from_data(f'{self.name}_conv1d_weight', data=np.ones((out_channels, in_channels // groups, kernel_size)))
        self.conv1d_bias = F._from_data(f'{self.name}_conv1d_bias', data=np.ones((out_channels,))) if conv_bias else None
        
        if self.learnable_init_states:
            assert not(self.learnable_init_states), 'learnable init states not implemented!'
        
        dt_expop = F.exp('expop_dt')
        dt = dt_expop(F._from_shape(f'{self.name}_dt', shape=[self.nheads]))
        dt_clipop = F.Clip('clip_dt', min=dt_init_floor)
        dt_clipop.set_module(self)
        dt = dt_clipop(dt)
        
        expm1op = F.exp('expm1')
        expm1op.set_module(self)
        inv_dt_logop = F.Log('log_inv_dt')
        inv_dt_logop.set_module(self)
        self.dt_bias = inv_dt_logop(expm1op(dt))

        # A parameter
        A = F._from_shape(f'{self.name}_A', shape=[self.nheads])
        A_logop = F.Log('log_A')
        A_logop.set_module(self)
        self.A_log = A_logop(A)

        # D "skip" parameter
        self.D = F._from_shape(f'{self.name}_D', shape=[self.nheads])

        # Extra normalization layer right before output projection
        self.norm = RMSNormGated(f'{self.name}_norm', self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        self.out_proj_weight = F._from_shape(f'{self.name}_out_proj_weight', shape=[self.d_model, self.d_inner])
        self.expop = F.exp('expop')
        self.expop.set_module(self)
        super().link_op2module()
        self.conv1d_weight.set_module(self)
        self.out_proj_weight.set_module(self)
        self.D.set_module(self)
    
    def create_input_tensors(self):
        self.input_tensors = {
                'x_in': F._from_shape('input_tensor', shape=[self.batch_size, self.seq_len, self.d_model]),
                }
        return

    def analytical_param_count(self):
        return 0

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def __call__(self, u=None, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        u = self.input_tensors['x_in'] if u is None else u

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = self.expop(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=None # repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                self.conv1d_weight.squeeze(1),
                self.conv1d_bias,   # type: ignore[arg-type]
                self.dt_bias,
                A,
                D=self.D,
                d_inner=self.d_inner,
                d_state=self.d_state,
                headdim=self.headdim,
                ngroups=self.ngroups,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj_weight,
                outproj_bias=None, # type: ignore[arg-type]
                norm_before_gate=False,
                initial_states=initial_states, # type: ignore[arg-type]
                **dt_limit_kwargs,
                module=self
            )
        else:
            assert self.use_mem_eff_path, "Only mem efficient path is implemented in this file"
        return out
