// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
module @jit_vit attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  ttcore.device_module {
    builtin.module @jit_vit attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
      func.func public @main(%arg0: tensor<8x3x224x224xf32>, %arg1: tensor<768x768xf32>, %arg2: tensor<8x1x768xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>, %arg6: tensor<768x768xf32>, %arg7: tensor<768x768xf32>, %arg8: tensor<768xf32>, %arg9: tensor<768xf32>, %arg10: tensor<768x3072xf32>, %arg11: tensor<3072x768xf32>, %arg12: tensor<768xf32>, %arg13: tensor<768xf32>, %arg14: tensor<768x768xf32>, %arg15: tensor<768x768xf32>, %arg16: tensor<768x768xf32>, %arg17: tensor<768x768xf32>, %arg18: tensor<768xf32>, %arg19: tensor<768xf32>, %arg20: tensor<768x3072xf32>, %arg21: tensor<3072x768xf32>, %arg22: tensor<768xf32>, %arg23: tensor<768xf32>, %arg24: tensor<768x768xf32>, %arg25: tensor<768x768xf32>, %arg26: tensor<768x768xf32>, %arg27: tensor<768x768xf32>, %arg28: tensor<768xf32>, %arg29: tensor<768xf32>, %arg30: tensor<768x3072xf32>, %arg31: tensor<3072x768xf32>, %arg32: tensor<768xf32>, %arg33: tensor<768xf32>, %arg34: tensor<768x768xf32>, %arg35: tensor<768x768xf32>, %arg36: tensor<768x768xf32>, %arg37: tensor<768x768xf32>, %arg38: tensor<768xf32>, %arg39: tensor<768xf32>, %arg40: tensor<768x3072xf32>, %arg41: tensor<3072x768xf32>, %arg42: tensor<768xf32>, %arg43: tensor<768xf32>, %arg44: tensor<768x768xf32>, %arg45: tensor<768x768xf32>, %arg46: tensor<768x768xf32>, %arg47: tensor<768x768xf32>, %arg48: tensor<768xf32>, %arg49: tensor<768xf32>, %arg50: tensor<768x3072xf32>, %arg51: tensor<3072x768xf32>, %arg52: tensor<768xf32>, %arg53: tensor<768xf32>, %arg54: tensor<768x768xf32>, %arg55: tensor<768x768xf32>, %arg56: tensor<768x768xf32>, %arg57: tensor<768x768xf32>, %arg58: tensor<768xf32>, %arg59: tensor<768xf32>, %arg60: tensor<768x3072xf32>, %arg61: tensor<3072x768xf32>, %arg62: tensor<768xf32>, %arg63: tensor<768xf32>, %arg64: tensor<768x768xf32>, %arg65: tensor<768x768xf32>, %arg66: tensor<768x768xf32>, %arg67: tensor<768x768xf32>, %arg68: tensor<768xf32>, %arg69: tensor<768xf32>, %arg70: tensor<768x3072xf32>, %arg71: tensor<3072x768xf32>, %arg72: tensor<768xf32>, %arg73: tensor<768xf32>, %arg74: tensor<768x768xf32>, %arg75: tensor<768x768xf32>, %arg76: tensor<768x768xf32>, %arg77: tensor<768x768xf32>, %arg78: tensor<768xf32>, %arg79: tensor<768xf32>, %arg80: tensor<768x3072xf32>, %arg81: tensor<3072x768xf32>, %arg82: tensor<768xf32>, %arg83: tensor<768xf32>, %arg84: tensor<768x768xf32>, %arg85: tensor<768x768xf32>, %arg86: tensor<768x768xf32>, %arg87: tensor<768x768xf32>, %arg88: tensor<768xf32>, %arg89: tensor<768xf32>, %arg90: tensor<768x3072xf32>, %arg91: tensor<3072x768xf32>, %arg92: tensor<768xf32>, %arg93: tensor<768xf32>, %arg94: tensor<768x768xf32>, %arg95: tensor<768x768xf32>, %arg96: tensor<768x768xf32>, %arg97: tensor<768x768xf32>, %arg98: tensor<768xf32>, %arg99: tensor<768xf32>, %arg100: tensor<768x3072xf32>, %arg101: tensor<3072x768xf32>, %arg102: tensor<768xf32>, %arg103: tensor<768xf32>, %arg104: tensor<768x768xf32>, %arg105: tensor<768x768xf32>, %arg106: tensor<768x768xf32>, %arg107: tensor<768x768xf32>, %arg108: tensor<768xf32>, %arg109: tensor<768xf32>, %arg110: tensor<768x3072xf32>, %arg111: tensor<3072x768xf32>, %arg112: tensor<768xf32>, %arg113: tensor<768xf32>, %arg114: tensor<768x768xf32>, %arg115: tensor<768x768xf32>, %arg116: tensor<768x768xf32>, %arg117: tensor<768x768xf32>, %arg118: tensor<768xf32>, %arg119: tensor<768xf32>, %arg120: tensor<768x3072xf32>, %arg121: tensor<3072x768xf32>, %arg122: tensor<768xf32>, %arg123: tensor<768xf32>) -> (tensor<8x197x768xf32> {jax.result_info = "result"}) {
        %0 = "ttir.constant"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
        %1 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
        %2 = "ttir.constant"() <{value = dense<0.797884583> : tensor<f32>}> : () -> tensor<f32>
        %3 = "ttir.constant"() <{value = dense<4.471500e-02> : tensor<f32>}> : () -> tensor<f32>
        %4 = "ttir.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
        %5 = "ttir.constant"() <{value = dense<6.400000e+01> : tensor<f32>}> : () -> tensor<f32>
        %6 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
        %7 = "ttir.constant"() <{value = dense<7.680000e+02> : tensor<f32>}> : () -> tensor<f32>
        %8 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
        %9 = "ttir.reshape"(%arg0) <{shape = [8 : i32, 3 : i32, 14 : i32, 16 : i32, 14 : i32, 16 : i32]}> : (tensor<8x3x224x224xf32>) -> tensor<8x3x14x16x14x16xf32>
        %10 = "ttir.permute"(%9) <{permutation = array<i64: 0, 2, 4, 1, 3, 5>}> : (tensor<8x3x14x16x14x16xf32>) -> tensor<8x14x14x3x16x16xf32>
        %11 = "ttir.reshape"(%10) <{shape = [8 : i32, 196 : i32, 768 : i32]}> : (tensor<8x14x14x3x16x16xf32>) -> tensor<8x196x768xf32>
        %12 = "ttir.dot_general"(%11, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x196x768xf32>, tensor<768x768xf32>) -> tensor<8x196x768xf32>
        %13 = "ttir.concat"(%arg2, %12) <{dim = 1 : si32}> : (tensor<8x1x768xf32>, tensor<8x196x768xf32>) -> tensor<8x197x768xf32>
        %14 = "ttir.broadcast"(%arg3) <{broadcast_dimensions = array<i64: 8, 1, 1>}> : (tensor<1x197x768xf32>) -> tensor<8x197x768xf32>
        %15 = "ttir.add"(%13, %14) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %16 = "ttir.sum"(%15) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %17 = "ttir.reshape"(%16) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %18 = "ttir.broadcast"(%17) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %19 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %20 = "ttir.broadcast"(%19) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %21 = "ttir.div"(%18, %20) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %22 = "ttir.broadcast"(%21) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %23 = "ttir.subtract"(%15, %22) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %24 = "ttir.multiply"(%23, %23) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %25 = "ttir.sum"(%24) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %26 = "ttir.reshape"(%25) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %27 = "ttir.broadcast"(%26) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %28 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %29 = "ttir.broadcast"(%28) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %30 = "ttir.div"(%27, %29) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %31 = "ttir.broadcast"(%21) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %32 = "ttir.subtract"(%15, %31) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %33 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %34 = "ttir.broadcast"(%33) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %35 = "ttir.add"(%30, %34) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %36 = "ttir.sqrt"(%35) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %37 = "ttir.broadcast"(%36) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %38 = "ttir.div"(%32, %37) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %39 = "ttir.reshape"(%arg8) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %40 = "ttir.broadcast"(%39) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %41 = "ttir.broadcast"(%40) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %42 = "ttir.multiply"(%38, %41) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %43 = "ttir.reshape"(%arg9) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %44 = "ttir.broadcast"(%43) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %45 = "ttir.broadcast"(%44) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %46 = "ttir.add"(%42, %45) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %47 = "ttir.dot_general"(%46, %arg4) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %48 = "ttir.dot_general"(%46, %arg5) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %49 = "ttir.dot_general"(%46, %arg6) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %50 = "ttir.reshape"(%47) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %51 = "ttir.permute"(%50) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %52 = "ttir.reshape"(%48) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %53 = "ttir.permute"(%52) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %54 = "ttir.reshape"(%49) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %55 = "ttir.permute"(%54) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %56 = "ttir.permute"(%53) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
        %57 = "ttir.dot_general"(%51, %56) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
        %58 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %59 = "ttir.typecast"(%58) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %60 = "ttir.reshape"(%59) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %61 = "ttir.broadcast"(%60) <{broadcast_dimensions = array<i64: 8, 12, 197, 197>}> : (tensor<1x1x1x1xf32>) -> tensor<8x12x197x197xf32>
        %62 = "ttir.div"(%57, %61) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %63 = "ttir.max"(%62) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %64 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %65 = "ttir.broadcast"(%64) <{broadcast_dimensions = array<i64: 8, 12, 197>}> : (tensor<1x1x1xf32>) -> tensor<8x12x197xf32>
        %66 = "ttir.maximum"(%65, %63) : (tensor<8x12x197xf32>, tensor<8x12x197xf32>) -> tensor<8x12x197xf32>
        %67 = "ttir.reshape"(%66) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %68 = "ttir.broadcast"(%67) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %69 = "ttir.broadcast"(%68) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %70 = "ttir.subtract"(%62, %69) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %71 = "ttir.exp"(%70) : (tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %72 = "ttir.sum"(%71) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %73 = "ttir.reshape"(%72) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %74 = "ttir.broadcast"(%73) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %75 = "ttir.broadcast"(%74) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %76 = "ttir.div"(%71, %75) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %77 = "ttir.dot_general"(%76, %55) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
        %78 = "ttir.permute"(%77) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
        %79 = "ttir.reshape"(%78) <{shape = [8 : i32, 197 : i32, 768 : i32]}> : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
        %80 = "ttir.dot_general"(%79, %arg7) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %81 = "ttir.add"(%15, %80) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %82 = "ttir.sum"(%81) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %83 = "ttir.reshape"(%82) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %84 = "ttir.broadcast"(%83) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %85 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %86 = "ttir.broadcast"(%85) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %87 = "ttir.div"(%84, %86) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %88 = "ttir.broadcast"(%87) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %89 = "ttir.subtract"(%81, %88) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %90 = "ttir.multiply"(%89, %89) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %91 = "ttir.sum"(%90) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %92 = "ttir.reshape"(%91) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %93 = "ttir.broadcast"(%92) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %94 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %95 = "ttir.broadcast"(%94) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %96 = "ttir.div"(%93, %95) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %97 = "ttir.broadcast"(%87) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %98 = "ttir.subtract"(%81, %97) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %99 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %100 = "ttir.broadcast"(%99) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %101 = "ttir.add"(%96, %100) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %102 = "ttir.sqrt"(%101) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %103 = "ttir.broadcast"(%102) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %104 = "ttir.div"(%98, %103) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %105 = "ttir.reshape"(%arg12) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %106 = "ttir.broadcast"(%105) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %107 = "ttir.broadcast"(%106) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %108 = "ttir.multiply"(%104, %107) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %109 = "ttir.reshape"(%arg13) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %110 = "ttir.broadcast"(%109) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %111 = "ttir.broadcast"(%110) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %112 = "ttir.add"(%108, %111) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %113 = "ttir.dot_general"(%112, %arg10) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
        %114 = "ttir.multiply"(%113, %113) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %115 = "ttir.multiply"(%114, %113) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %116 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %117 = "ttir.broadcast"(%116) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %118 = "ttir.multiply"(%117, %115) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %119 = "ttir.add"(%113, %118) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %120 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %121 = "ttir.broadcast"(%120) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %122 = "ttir.multiply"(%121, %119) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %123 = "ttir.tanh"(%122) : (tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %124 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %125 = "ttir.broadcast"(%124) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %126 = "ttir.add"(%125, %123) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %127 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %128 = "ttir.broadcast"(%127) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %129 = "ttir.multiply"(%128, %126) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %130 = "ttir.multiply"(%113, %129) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %131 = "ttir.dot_general"(%130, %arg11) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
        %132 = "ttir.add"(%81, %131) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %133 = "ttir.sum"(%132) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %134 = "ttir.reshape"(%133) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %135 = "ttir.broadcast"(%134) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %136 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %137 = "ttir.broadcast"(%136) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %138 = "ttir.div"(%135, %137) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %139 = "ttir.broadcast"(%138) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %140 = "ttir.subtract"(%132, %139) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %141 = "ttir.multiply"(%140, %140) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %142 = "ttir.sum"(%141) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %143 = "ttir.reshape"(%142) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %144 = "ttir.broadcast"(%143) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %145 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %146 = "ttir.broadcast"(%145) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %147 = "ttir.div"(%144, %146) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %148 = "ttir.broadcast"(%138) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %149 = "ttir.subtract"(%132, %148) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %150 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %151 = "ttir.broadcast"(%150) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %152 = "ttir.add"(%147, %151) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %153 = "ttir.sqrt"(%152) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %154 = "ttir.broadcast"(%153) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %155 = "ttir.div"(%149, %154) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %156 = "ttir.reshape"(%arg18) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %157 = "ttir.broadcast"(%156) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %158 = "ttir.broadcast"(%157) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %159 = "ttir.multiply"(%155, %158) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %160 = "ttir.reshape"(%arg19) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %161 = "ttir.broadcast"(%160) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %162 = "ttir.broadcast"(%161) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %163 = "ttir.add"(%159, %162) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %164 = "ttir.dot_general"(%163, %arg14) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %165 = "ttir.dot_general"(%163, %arg15) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %166 = "ttir.dot_general"(%163, %arg16) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %167 = "ttir.reshape"(%164) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %168 = "ttir.permute"(%167) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %169 = "ttir.reshape"(%165) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %170 = "ttir.permute"(%169) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %171 = "ttir.reshape"(%166) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %172 = "ttir.permute"(%171) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %173 = "ttir.permute"(%170) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
        %174 = "ttir.dot_general"(%168, %173) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
        %175 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %176 = "ttir.typecast"(%175) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %177 = "ttir.reshape"(%176) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %178 = "ttir.broadcast"(%177) <{broadcast_dimensions = array<i64: 8, 12, 197, 197>}> : (tensor<1x1x1x1xf32>) -> tensor<8x12x197x197xf32>
        %179 = "ttir.div"(%174, %178) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %180 = "ttir.max"(%179) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %181 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %182 = "ttir.broadcast"(%181) <{broadcast_dimensions = array<i64: 8, 12, 197>}> : (tensor<1x1x1xf32>) -> tensor<8x12x197xf32>
        %183 = "ttir.maximum"(%182, %180) : (tensor<8x12x197xf32>, tensor<8x12x197xf32>) -> tensor<8x12x197xf32>
        %184 = "ttir.reshape"(%183) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %185 = "ttir.broadcast"(%184) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %186 = "ttir.broadcast"(%185) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %187 = "ttir.subtract"(%179, %186) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %188 = "ttir.exp"(%187) : (tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %189 = "ttir.sum"(%188) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %190 = "ttir.reshape"(%189) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %191 = "ttir.broadcast"(%190) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %192 = "ttir.broadcast"(%191) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %193 = "ttir.div"(%188, %192) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %194 = "ttir.dot_general"(%193, %172) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
        %195 = "ttir.permute"(%194) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
        %196 = "ttir.reshape"(%195) <{shape = [8 : i32, 197 : i32, 768 : i32]}> : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
        %197 = "ttir.dot_general"(%196, %arg17) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %198 = "ttir.add"(%132, %197) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %199 = "ttir.sum"(%198) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %200 = "ttir.reshape"(%199) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %201 = "ttir.broadcast"(%200) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %202 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %203 = "ttir.broadcast"(%202) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %204 = "ttir.div"(%201, %203) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %205 = "ttir.broadcast"(%204) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %206 = "ttir.subtract"(%198, %205) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %207 = "ttir.multiply"(%206, %206) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %208 = "ttir.sum"(%207) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %209 = "ttir.reshape"(%208) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %210 = "ttir.broadcast"(%209) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %211 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %212 = "ttir.broadcast"(%211) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %213 = "ttir.div"(%210, %212) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %214 = "ttir.broadcast"(%204) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %215 = "ttir.subtract"(%198, %214) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %216 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %217 = "ttir.broadcast"(%216) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %218 = "ttir.add"(%213, %217) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %219 = "ttir.sqrt"(%218) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %220 = "ttir.broadcast"(%219) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %221 = "ttir.div"(%215, %220) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %222 = "ttir.reshape"(%arg22) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %223 = "ttir.broadcast"(%222) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %224 = "ttir.broadcast"(%223) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %225 = "ttir.multiply"(%221, %224) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %226 = "ttir.reshape"(%arg23) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %227 = "ttir.broadcast"(%226) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %228 = "ttir.broadcast"(%227) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %229 = "ttir.add"(%225, %228) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %230 = "ttir.dot_general"(%229, %arg20) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
        %231 = "ttir.multiply"(%230, %230) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %232 = "ttir.multiply"(%231, %230) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %233 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %234 = "ttir.broadcast"(%233) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %235 = "ttir.multiply"(%234, %232) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %236 = "ttir.add"(%230, %235) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %237 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %238 = "ttir.broadcast"(%237) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %239 = "ttir.multiply"(%238, %236) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %240 = "ttir.tanh"(%239) : (tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %241 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %242 = "ttir.broadcast"(%241) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %243 = "ttir.add"(%242, %240) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %244 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %245 = "ttir.broadcast"(%244) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %246 = "ttir.multiply"(%245, %243) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %247 = "ttir.multiply"(%230, %246) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %248 = "ttir.dot_general"(%247, %arg21) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
        %249 = "ttir.add"(%198, %248) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %250 = "ttir.sum"(%249) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %251 = "ttir.reshape"(%250) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %252 = "ttir.broadcast"(%251) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %253 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %254 = "ttir.broadcast"(%253) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %255 = "ttir.div"(%252, %254) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %256 = "ttir.broadcast"(%255) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %257 = "ttir.subtract"(%249, %256) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %258 = "ttir.multiply"(%257, %257) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %259 = "ttir.sum"(%258) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %260 = "ttir.reshape"(%259) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %261 = "ttir.broadcast"(%260) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %262 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %263 = "ttir.broadcast"(%262) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %264 = "ttir.div"(%261, %263) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %265 = "ttir.broadcast"(%255) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %266 = "ttir.subtract"(%249, %265) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %267 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %268 = "ttir.broadcast"(%267) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %269 = "ttir.add"(%264, %268) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %270 = "ttir.sqrt"(%269) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %271 = "ttir.broadcast"(%270) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %272 = "ttir.div"(%266, %271) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %273 = "ttir.reshape"(%arg28) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %274 = "ttir.broadcast"(%273) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %275 = "ttir.broadcast"(%274) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %276 = "ttir.multiply"(%272, %275) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %277 = "ttir.reshape"(%arg29) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %278 = "ttir.broadcast"(%277) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %279 = "ttir.broadcast"(%278) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %280 = "ttir.add"(%276, %279) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %281 = "ttir.dot_general"(%280, %arg24) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %282 = "ttir.dot_general"(%280, %arg25) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %283 = "ttir.dot_general"(%280, %arg26) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %284 = "ttir.reshape"(%281) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %285 = "ttir.permute"(%284) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %286 = "ttir.reshape"(%282) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %287 = "ttir.permute"(%286) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %288 = "ttir.reshape"(%283) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %289 = "ttir.permute"(%288) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %290 = "ttir.permute"(%287) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
        %291 = "ttir.dot_general"(%285, %290) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
        %292 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %293 = "ttir.typecast"(%292) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %294 = "ttir.reshape"(%293) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %295 = "ttir.broadcast"(%294) <{broadcast_dimensions = array<i64: 8, 12, 197, 197>}> : (tensor<1x1x1x1xf32>) -> tensor<8x12x197x197xf32>
        %296 = "ttir.div"(%291, %295) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %297 = "ttir.max"(%296) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %298 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %299 = "ttir.broadcast"(%298) <{broadcast_dimensions = array<i64: 8, 12, 197>}> : (tensor<1x1x1xf32>) -> tensor<8x12x197xf32>
        %300 = "ttir.maximum"(%299, %297) : (tensor<8x12x197xf32>, tensor<8x12x197xf32>) -> tensor<8x12x197xf32>
        %301 = "ttir.reshape"(%300) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %302 = "ttir.broadcast"(%301) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %303 = "ttir.broadcast"(%302) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %304 = "ttir.subtract"(%296, %303) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %305 = "ttir.exp"(%304) : (tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %306 = "ttir.sum"(%305) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %307 = "ttir.reshape"(%306) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %308 = "ttir.broadcast"(%307) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %309 = "ttir.broadcast"(%308) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %310 = "ttir.div"(%305, %309) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %311 = "ttir.dot_general"(%310, %289) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
        %312 = "ttir.permute"(%311) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
        %313 = "ttir.reshape"(%312) <{shape = [8 : i32, 197 : i32, 768 : i32]}> : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
        %314 = "ttir.dot_general"(%313, %arg27) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %315 = "ttir.add"(%249, %314) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %316 = "ttir.sum"(%315) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %317 = "ttir.reshape"(%316) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %318 = "ttir.broadcast"(%317) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %319 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %320 = "ttir.broadcast"(%319) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %321 = "ttir.div"(%318, %320) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %322 = "ttir.broadcast"(%321) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %323 = "ttir.subtract"(%315, %322) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %324 = "ttir.multiply"(%323, %323) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %325 = "ttir.sum"(%324) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %326 = "ttir.reshape"(%325) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %327 = "ttir.broadcast"(%326) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %328 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %329 = "ttir.broadcast"(%328) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %330 = "ttir.div"(%327, %329) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %331 = "ttir.broadcast"(%321) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %332 = "ttir.subtract"(%315, %331) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %333 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %334 = "ttir.broadcast"(%333) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %335 = "ttir.add"(%330, %334) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %336 = "ttir.sqrt"(%335) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %337 = "ttir.broadcast"(%336) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %338 = "ttir.div"(%332, %337) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %339 = "ttir.reshape"(%arg32) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %340 = "ttir.broadcast"(%339) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %341 = "ttir.broadcast"(%340) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %342 = "ttir.multiply"(%338, %341) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %343 = "ttir.reshape"(%arg33) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %344 = "ttir.broadcast"(%343) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %345 = "ttir.broadcast"(%344) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %346 = "ttir.add"(%342, %345) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %347 = "ttir.dot_general"(%346, %arg30) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
        %348 = "ttir.multiply"(%347, %347) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %349 = "ttir.multiply"(%348, %347) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %350 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %351 = "ttir.broadcast"(%350) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %352 = "ttir.multiply"(%351, %349) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %353 = "ttir.add"(%347, %352) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %354 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %355 = "ttir.broadcast"(%354) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %356 = "ttir.multiply"(%355, %353) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %357 = "ttir.tanh"(%356) : (tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %358 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %359 = "ttir.broadcast"(%358) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %360 = "ttir.add"(%359, %357) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %361 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %362 = "ttir.broadcast"(%361) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %363 = "ttir.multiply"(%362, %360) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %364 = "ttir.multiply"(%347, %363) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %365 = "ttir.dot_general"(%364, %arg31) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
        %366 = "ttir.add"(%315, %365) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %367 = "ttir.sum"(%366) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %368 = "ttir.reshape"(%367) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %369 = "ttir.broadcast"(%368) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %370 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %371 = "ttir.broadcast"(%370) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %372 = "ttir.div"(%369, %371) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %373 = "ttir.broadcast"(%372) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %374 = "ttir.subtract"(%366, %373) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %375 = "ttir.multiply"(%374, %374) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %376 = "ttir.sum"(%375) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %377 = "ttir.reshape"(%376) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %378 = "ttir.broadcast"(%377) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %379 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %380 = "ttir.broadcast"(%379) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %381 = "ttir.div"(%378, %380) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %382 = "ttir.broadcast"(%372) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %383 = "ttir.subtract"(%366, %382) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %384 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %385 = "ttir.broadcast"(%384) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %386 = "ttir.add"(%381, %385) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %387 = "ttir.sqrt"(%386) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %388 = "ttir.broadcast"(%387) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %389 = "ttir.div"(%383, %388) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %390 = "ttir.reshape"(%arg38) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %391 = "ttir.broadcast"(%390) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %392 = "ttir.broadcast"(%391) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %393 = "ttir.multiply"(%389, %392) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %394 = "ttir.reshape"(%arg39) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %395 = "ttir.broadcast"(%394) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %396 = "ttir.broadcast"(%395) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %397 = "ttir.add"(%393, %396) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %398 = "ttir.dot_general"(%397, %arg34) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %399 = "ttir.dot_general"(%397, %arg35) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %400 = "ttir.dot_general"(%397, %arg36) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %401 = "ttir.reshape"(%398) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %402 = "ttir.permute"(%401) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %403 = "ttir.reshape"(%399) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %404 = "ttir.permute"(%403) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %405 = "ttir.reshape"(%400) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %406 = "ttir.permute"(%405) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %407 = "ttir.permute"(%404) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
        %408 = "ttir.dot_general"(%402, %407) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
        %409 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %410 = "ttir.typecast"(%409) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %411 = "ttir.reshape"(%410) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %412 = "ttir.broadcast"(%411) <{broadcast_dimensions = array<i64: 8, 12, 197, 197>}> : (tensor<1x1x1x1xf32>) -> tensor<8x12x197x197xf32>
        %413 = "ttir.div"(%408, %412) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %414 = "ttir.max"(%413) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %415 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %416 = "ttir.broadcast"(%415) <{broadcast_dimensions = array<i64: 8, 12, 197>}> : (tensor<1x1x1xf32>) -> tensor<8x12x197xf32>
        %417 = "ttir.maximum"(%416, %414) : (tensor<8x12x197xf32>, tensor<8x12x197xf32>) -> tensor<8x12x197xf32>
        %418 = "ttir.reshape"(%417) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %419 = "ttir.broadcast"(%418) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %420 = "ttir.broadcast"(%419) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %421 = "ttir.subtract"(%413, %420) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %422 = "ttir.exp"(%421) : (tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %423 = "ttir.sum"(%422) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %424 = "ttir.reshape"(%423) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %425 = "ttir.broadcast"(%424) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %426 = "ttir.broadcast"(%425) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %427 = "ttir.div"(%422, %426) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %428 = "ttir.dot_general"(%427, %406) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
        %429 = "ttir.permute"(%428) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
        %430 = "ttir.reshape"(%429) <{shape = [8 : i32, 197 : i32, 768 : i32]}> : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
        %431 = "ttir.dot_general"(%430, %arg37) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %432 = "ttir.add"(%366, %431) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %433 = "ttir.sum"(%432) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %434 = "ttir.reshape"(%433) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %435 = "ttir.broadcast"(%434) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %436 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %437 = "ttir.broadcast"(%436) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %438 = "ttir.div"(%435, %437) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %439 = "ttir.broadcast"(%438) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %440 = "ttir.subtract"(%432, %439) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %441 = "ttir.multiply"(%440, %440) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %442 = "ttir.sum"(%441) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %443 = "ttir.reshape"(%442) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %444 = "ttir.broadcast"(%443) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %445 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %446 = "ttir.broadcast"(%445) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %447 = "ttir.div"(%444, %446) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %448 = "ttir.broadcast"(%438) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %449 = "ttir.subtract"(%432, %448) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %450 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %451 = "ttir.broadcast"(%450) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %452 = "ttir.add"(%447, %451) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %453 = "ttir.sqrt"(%452) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %454 = "ttir.broadcast"(%453) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %455 = "ttir.div"(%449, %454) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %456 = "ttir.reshape"(%arg42) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %457 = "ttir.broadcast"(%456) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %458 = "ttir.broadcast"(%457) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %459 = "ttir.multiply"(%455, %458) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %460 = "ttir.reshape"(%arg43) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %461 = "ttir.broadcast"(%460) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %462 = "ttir.broadcast"(%461) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %463 = "ttir.add"(%459, %462) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %464 = "ttir.dot_general"(%463, %arg40) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
        %465 = "ttir.multiply"(%464, %464) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %466 = "ttir.multiply"(%465, %464) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %467 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %468 = "ttir.broadcast"(%467) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %469 = "ttir.multiply"(%468, %466) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %470 = "ttir.add"(%464, %469) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %471 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %472 = "ttir.broadcast"(%471) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %473 = "ttir.multiply"(%472, %470) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %474 = "ttir.tanh"(%473) : (tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %475 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %476 = "ttir.broadcast"(%475) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %477 = "ttir.add"(%476, %474) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %478 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %479 = "ttir.broadcast"(%478) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %480 = "ttir.multiply"(%479, %477) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %481 = "ttir.multiply"(%464, %480) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %482 = "ttir.dot_general"(%481, %arg41) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
        %483 = "ttir.add"(%432, %482) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %484 = "ttir.sum"(%483) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %485 = "ttir.reshape"(%484) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %486 = "ttir.broadcast"(%485) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %487 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %488 = "ttir.broadcast"(%487) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %489 = "ttir.div"(%486, %488) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %490 = "ttir.broadcast"(%489) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %491 = "ttir.subtract"(%483, %490) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %492 = "ttir.multiply"(%491, %491) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %493 = "ttir.sum"(%492) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %494 = "ttir.reshape"(%493) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %495 = "ttir.broadcast"(%494) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %496 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %497 = "ttir.broadcast"(%496) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %498 = "ttir.div"(%495, %497) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %499 = "ttir.broadcast"(%489) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %500 = "ttir.subtract"(%483, %499) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %501 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %502 = "ttir.broadcast"(%501) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %503 = "ttir.add"(%498, %502) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %504 = "ttir.sqrt"(%503) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %505 = "ttir.broadcast"(%504) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %506 = "ttir.div"(%500, %505) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %507 = "ttir.reshape"(%arg48) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %508 = "ttir.broadcast"(%507) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %509 = "ttir.broadcast"(%508) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %510 = "ttir.multiply"(%506, %509) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %511 = "ttir.reshape"(%arg49) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %512 = "ttir.broadcast"(%511) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %513 = "ttir.broadcast"(%512) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %514 = "ttir.add"(%510, %513) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %515 = "ttir.dot_general"(%514, %arg44) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %516 = "ttir.dot_general"(%514, %arg45) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %517 = "ttir.dot_general"(%514, %arg46) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %518 = "ttir.reshape"(%515) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %519 = "ttir.permute"(%518) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %520 = "ttir.reshape"(%516) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %521 = "ttir.permute"(%520) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %522 = "ttir.reshape"(%517) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %523 = "ttir.permute"(%522) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %524 = "ttir.permute"(%521) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
        %525 = "ttir.dot_general"(%519, %524) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
        %526 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %527 = "ttir.typecast"(%526) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %528 = "ttir.reshape"(%527) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %529 = "ttir.broadcast"(%528) <{broadcast_dimensions = array<i64: 8, 12, 197, 197>}> : (tensor<1x1x1x1xf32>) -> tensor<8x12x197x197xf32>
        %530 = "ttir.div"(%525, %529) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %531 = "ttir.max"(%530) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %532 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %533 = "ttir.broadcast"(%532) <{broadcast_dimensions = array<i64: 8, 12, 197>}> : (tensor<1x1x1xf32>) -> tensor<8x12x197xf32>
        %534 = "ttir.maximum"(%533, %531) : (tensor<8x12x197xf32>, tensor<8x12x197xf32>) -> tensor<8x12x197xf32>
        %535 = "ttir.reshape"(%534) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %536 = "ttir.broadcast"(%535) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %537 = "ttir.broadcast"(%536) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %538 = "ttir.subtract"(%530, %537) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %539 = "ttir.exp"(%538) : (tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %540 = "ttir.sum"(%539) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %541 = "ttir.reshape"(%540) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %542 = "ttir.broadcast"(%541) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %543 = "ttir.broadcast"(%542) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %544 = "ttir.div"(%539, %543) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %545 = "ttir.dot_general"(%544, %523) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
        %546 = "ttir.permute"(%545) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
        %547 = "ttir.reshape"(%546) <{shape = [8 : i32, 197 : i32, 768 : i32]}> : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
        %548 = "ttir.dot_general"(%547, %arg47) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %549 = "ttir.add"(%483, %548) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %550 = "ttir.sum"(%549) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %551 = "ttir.reshape"(%550) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %552 = "ttir.broadcast"(%551) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %553 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %554 = "ttir.broadcast"(%553) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %555 = "ttir.div"(%552, %554) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %556 = "ttir.broadcast"(%555) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %557 = "ttir.subtract"(%549, %556) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %558 = "ttir.multiply"(%557, %557) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %559 = "ttir.sum"(%558) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %560 = "ttir.reshape"(%559) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %561 = "ttir.broadcast"(%560) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %562 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %563 = "ttir.broadcast"(%562) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %564 = "ttir.div"(%561, %563) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %565 = "ttir.broadcast"(%555) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %566 = "ttir.subtract"(%549, %565) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %567 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %568 = "ttir.broadcast"(%567) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %569 = "ttir.add"(%564, %568) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %570 = "ttir.sqrt"(%569) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %571 = "ttir.broadcast"(%570) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %572 = "ttir.div"(%566, %571) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %573 = "ttir.reshape"(%arg52) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %574 = "ttir.broadcast"(%573) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %575 = "ttir.broadcast"(%574) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %576 = "ttir.multiply"(%572, %575) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %577 = "ttir.reshape"(%arg53) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %578 = "ttir.broadcast"(%577) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %579 = "ttir.broadcast"(%578) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %580 = "ttir.add"(%576, %579) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %581 = "ttir.dot_general"(%580, %arg50) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
        %582 = "ttir.multiply"(%581, %581) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %583 = "ttir.multiply"(%582, %581) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %584 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %585 = "ttir.broadcast"(%584) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %586 = "ttir.multiply"(%585, %583) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %587 = "ttir.add"(%581, %586) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %588 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %589 = "ttir.broadcast"(%588) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %590 = "ttir.multiply"(%589, %587) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %591 = "ttir.tanh"(%590) : (tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %592 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %593 = "ttir.broadcast"(%592) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %594 = "ttir.add"(%593, %591) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %595 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %596 = "ttir.broadcast"(%595) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %597 = "ttir.multiply"(%596, %594) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %598 = "ttir.multiply"(%581, %597) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %599 = "ttir.dot_general"(%598, %arg51) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
        %600 = "ttir.add"(%549, %599) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %601 = "ttir.sum"(%600) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %602 = "ttir.reshape"(%601) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %603 = "ttir.broadcast"(%602) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %604 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %605 = "ttir.broadcast"(%604) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %606 = "ttir.div"(%603, %605) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %607 = "ttir.broadcast"(%606) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %608 = "ttir.subtract"(%600, %607) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %609 = "ttir.multiply"(%608, %608) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %610 = "ttir.sum"(%609) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %611 = "ttir.reshape"(%610) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %612 = "ttir.broadcast"(%611) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %613 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %614 = "ttir.broadcast"(%613) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %615 = "ttir.div"(%612, %614) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %616 = "ttir.broadcast"(%606) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %617 = "ttir.subtract"(%600, %616) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %618 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %619 = "ttir.broadcast"(%618) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %620 = "ttir.add"(%615, %619) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %621 = "ttir.sqrt"(%620) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %622 = "ttir.broadcast"(%621) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %623 = "ttir.div"(%617, %622) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %624 = "ttir.reshape"(%arg58) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %625 = "ttir.broadcast"(%624) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %626 = "ttir.broadcast"(%625) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %627 = "ttir.multiply"(%623, %626) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %628 = "ttir.reshape"(%arg59) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %629 = "ttir.broadcast"(%628) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %630 = "ttir.broadcast"(%629) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %631 = "ttir.add"(%627, %630) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %632 = "ttir.dot_general"(%631, %arg54) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %633 = "ttir.dot_general"(%631, %arg55) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %634 = "ttir.dot_general"(%631, %arg56) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %635 = "ttir.reshape"(%632) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %636 = "ttir.permute"(%635) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %637 = "ttir.reshape"(%633) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %638 = "ttir.permute"(%637) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %639 = "ttir.reshape"(%634) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %640 = "ttir.permute"(%639) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %641 = "ttir.permute"(%638) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
        %642 = "ttir.dot_general"(%636, %641) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
        %643 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %644 = "ttir.typecast"(%643) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %645 = "ttir.reshape"(%644) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %646 = "ttir.broadcast"(%645) <{broadcast_dimensions = array<i64: 8, 12, 197, 197>}> : (tensor<1x1x1x1xf32>) -> tensor<8x12x197x197xf32>
        %647 = "ttir.div"(%642, %646) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %648 = "ttir.max"(%647) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %649 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %650 = "ttir.broadcast"(%649) <{broadcast_dimensions = array<i64: 8, 12, 197>}> : (tensor<1x1x1xf32>) -> tensor<8x12x197xf32>
        %651 = "ttir.maximum"(%650, %648) : (tensor<8x12x197xf32>, tensor<8x12x197xf32>) -> tensor<8x12x197xf32>
        %652 = "ttir.reshape"(%651) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %653 = "ttir.broadcast"(%652) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %654 = "ttir.broadcast"(%653) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %655 = "ttir.subtract"(%647, %654) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %656 = "ttir.exp"(%655) : (tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %657 = "ttir.sum"(%656) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %658 = "ttir.reshape"(%657) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %659 = "ttir.broadcast"(%658) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %660 = "ttir.broadcast"(%659) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %661 = "ttir.div"(%656, %660) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %662 = "ttir.dot_general"(%661, %640) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
        %663 = "ttir.permute"(%662) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
        %664 = "ttir.reshape"(%663) <{shape = [8 : i32, 197 : i32, 768 : i32]}> : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
        %665 = "ttir.dot_general"(%664, %arg57) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %666 = "ttir.add"(%600, %665) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %667 = "ttir.sum"(%666) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %668 = "ttir.reshape"(%667) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %669 = "ttir.broadcast"(%668) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %670 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %671 = "ttir.broadcast"(%670) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %672 = "ttir.div"(%669, %671) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %673 = "ttir.broadcast"(%672) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %674 = "ttir.subtract"(%666, %673) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %675 = "ttir.multiply"(%674, %674) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %676 = "ttir.sum"(%675) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %677 = "ttir.reshape"(%676) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %678 = "ttir.broadcast"(%677) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %679 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %680 = "ttir.broadcast"(%679) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %681 = "ttir.div"(%678, %680) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %682 = "ttir.broadcast"(%672) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %683 = "ttir.subtract"(%666, %682) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %684 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %685 = "ttir.broadcast"(%684) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %686 = "ttir.add"(%681, %685) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %687 = "ttir.sqrt"(%686) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %688 = "ttir.broadcast"(%687) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %689 = "ttir.div"(%683, %688) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %690 = "ttir.reshape"(%arg62) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %691 = "ttir.broadcast"(%690) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %692 = "ttir.broadcast"(%691) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %693 = "ttir.multiply"(%689, %692) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %694 = "ttir.reshape"(%arg63) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %695 = "ttir.broadcast"(%694) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %696 = "ttir.broadcast"(%695) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %697 = "ttir.add"(%693, %696) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %698 = "ttir.dot_general"(%697, %arg60) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
        %699 = "ttir.multiply"(%698, %698) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %700 = "ttir.multiply"(%699, %698) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %701 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %702 = "ttir.broadcast"(%701) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %703 = "ttir.multiply"(%702, %700) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %704 = "ttir.add"(%698, %703) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %705 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %706 = "ttir.broadcast"(%705) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %707 = "ttir.multiply"(%706, %704) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %708 = "ttir.tanh"(%707) : (tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %709 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %710 = "ttir.broadcast"(%709) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %711 = "ttir.add"(%710, %708) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %712 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %713 = "ttir.broadcast"(%712) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %714 = "ttir.multiply"(%713, %711) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %715 = "ttir.multiply"(%698, %714) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %716 = "ttir.dot_general"(%715, %arg61) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
        %717 = "ttir.add"(%666, %716) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %718 = "ttir.sum"(%717) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %719 = "ttir.reshape"(%718) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %720 = "ttir.broadcast"(%719) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %721 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %722 = "ttir.broadcast"(%721) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %723 = "ttir.div"(%720, %722) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %724 = "ttir.broadcast"(%723) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %725 = "ttir.subtract"(%717, %724) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %726 = "ttir.multiply"(%725, %725) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %727 = "ttir.sum"(%726) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %728 = "ttir.reshape"(%727) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %729 = "ttir.broadcast"(%728) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %730 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %731 = "ttir.broadcast"(%730) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %732 = "ttir.div"(%729, %731) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %733 = "ttir.broadcast"(%723) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %734 = "ttir.subtract"(%717, %733) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %735 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %736 = "ttir.broadcast"(%735) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %737 = "ttir.add"(%732, %736) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %738 = "ttir.sqrt"(%737) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %739 = "ttir.broadcast"(%738) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %740 = "ttir.div"(%734, %739) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %741 = "ttir.reshape"(%arg68) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %742 = "ttir.broadcast"(%741) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %743 = "ttir.broadcast"(%742) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %744 = "ttir.multiply"(%740, %743) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %745 = "ttir.reshape"(%arg69) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %746 = "ttir.broadcast"(%745) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %747 = "ttir.broadcast"(%746) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %748 = "ttir.add"(%744, %747) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %749 = "ttir.dot_general"(%748, %arg64) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %750 = "ttir.dot_general"(%748, %arg65) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %751 = "ttir.dot_general"(%748, %arg66) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %752 = "ttir.reshape"(%749) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %753 = "ttir.permute"(%752) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %754 = "ttir.reshape"(%750) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %755 = "ttir.permute"(%754) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %756 = "ttir.reshape"(%751) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %757 = "ttir.permute"(%756) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %758 = "ttir.permute"(%755) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
        %759 = "ttir.dot_general"(%753, %758) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
        %760 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %761 = "ttir.typecast"(%760) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %762 = "ttir.reshape"(%761) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %763 = "ttir.broadcast"(%762) <{broadcast_dimensions = array<i64: 8, 12, 197, 197>}> : (tensor<1x1x1x1xf32>) -> tensor<8x12x197x197xf32>
        %764 = "ttir.div"(%759, %763) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %765 = "ttir.max"(%764) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %766 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %767 = "ttir.broadcast"(%766) <{broadcast_dimensions = array<i64: 8, 12, 197>}> : (tensor<1x1x1xf32>) -> tensor<8x12x197xf32>
        %768 = "ttir.maximum"(%767, %765) : (tensor<8x12x197xf32>, tensor<8x12x197xf32>) -> tensor<8x12x197xf32>
        %769 = "ttir.reshape"(%768) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %770 = "ttir.broadcast"(%769) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %771 = "ttir.broadcast"(%770) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %772 = "ttir.subtract"(%764, %771) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %773 = "ttir.exp"(%772) : (tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %774 = "ttir.sum"(%773) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %775 = "ttir.reshape"(%774) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %776 = "ttir.broadcast"(%775) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %777 = "ttir.broadcast"(%776) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %778 = "ttir.div"(%773, %777) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %779 = "ttir.dot_general"(%778, %757) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
        %780 = "ttir.permute"(%779) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
        %781 = "ttir.reshape"(%780) <{shape = [8 : i32, 197 : i32, 768 : i32]}> : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
        %782 = "ttir.dot_general"(%781, %arg67) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %783 = "ttir.add"(%717, %782) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %784 = "ttir.sum"(%783) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %785 = "ttir.reshape"(%784) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %786 = "ttir.broadcast"(%785) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %787 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %788 = "ttir.broadcast"(%787) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %789 = "ttir.div"(%786, %788) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %790 = "ttir.broadcast"(%789) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %791 = "ttir.subtract"(%783, %790) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %792 = "ttir.multiply"(%791, %791) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %793 = "ttir.sum"(%792) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %794 = "ttir.reshape"(%793) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %795 = "ttir.broadcast"(%794) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %796 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %797 = "ttir.broadcast"(%796) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %798 = "ttir.div"(%795, %797) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %799 = "ttir.broadcast"(%789) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %800 = "ttir.subtract"(%783, %799) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %801 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %802 = "ttir.broadcast"(%801) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %803 = "ttir.add"(%798, %802) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %804 = "ttir.sqrt"(%803) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %805 = "ttir.broadcast"(%804) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %806 = "ttir.div"(%800, %805) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %807 = "ttir.reshape"(%arg72) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %808 = "ttir.broadcast"(%807) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %809 = "ttir.broadcast"(%808) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %810 = "ttir.multiply"(%806, %809) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %811 = "ttir.reshape"(%arg73) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %812 = "ttir.broadcast"(%811) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %813 = "ttir.broadcast"(%812) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %814 = "ttir.add"(%810, %813) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %815 = "ttir.dot_general"(%814, %arg70) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
        %816 = "ttir.multiply"(%815, %815) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %817 = "ttir.multiply"(%816, %815) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %818 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %819 = "ttir.broadcast"(%818) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %820 = "ttir.multiply"(%819, %817) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %821 = "ttir.add"(%815, %820) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %822 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %823 = "ttir.broadcast"(%822) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %824 = "ttir.multiply"(%823, %821) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %825 = "ttir.tanh"(%824) : (tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %826 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %827 = "ttir.broadcast"(%826) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %828 = "ttir.add"(%827, %825) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %829 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %830 = "ttir.broadcast"(%829) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %831 = "ttir.multiply"(%830, %828) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %832 = "ttir.multiply"(%815, %831) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %833 = "ttir.dot_general"(%832, %arg71) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
        %834 = "ttir.add"(%783, %833) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %835 = "ttir.sum"(%834) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %836 = "ttir.reshape"(%835) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %837 = "ttir.broadcast"(%836) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %838 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %839 = "ttir.broadcast"(%838) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %840 = "ttir.div"(%837, %839) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %841 = "ttir.broadcast"(%840) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %842 = "ttir.subtract"(%834, %841) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %843 = "ttir.multiply"(%842, %842) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %844 = "ttir.sum"(%843) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %845 = "ttir.reshape"(%844) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %846 = "ttir.broadcast"(%845) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %847 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %848 = "ttir.broadcast"(%847) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %849 = "ttir.div"(%846, %848) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %850 = "ttir.broadcast"(%840) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %851 = "ttir.subtract"(%834, %850) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %852 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %853 = "ttir.broadcast"(%852) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %854 = "ttir.add"(%849, %853) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %855 = "ttir.sqrt"(%854) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %856 = "ttir.broadcast"(%855) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %857 = "ttir.div"(%851, %856) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %858 = "ttir.reshape"(%arg78) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %859 = "ttir.broadcast"(%858) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %860 = "ttir.broadcast"(%859) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %861 = "ttir.multiply"(%857, %860) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %862 = "ttir.reshape"(%arg79) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %863 = "ttir.broadcast"(%862) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %864 = "ttir.broadcast"(%863) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %865 = "ttir.add"(%861, %864) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %866 = "ttir.dot_general"(%865, %arg74) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %867 = "ttir.dot_general"(%865, %arg75) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %868 = "ttir.dot_general"(%865, %arg76) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %869 = "ttir.reshape"(%866) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %870 = "ttir.permute"(%869) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %871 = "ttir.reshape"(%867) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %872 = "ttir.permute"(%871) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %873 = "ttir.reshape"(%868) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %874 = "ttir.permute"(%873) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %875 = "ttir.permute"(%872) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
        %876 = "ttir.dot_general"(%870, %875) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
        %877 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %878 = "ttir.typecast"(%877) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %879 = "ttir.reshape"(%878) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %880 = "ttir.broadcast"(%879) <{broadcast_dimensions = array<i64: 8, 12, 197, 197>}> : (tensor<1x1x1x1xf32>) -> tensor<8x12x197x197xf32>
        %881 = "ttir.div"(%876, %880) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %882 = "ttir.max"(%881) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %883 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %884 = "ttir.broadcast"(%883) <{broadcast_dimensions = array<i64: 8, 12, 197>}> : (tensor<1x1x1xf32>) -> tensor<8x12x197xf32>
        %885 = "ttir.maximum"(%884, %882) : (tensor<8x12x197xf32>, tensor<8x12x197xf32>) -> tensor<8x12x197xf32>
        %886 = "ttir.reshape"(%885) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %887 = "ttir.broadcast"(%886) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %888 = "ttir.broadcast"(%887) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %889 = "ttir.subtract"(%881, %888) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %890 = "ttir.exp"(%889) : (tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %891 = "ttir.sum"(%890) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %892 = "ttir.reshape"(%891) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %893 = "ttir.broadcast"(%892) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %894 = "ttir.broadcast"(%893) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %895 = "ttir.div"(%890, %894) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %896 = "ttir.dot_general"(%895, %874) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
        %897 = "ttir.permute"(%896) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
        %898 = "ttir.reshape"(%897) <{shape = [8 : i32, 197 : i32, 768 : i32]}> : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
        %899 = "ttir.dot_general"(%898, %arg77) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %900 = "ttir.add"(%834, %899) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %901 = "ttir.sum"(%900) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %902 = "ttir.reshape"(%901) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %903 = "ttir.broadcast"(%902) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %904 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %905 = "ttir.broadcast"(%904) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %906 = "ttir.div"(%903, %905) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %907 = "ttir.broadcast"(%906) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %908 = "ttir.subtract"(%900, %907) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %909 = "ttir.multiply"(%908, %908) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %910 = "ttir.sum"(%909) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %911 = "ttir.reshape"(%910) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %912 = "ttir.broadcast"(%911) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %913 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %914 = "ttir.broadcast"(%913) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %915 = "ttir.div"(%912, %914) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %916 = "ttir.broadcast"(%906) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %917 = "ttir.subtract"(%900, %916) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %918 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %919 = "ttir.broadcast"(%918) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %920 = "ttir.add"(%915, %919) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %921 = "ttir.sqrt"(%920) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %922 = "ttir.broadcast"(%921) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %923 = "ttir.div"(%917, %922) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %924 = "ttir.reshape"(%arg82) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %925 = "ttir.broadcast"(%924) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %926 = "ttir.broadcast"(%925) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %927 = "ttir.multiply"(%923, %926) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %928 = "ttir.reshape"(%arg83) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %929 = "ttir.broadcast"(%928) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %930 = "ttir.broadcast"(%929) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %931 = "ttir.add"(%927, %930) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %932 = "ttir.dot_general"(%931, %arg80) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
        %933 = "ttir.multiply"(%932, %932) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %934 = "ttir.multiply"(%933, %932) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %935 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %936 = "ttir.broadcast"(%935) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %937 = "ttir.multiply"(%936, %934) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %938 = "ttir.add"(%932, %937) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %939 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %940 = "ttir.broadcast"(%939) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %941 = "ttir.multiply"(%940, %938) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %942 = "ttir.tanh"(%941) : (tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %943 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %944 = "ttir.broadcast"(%943) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %945 = "ttir.add"(%944, %942) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %946 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %947 = "ttir.broadcast"(%946) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %948 = "ttir.multiply"(%947, %945) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %949 = "ttir.multiply"(%932, %948) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %950 = "ttir.dot_general"(%949, %arg81) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
        %951 = "ttir.add"(%900, %950) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %952 = "ttir.sum"(%951) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %953 = "ttir.reshape"(%952) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %954 = "ttir.broadcast"(%953) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %955 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %956 = "ttir.broadcast"(%955) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %957 = "ttir.div"(%954, %956) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %958 = "ttir.broadcast"(%957) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %959 = "ttir.subtract"(%951, %958) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %960 = "ttir.multiply"(%959, %959) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %961 = "ttir.sum"(%960) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %962 = "ttir.reshape"(%961) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %963 = "ttir.broadcast"(%962) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %964 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %965 = "ttir.broadcast"(%964) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %966 = "ttir.div"(%963, %965) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %967 = "ttir.broadcast"(%957) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %968 = "ttir.subtract"(%951, %967) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %969 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %970 = "ttir.broadcast"(%969) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %971 = "ttir.add"(%966, %970) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %972 = "ttir.sqrt"(%971) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %973 = "ttir.broadcast"(%972) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %974 = "ttir.div"(%968, %973) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %975 = "ttir.reshape"(%arg88) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %976 = "ttir.broadcast"(%975) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %977 = "ttir.broadcast"(%976) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %978 = "ttir.multiply"(%974, %977) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %979 = "ttir.reshape"(%arg89) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %980 = "ttir.broadcast"(%979) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %981 = "ttir.broadcast"(%980) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %982 = "ttir.add"(%978, %981) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %983 = "ttir.dot_general"(%982, %arg84) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %984 = "ttir.dot_general"(%982, %arg85) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %985 = "ttir.dot_general"(%982, %arg86) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %986 = "ttir.reshape"(%983) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %987 = "ttir.permute"(%986) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %988 = "ttir.reshape"(%984) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %989 = "ttir.permute"(%988) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %990 = "ttir.reshape"(%985) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %991 = "ttir.permute"(%990) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %992 = "ttir.permute"(%989) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
        %993 = "ttir.dot_general"(%987, %992) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
        %994 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %995 = "ttir.typecast"(%994) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %996 = "ttir.reshape"(%995) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %997 = "ttir.broadcast"(%996) <{broadcast_dimensions = array<i64: 8, 12, 197, 197>}> : (tensor<1x1x1x1xf32>) -> tensor<8x12x197x197xf32>
        %998 = "ttir.div"(%993, %997) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %999 = "ttir.max"(%998) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %1000 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1001 = "ttir.broadcast"(%1000) <{broadcast_dimensions = array<i64: 8, 12, 197>}> : (tensor<1x1x1xf32>) -> tensor<8x12x197xf32>
        %1002 = "ttir.maximum"(%1001, %999) : (tensor<8x12x197xf32>, tensor<8x12x197xf32>) -> tensor<8x12x197xf32>
        %1003 = "ttir.reshape"(%1002) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %1004 = "ttir.broadcast"(%1003) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %1005 = "ttir.broadcast"(%1004) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %1006 = "ttir.subtract"(%998, %1005) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1007 = "ttir.exp"(%1006) : (tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1008 = "ttir.sum"(%1007) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %1009 = "ttir.reshape"(%1008) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %1010 = "ttir.broadcast"(%1009) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %1011 = "ttir.broadcast"(%1010) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %1012 = "ttir.div"(%1007, %1011) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1013 = "ttir.dot_general"(%1012, %991) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
        %1014 = "ttir.permute"(%1013) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
        %1015 = "ttir.reshape"(%1014) <{shape = [8 : i32, 197 : i32, 768 : i32]}> : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
        %1016 = "ttir.dot_general"(%1015, %arg87) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %1017 = "ttir.add"(%951, %1016) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1018 = "ttir.sum"(%1017) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %1019 = "ttir.reshape"(%1018) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %1020 = "ttir.broadcast"(%1019) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1021 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1022 = "ttir.broadcast"(%1021) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1023 = "ttir.div"(%1020, %1022) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1024 = "ttir.broadcast"(%1023) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1025 = "ttir.subtract"(%1017, %1024) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1026 = "ttir.multiply"(%1025, %1025) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1027 = "ttir.sum"(%1026) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %1028 = "ttir.reshape"(%1027) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %1029 = "ttir.broadcast"(%1028) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1030 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1031 = "ttir.broadcast"(%1030) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1032 = "ttir.div"(%1029, %1031) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1033 = "ttir.broadcast"(%1023) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1034 = "ttir.subtract"(%1017, %1033) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1035 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1036 = "ttir.broadcast"(%1035) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1037 = "ttir.add"(%1032, %1036) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1038 = "ttir.sqrt"(%1037) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1039 = "ttir.broadcast"(%1038) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1040 = "ttir.div"(%1034, %1039) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1041 = "ttir.reshape"(%arg92) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1042 = "ttir.broadcast"(%1041) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1043 = "ttir.broadcast"(%1042) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %1044 = "ttir.multiply"(%1040, %1043) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1045 = "ttir.reshape"(%arg93) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1046 = "ttir.broadcast"(%1045) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1047 = "ttir.broadcast"(%1046) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %1048 = "ttir.add"(%1044, %1047) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1049 = "ttir.dot_general"(%1048, %arg90) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
        %1050 = "ttir.multiply"(%1049, %1049) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1051 = "ttir.multiply"(%1050, %1049) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1052 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1053 = "ttir.broadcast"(%1052) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1054 = "ttir.multiply"(%1053, %1051) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1055 = "ttir.add"(%1049, %1054) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1056 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1057 = "ttir.broadcast"(%1056) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1058 = "ttir.multiply"(%1057, %1055) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1059 = "ttir.tanh"(%1058) : (tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1060 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1061 = "ttir.broadcast"(%1060) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1062 = "ttir.add"(%1061, %1059) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1063 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1064 = "ttir.broadcast"(%1063) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1065 = "ttir.multiply"(%1064, %1062) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1066 = "ttir.multiply"(%1049, %1065) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1067 = "ttir.dot_general"(%1066, %arg91) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
        %1068 = "ttir.add"(%1017, %1067) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1069 = "ttir.sum"(%1068) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %1070 = "ttir.reshape"(%1069) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %1071 = "ttir.broadcast"(%1070) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1072 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1073 = "ttir.broadcast"(%1072) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1074 = "ttir.div"(%1071, %1073) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1075 = "ttir.broadcast"(%1074) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1076 = "ttir.subtract"(%1068, %1075) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1077 = "ttir.multiply"(%1076, %1076) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1078 = "ttir.sum"(%1077) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %1079 = "ttir.reshape"(%1078) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %1080 = "ttir.broadcast"(%1079) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1081 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1082 = "ttir.broadcast"(%1081) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1083 = "ttir.div"(%1080, %1082) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1084 = "ttir.broadcast"(%1074) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1085 = "ttir.subtract"(%1068, %1084) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1086 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1087 = "ttir.broadcast"(%1086) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1088 = "ttir.add"(%1083, %1087) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1089 = "ttir.sqrt"(%1088) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1090 = "ttir.broadcast"(%1089) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1091 = "ttir.div"(%1085, %1090) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1092 = "ttir.reshape"(%arg98) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1093 = "ttir.broadcast"(%1092) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1094 = "ttir.broadcast"(%1093) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %1095 = "ttir.multiply"(%1091, %1094) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1096 = "ttir.reshape"(%arg99) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1097 = "ttir.broadcast"(%1096) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1098 = "ttir.broadcast"(%1097) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %1099 = "ttir.add"(%1095, %1098) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1100 = "ttir.dot_general"(%1099, %arg94) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %1101 = "ttir.dot_general"(%1099, %arg95) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %1102 = "ttir.dot_general"(%1099, %arg96) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %1103 = "ttir.reshape"(%1100) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %1104 = "ttir.permute"(%1103) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %1105 = "ttir.reshape"(%1101) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %1106 = "ttir.permute"(%1105) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %1107 = "ttir.reshape"(%1102) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %1108 = "ttir.permute"(%1107) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %1109 = "ttir.permute"(%1106) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
        %1110 = "ttir.dot_general"(%1104, %1109) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
        %1111 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %1112 = "ttir.typecast"(%1111) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %1113 = "ttir.reshape"(%1112) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %1114 = "ttir.broadcast"(%1113) <{broadcast_dimensions = array<i64: 8, 12, 197, 197>}> : (tensor<1x1x1x1xf32>) -> tensor<8x12x197x197xf32>
        %1115 = "ttir.div"(%1110, %1114) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1116 = "ttir.max"(%1115) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %1117 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1118 = "ttir.broadcast"(%1117) <{broadcast_dimensions = array<i64: 8, 12, 197>}> : (tensor<1x1x1xf32>) -> tensor<8x12x197xf32>
        %1119 = "ttir.maximum"(%1118, %1116) : (tensor<8x12x197xf32>, tensor<8x12x197xf32>) -> tensor<8x12x197xf32>
        %1120 = "ttir.reshape"(%1119) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %1121 = "ttir.broadcast"(%1120) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %1122 = "ttir.broadcast"(%1121) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %1123 = "ttir.subtract"(%1115, %1122) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1124 = "ttir.exp"(%1123) : (tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1125 = "ttir.sum"(%1124) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %1126 = "ttir.reshape"(%1125) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %1127 = "ttir.broadcast"(%1126) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %1128 = "ttir.broadcast"(%1127) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %1129 = "ttir.div"(%1124, %1128) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1130 = "ttir.dot_general"(%1129, %1108) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
        %1131 = "ttir.permute"(%1130) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
        %1132 = "ttir.reshape"(%1131) <{shape = [8 : i32, 197 : i32, 768 : i32]}> : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
        %1133 = "ttir.dot_general"(%1132, %arg97) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %1134 = "ttir.add"(%1068, %1133) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1135 = "ttir.sum"(%1134) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %1136 = "ttir.reshape"(%1135) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %1137 = "ttir.broadcast"(%1136) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1138 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1139 = "ttir.broadcast"(%1138) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1140 = "ttir.div"(%1137, %1139) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1141 = "ttir.broadcast"(%1140) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1142 = "ttir.subtract"(%1134, %1141) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1143 = "ttir.multiply"(%1142, %1142) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1144 = "ttir.sum"(%1143) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %1145 = "ttir.reshape"(%1144) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %1146 = "ttir.broadcast"(%1145) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1147 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1148 = "ttir.broadcast"(%1147) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1149 = "ttir.div"(%1146, %1148) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1150 = "ttir.broadcast"(%1140) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1151 = "ttir.subtract"(%1134, %1150) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1152 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1153 = "ttir.broadcast"(%1152) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1154 = "ttir.add"(%1149, %1153) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1155 = "ttir.sqrt"(%1154) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1156 = "ttir.broadcast"(%1155) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1157 = "ttir.div"(%1151, %1156) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1158 = "ttir.reshape"(%arg102) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1159 = "ttir.broadcast"(%1158) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1160 = "ttir.broadcast"(%1159) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %1161 = "ttir.multiply"(%1157, %1160) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1162 = "ttir.reshape"(%arg103) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1163 = "ttir.broadcast"(%1162) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1164 = "ttir.broadcast"(%1163) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %1165 = "ttir.add"(%1161, %1164) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1166 = "ttir.dot_general"(%1165, %arg100) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
        %1167 = "ttir.multiply"(%1166, %1166) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1168 = "ttir.multiply"(%1167, %1166) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1169 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1170 = "ttir.broadcast"(%1169) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1171 = "ttir.multiply"(%1170, %1168) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1172 = "ttir.add"(%1166, %1171) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1173 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1174 = "ttir.broadcast"(%1173) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1175 = "ttir.multiply"(%1174, %1172) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1176 = "ttir.tanh"(%1175) : (tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1177 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1178 = "ttir.broadcast"(%1177) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1179 = "ttir.add"(%1178, %1176) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1180 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1181 = "ttir.broadcast"(%1180) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1182 = "ttir.multiply"(%1181, %1179) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1183 = "ttir.multiply"(%1166, %1182) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1184 = "ttir.dot_general"(%1183, %arg101) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
        %1185 = "ttir.add"(%1134, %1184) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1186 = "ttir.sum"(%1185) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %1187 = "ttir.reshape"(%1186) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %1188 = "ttir.broadcast"(%1187) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1189 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1190 = "ttir.broadcast"(%1189) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1191 = "ttir.div"(%1188, %1190) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1192 = "ttir.broadcast"(%1191) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1193 = "ttir.subtract"(%1185, %1192) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1194 = "ttir.multiply"(%1193, %1193) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1195 = "ttir.sum"(%1194) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %1196 = "ttir.reshape"(%1195) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %1197 = "ttir.broadcast"(%1196) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1198 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1199 = "ttir.broadcast"(%1198) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1200 = "ttir.div"(%1197, %1199) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1201 = "ttir.broadcast"(%1191) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1202 = "ttir.subtract"(%1185, %1201) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1203 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1204 = "ttir.broadcast"(%1203) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1205 = "ttir.add"(%1200, %1204) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1206 = "ttir.sqrt"(%1205) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1207 = "ttir.broadcast"(%1206) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1208 = "ttir.div"(%1202, %1207) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1209 = "ttir.reshape"(%arg108) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1210 = "ttir.broadcast"(%1209) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1211 = "ttir.broadcast"(%1210) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %1212 = "ttir.multiply"(%1208, %1211) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1213 = "ttir.reshape"(%arg109) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1214 = "ttir.broadcast"(%1213) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1215 = "ttir.broadcast"(%1214) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %1216 = "ttir.add"(%1212, %1215) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1217 = "ttir.dot_general"(%1216, %arg104) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %1218 = "ttir.dot_general"(%1216, %arg105) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %1219 = "ttir.dot_general"(%1216, %arg106) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %1220 = "ttir.reshape"(%1217) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %1221 = "ttir.permute"(%1220) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %1222 = "ttir.reshape"(%1218) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %1223 = "ttir.permute"(%1222) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %1224 = "ttir.reshape"(%1219) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %1225 = "ttir.permute"(%1224) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %1226 = "ttir.permute"(%1223) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
        %1227 = "ttir.dot_general"(%1221, %1226) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
        %1228 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %1229 = "ttir.typecast"(%1228) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %1230 = "ttir.reshape"(%1229) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %1231 = "ttir.broadcast"(%1230) <{broadcast_dimensions = array<i64: 8, 12, 197, 197>}> : (tensor<1x1x1x1xf32>) -> tensor<8x12x197x197xf32>
        %1232 = "ttir.div"(%1227, %1231) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1233 = "ttir.max"(%1232) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %1234 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1235 = "ttir.broadcast"(%1234) <{broadcast_dimensions = array<i64: 8, 12, 197>}> : (tensor<1x1x1xf32>) -> tensor<8x12x197xf32>
        %1236 = "ttir.maximum"(%1235, %1233) : (tensor<8x12x197xf32>, tensor<8x12x197xf32>) -> tensor<8x12x197xf32>
        %1237 = "ttir.reshape"(%1236) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %1238 = "ttir.broadcast"(%1237) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %1239 = "ttir.broadcast"(%1238) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %1240 = "ttir.subtract"(%1232, %1239) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1241 = "ttir.exp"(%1240) : (tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1242 = "ttir.sum"(%1241) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %1243 = "ttir.reshape"(%1242) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %1244 = "ttir.broadcast"(%1243) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %1245 = "ttir.broadcast"(%1244) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %1246 = "ttir.div"(%1241, %1245) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1247 = "ttir.dot_general"(%1246, %1225) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
        %1248 = "ttir.permute"(%1247) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
        %1249 = "ttir.reshape"(%1248) <{shape = [8 : i32, 197 : i32, 768 : i32]}> : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
        %1250 = "ttir.dot_general"(%1249, %arg107) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %1251 = "ttir.add"(%1185, %1250) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1252 = "ttir.sum"(%1251) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %1253 = "ttir.reshape"(%1252) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %1254 = "ttir.broadcast"(%1253) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1255 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1256 = "ttir.broadcast"(%1255) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1257 = "ttir.div"(%1254, %1256) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1258 = "ttir.broadcast"(%1257) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1259 = "ttir.subtract"(%1251, %1258) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1260 = "ttir.multiply"(%1259, %1259) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1261 = "ttir.sum"(%1260) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %1262 = "ttir.reshape"(%1261) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %1263 = "ttir.broadcast"(%1262) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1264 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1265 = "ttir.broadcast"(%1264) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1266 = "ttir.div"(%1263, %1265) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1267 = "ttir.broadcast"(%1257) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1268 = "ttir.subtract"(%1251, %1267) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1269 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1270 = "ttir.broadcast"(%1269) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1271 = "ttir.add"(%1266, %1270) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1272 = "ttir.sqrt"(%1271) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1273 = "ttir.broadcast"(%1272) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1274 = "ttir.div"(%1268, %1273) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1275 = "ttir.reshape"(%arg112) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1276 = "ttir.broadcast"(%1275) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1277 = "ttir.broadcast"(%1276) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %1278 = "ttir.multiply"(%1274, %1277) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1279 = "ttir.reshape"(%arg113) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1280 = "ttir.broadcast"(%1279) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1281 = "ttir.broadcast"(%1280) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %1282 = "ttir.add"(%1278, %1281) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1283 = "ttir.dot_general"(%1282, %arg110) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
        %1284 = "ttir.multiply"(%1283, %1283) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1285 = "ttir.multiply"(%1284, %1283) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1286 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1287 = "ttir.broadcast"(%1286) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1288 = "ttir.multiply"(%1287, %1285) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1289 = "ttir.add"(%1283, %1288) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1290 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1291 = "ttir.broadcast"(%1290) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1292 = "ttir.multiply"(%1291, %1289) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1293 = "ttir.tanh"(%1292) : (tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1294 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1295 = "ttir.broadcast"(%1294) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1296 = "ttir.add"(%1295, %1293) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1297 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1298 = "ttir.broadcast"(%1297) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1299 = "ttir.multiply"(%1298, %1296) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1300 = "ttir.multiply"(%1283, %1299) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1301 = "ttir.dot_general"(%1300, %arg111) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
        %1302 = "ttir.add"(%1251, %1301) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1303 = "ttir.sum"(%1302) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %1304 = "ttir.reshape"(%1303) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %1305 = "ttir.broadcast"(%1304) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1306 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1307 = "ttir.broadcast"(%1306) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1308 = "ttir.div"(%1305, %1307) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1309 = "ttir.broadcast"(%1308) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1310 = "ttir.subtract"(%1302, %1309) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1311 = "ttir.multiply"(%1310, %1310) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1312 = "ttir.sum"(%1311) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %1313 = "ttir.reshape"(%1312) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %1314 = "ttir.broadcast"(%1313) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1315 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1316 = "ttir.broadcast"(%1315) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1317 = "ttir.div"(%1314, %1316) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1318 = "ttir.broadcast"(%1308) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1319 = "ttir.subtract"(%1302, %1318) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1320 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1321 = "ttir.broadcast"(%1320) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1322 = "ttir.add"(%1317, %1321) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1323 = "ttir.sqrt"(%1322) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1324 = "ttir.broadcast"(%1323) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1325 = "ttir.div"(%1319, %1324) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1326 = "ttir.reshape"(%arg118) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1327 = "ttir.broadcast"(%1326) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1328 = "ttir.broadcast"(%1327) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %1329 = "ttir.multiply"(%1325, %1328) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1330 = "ttir.reshape"(%arg119) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1331 = "ttir.broadcast"(%1330) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1332 = "ttir.broadcast"(%1331) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %1333 = "ttir.add"(%1329, %1332) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1334 = "ttir.dot_general"(%1333, %arg114) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %1335 = "ttir.dot_general"(%1333, %arg115) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %1336 = "ttir.dot_general"(%1333, %arg116) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %1337 = "ttir.reshape"(%1334) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %1338 = "ttir.permute"(%1337) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %1339 = "ttir.reshape"(%1335) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %1340 = "ttir.permute"(%1339) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %1341 = "ttir.reshape"(%1336) <{shape = [8 : i32, 197 : i32, 12 : i32, 64 : i32]}> : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
        %1342 = "ttir.permute"(%1341) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
        %1343 = "ttir.permute"(%1340) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
        %1344 = "ttir.dot_general"(%1338, %1343) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
        %1345 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %1346 = "ttir.typecast"(%1345) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %1347 = "ttir.reshape"(%1346) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %1348 = "ttir.broadcast"(%1347) <{broadcast_dimensions = array<i64: 8, 12, 197, 197>}> : (tensor<1x1x1x1xf32>) -> tensor<8x12x197x197xf32>
        %1349 = "ttir.div"(%1344, %1348) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1350 = "ttir.max"(%1349) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %1351 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1352 = "ttir.broadcast"(%1351) <{broadcast_dimensions = array<i64: 8, 12, 197>}> : (tensor<1x1x1xf32>) -> tensor<8x12x197xf32>
        %1353 = "ttir.maximum"(%1352, %1350) : (tensor<8x12x197xf32>, tensor<8x12x197xf32>) -> tensor<8x12x197xf32>
        %1354 = "ttir.reshape"(%1353) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %1355 = "ttir.broadcast"(%1354) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %1356 = "ttir.broadcast"(%1355) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %1357 = "ttir.subtract"(%1349, %1356) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1358 = "ttir.exp"(%1357) : (tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1359 = "ttir.sum"(%1358) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32>
        %1360 = "ttir.reshape"(%1359) <{shape = [8 : i32, 12 : i32, 197 : i32, 1 : i32]}> : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
        %1361 = "ttir.broadcast"(%1360) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x1xf32>
        %1362 = "ttir.broadcast"(%1361) <{broadcast_dimensions = array<i64: 1, 1, 1, 197>}> : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
        %1363 = "ttir.div"(%1358, %1362) : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xf32>
        %1364 = "ttir.dot_general"(%1363, %1342) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
        %1365 = "ttir.permute"(%1364) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
        %1366 = "ttir.reshape"(%1365) <{shape = [8 : i32, 197 : i32, 768 : i32]}> : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
        %1367 = "ttir.dot_general"(%1366, %arg117) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
        %1368 = "ttir.add"(%1302, %1367) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1369 = "ttir.sum"(%1368) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %1370 = "ttir.reshape"(%1369) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %1371 = "ttir.broadcast"(%1370) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1372 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1373 = "ttir.broadcast"(%1372) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1374 = "ttir.div"(%1371, %1373) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1375 = "ttir.broadcast"(%1374) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1376 = "ttir.subtract"(%1368, %1375) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1377 = "ttir.multiply"(%1376, %1376) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1378 = "ttir.sum"(%1377) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<8x197x768xf32>) -> tensor<8x197xf32>
        %1379 = "ttir.reshape"(%1378) <{shape = [8 : i32, 197 : i32, 1 : i32]}> : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
        %1380 = "ttir.broadcast"(%1379) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1381 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1382 = "ttir.broadcast"(%1381) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1383 = "ttir.div"(%1380, %1382) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1384 = "ttir.broadcast"(%1374) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1385 = "ttir.subtract"(%1368, %1384) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1386 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1387 = "ttir.broadcast"(%1386) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x1xf32>) -> tensor<8x197x1xf32>
        %1388 = "ttir.add"(%1383, %1387) : (tensor<8x197x1xf32>, tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1389 = "ttir.sqrt"(%1388) : (tensor<8x197x1xf32>) -> tensor<8x197x1xf32>
        %1390 = "ttir.broadcast"(%1389) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
        %1391 = "ttir.div"(%1385, %1390) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1392 = "ttir.reshape"(%arg122) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1393 = "ttir.broadcast"(%1392) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1394 = "ttir.broadcast"(%1393) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %1395 = "ttir.multiply"(%1391, %1394) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1396 = "ttir.reshape"(%arg123) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1397 = "ttir.broadcast"(%1396) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1398 = "ttir.broadcast"(%1397) <{broadcast_dimensions = array<i64: 8, 197, 1>}> : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
        %1399 = "ttir.add"(%1395, %1398) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        %1400 = "ttir.dot_general"(%1399, %arg120) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
        %1401 = "ttir.multiply"(%1400, %1400) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1402 = "ttir.multiply"(%1401, %1400) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1403 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1404 = "ttir.broadcast"(%1403) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1405 = "ttir.multiply"(%1404, %1402) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1406 = "ttir.add"(%1400, %1405) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1407 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1408 = "ttir.broadcast"(%1407) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1409 = "ttir.multiply"(%1408, %1406) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1410 = "ttir.tanh"(%1409) : (tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1411 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1412 = "ttir.broadcast"(%1411) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1413 = "ttir.add"(%1412, %1410) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1414 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1415 = "ttir.broadcast"(%1414) <{broadcast_dimensions = array<i64: 8, 197, 3072>}> : (tensor<1x1x1xf32>) -> tensor<8x197x3072xf32>
        %1416 = "ttir.multiply"(%1415, %1413) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1417 = "ttir.multiply"(%1400, %1416) : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xf32>
        %1418 = "ttir.dot_general"(%1417, %arg121) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
        %1419 = "ttir.add"(%1368, %1418) : (tensor<8x197x768xf32>, tensor<8x197x768xf32>) -> tensor<8x197x768xf32>
        return %1419 : tensor<8x197x768xf32>
      }
    }
  }
}

