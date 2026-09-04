// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
module @jit_vit attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8x3x224x224xf32>, %arg1: tensor<768x768xf32>, %arg2: tensor<8x1x768xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>, %arg6: tensor<768x768xf32>, %arg7: tensor<768x768xf32>, %arg8: tensor<768xf32>, %arg9: tensor<768xf32>, %arg10: tensor<768x3072xf32>, %arg11: tensor<3072x768xf32>, %arg12: tensor<768xf32>, %arg13: tensor<768xf32>, %arg14: tensor<768x768xf32>, %arg15: tensor<768x768xf32>, %arg16: tensor<768x768xf32>, %arg17: tensor<768x768xf32>, %arg18: tensor<768xf32>, %arg19: tensor<768xf32>, %arg20: tensor<768x3072xf32>, %arg21: tensor<3072x768xf32>, %arg22: tensor<768xf32>, %arg23: tensor<768xf32>, %arg24: tensor<768x768xf32>, %arg25: tensor<768x768xf32>, %arg26: tensor<768x768xf32>, %arg27: tensor<768x768xf32>, %arg28: tensor<768xf32>, %arg29: tensor<768xf32>, %arg30: tensor<768x3072xf32>, %arg31: tensor<3072x768xf32>, %arg32: tensor<768xf32>, %arg33: tensor<768xf32>, %arg34: tensor<768x768xf32>, %arg35: tensor<768x768xf32>, %arg36: tensor<768x768xf32>, %arg37: tensor<768x768xf32>, %arg38: tensor<768xf32>, %arg39: tensor<768xf32>, %arg40: tensor<768x3072xf32>, %arg41: tensor<3072x768xf32>, %arg42: tensor<768xf32>, %arg43: tensor<768xf32>, %arg44: tensor<768x768xf32>, %arg45: tensor<768x768xf32>, %arg46: tensor<768x768xf32>, %arg47: tensor<768x768xf32>, %arg48: tensor<768xf32>, %arg49: tensor<768xf32>, %arg50: tensor<768x3072xf32>, %arg51: tensor<3072x768xf32>, %arg52: tensor<768xf32>, %arg53: tensor<768xf32>, %arg54: tensor<768x768xf32>, %arg55: tensor<768x768xf32>, %arg56: tensor<768x768xf32>, %arg57: tensor<768x768xf32>, %arg58: tensor<768xf32>, %arg59: tensor<768xf32>, %arg60: tensor<768x3072xf32>, %arg61: tensor<3072x768xf32>, %arg62: tensor<768xf32>, %arg63: tensor<768xf32>, %arg64: tensor<768x768xf32>, %arg65: tensor<768x768xf32>, %arg66: tensor<768x768xf32>, %arg67: tensor<768x768xf32>, %arg68: tensor<768xf32>, %arg69: tensor<768xf32>, %arg70: tensor<768x3072xf32>, %arg71: tensor<3072x768xf32>, %arg72: tensor<768xf32>, %arg73: tensor<768xf32>, %arg74: tensor<768x768xf32>, %arg75: tensor<768x768xf32>, %arg76: tensor<768x768xf32>, %arg77: tensor<768x768xf32>, %arg78: tensor<768xf32>, %arg79: tensor<768xf32>, %arg80: tensor<768x3072xf32>, %arg81: tensor<3072x768xf32>, %arg82: tensor<768xf32>, %arg83: tensor<768xf32>, %arg84: tensor<768x768xf32>, %arg85: tensor<768x768xf32>, %arg86: tensor<768x768xf32>, %arg87: tensor<768x768xf32>, %arg88: tensor<768xf32>, %arg89: tensor<768xf32>, %arg90: tensor<768x3072xf32>, %arg91: tensor<3072x768xf32>, %arg92: tensor<768xf32>, %arg93: tensor<768xf32>, %arg94: tensor<768x768xf32>, %arg95: tensor<768x768xf32>, %arg96: tensor<768x768xf32>, %arg97: tensor<768x768xf32>, %arg98: tensor<768xf32>, %arg99: tensor<768xf32>, %arg100: tensor<768x3072xf32>, %arg101: tensor<3072x768xf32>, %arg102: tensor<768xf32>, %arg103: tensor<768xf32>, %arg104: tensor<768x768xf32>, %arg105: tensor<768x768xf32>, %arg106: tensor<768x768xf32>, %arg107: tensor<768x768xf32>, %arg108: tensor<768xf32>, %arg109: tensor<768xf32>, %arg110: tensor<768x3072xf32>, %arg111: tensor<3072x768xf32>, %arg112: tensor<768xf32>, %arg113: tensor<768xf32>, %arg114: tensor<768x768xf32>, %arg115: tensor<768x768xf32>, %arg116: tensor<768x768xf32>, %arg117: tensor<768x768xf32>, %arg118: tensor<768xf32>, %arg119: tensor<768xf32>, %arg120: tensor<768x3072xf32>, %arg121: tensor<3072x768xf32>, %arg122: tensor<768xf32>, %arg123: tensor<768xf32>) -> (tensor<8x197x768xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<8x3x224x224xf32>) -> tensor<8x3x14x16x14x16xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 4, 1, 3, 5] : (tensor<8x3x14x16x14x16xf32>) -> tensor<8x14x14x3x16x16xf32>
    %2 = stablehlo.reshape %1 : (tensor<8x14x14x3x16x16xf32>) -> tensor<8x196x768xf32>
    %3 = stablehlo.dot_general %2, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x196x768xf32>, tensor<768x768xf32>) -> tensor<8x196x768xf32>
    %4 = stablehlo.concatenate %arg2, %3, dim = 1 : (tensor<8x1x768xf32>, tensor<8x196x768xf32>) -> tensor<8x197x768xf32>
    %5 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1, 2] : (tensor<1x197x768xf32>) -> tensor<8x197x768xf32>
    %6 = stablehlo.add %4, %5 : tensor<8x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %cst_0 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %10 = stablehlo.divide %8, %9 : tensor<8x197x1xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %12 = stablehlo.subtract %6, %11 : tensor<8x197x768xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<8x197x768xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = stablehlo.reduce(%13 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %17 = stablehlo.divide %15, %16 : tensor<8x197x1xf32>
    %18 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %19 = stablehlo.subtract %6, %18 : tensor<8x197x768xf32>
    %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %21 = stablehlo.add %17, %20 : tensor<8x197x1xf32>
    %22 = stablehlo.sqrt %21 : tensor<8x197x1xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %24 = stablehlo.divide %19, %23 : tensor<8x197x768xf32>
    %25 = stablehlo.broadcast_in_dim %arg8, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %27 = stablehlo.multiply %24, %26 : tensor<8x197x768xf32>
    %28 = stablehlo.broadcast_in_dim %arg9, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %30 = stablehlo.add %27, %29 : tensor<8x197x768xf32>
    %31 = stablehlo.dot_general %30, %arg4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %32 = stablehlo.dot_general %30, %arg5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %33 = stablehlo.dot_general %30, %arg6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %34 = stablehlo.reshape %31 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %35 = stablehlo.transpose %34, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %36 = stablehlo.reshape %32 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %37 = stablehlo.transpose %36, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %38 = stablehlo.reshape %33 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %39 = stablehlo.transpose %38, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %40 = stablehlo.transpose %37, dims = [0, 1, 3, 2] : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
    %41 = stablehlo.dot_general %35, %40, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
    %cst_3 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %42 = stablehlo.sqrt %cst_3 : tensor<f32>
    %43 = stablehlo.convert %42 : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
    %45 = stablehlo.divide %41, %44 : tensor<8x12x197x197xf32>
    %cst_4 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %46 = stablehlo.reduce(%45 init: %cst_4) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %cst_5 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
    %48 = stablehlo.maximum %47, %46 : tensor<8x12x197xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %50 = stablehlo.broadcast_in_dim %49, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %51 = stablehlo.subtract %45, %50 : tensor<8x12x197x197xf32>
    %52 = stablehlo.exponential %51 : tensor<8x12x197x197xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %53 = stablehlo.reduce(%52 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %56 = stablehlo.divide %52, %55 : tensor<8x12x197x197xf32>
    %57 = stablehlo.dot_general %56, %39, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
    %58 = stablehlo.transpose %57, dims = [0, 2, 1, 3] : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
    %59 = stablehlo.reshape %58 : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
    %60 = stablehlo.dot_general %59, %arg7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %61 = stablehlo.add %6, %60 : tensor<8x197x768xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %62 = stablehlo.reduce(%61 init: %cst_7) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %63 = stablehlo.broadcast_in_dim %62, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %64 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %65 = stablehlo.divide %63, %64 : tensor<8x197x1xf32>
    %66 = stablehlo.broadcast_in_dim %65, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %67 = stablehlo.subtract %61, %66 : tensor<8x197x768xf32>
    %68 = stablehlo.multiply %67, %67 : tensor<8x197x768xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %69 = stablehlo.reduce(%68 init: %cst_8) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %71 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %72 = stablehlo.divide %70, %71 : tensor<8x197x1xf32>
    %73 = stablehlo.broadcast_in_dim %65, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %74 = stablehlo.subtract %61, %73 : tensor<8x197x768xf32>
    %75 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %76 = stablehlo.add %72, %75 : tensor<8x197x1xf32>
    %77 = stablehlo.sqrt %76 : tensor<8x197x1xf32>
    %78 = stablehlo.broadcast_in_dim %77, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %79 = stablehlo.divide %74, %78 : tensor<8x197x768xf32>
    %80 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %82 = stablehlo.multiply %79, %81 : tensor<8x197x768xf32>
    %83 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %85 = stablehlo.add %82, %84 : tensor<8x197x768xf32>
    %86 = stablehlo.dot_general %85, %arg10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
    %87 = stablehlo.multiply %86, %86 : tensor<8x197x3072xf32>
    %88 = stablehlo.multiply %87, %86 : tensor<8x197x3072xf32>
    %cst_9 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %89 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %90 = stablehlo.multiply %89, %88 : tensor<8x197x3072xf32>
    %91 = stablehlo.add %86, %90 : tensor<8x197x3072xf32>
    %cst_10 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %92 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %93 = stablehlo.multiply %92, %91 : tensor<8x197x3072xf32>
    %94 = stablehlo.tanh %93 : tensor<8x197x3072xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %95 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %96 = stablehlo.add %95, %94 : tensor<8x197x3072xf32>
    %cst_12 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %97 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %98 = stablehlo.multiply %97, %96 : tensor<8x197x3072xf32>
    %99 = stablehlo.multiply %86, %98 : tensor<8x197x3072xf32>
    %100 = stablehlo.dot_general %99, %arg11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
    %101 = stablehlo.add %61, %100 : tensor<8x197x768xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %102 = stablehlo.reduce(%101 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %103 = stablehlo.broadcast_in_dim %102, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %104 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %105 = stablehlo.divide %103, %104 : tensor<8x197x1xf32>
    %106 = stablehlo.broadcast_in_dim %105, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %107 = stablehlo.subtract %101, %106 : tensor<8x197x768xf32>
    %108 = stablehlo.multiply %107, %107 : tensor<8x197x768xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %109 = stablehlo.reduce(%108 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %110 = stablehlo.broadcast_in_dim %109, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %111 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %112 = stablehlo.divide %110, %111 : tensor<8x197x1xf32>
    %113 = stablehlo.broadcast_in_dim %105, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %114 = stablehlo.subtract %101, %113 : tensor<8x197x768xf32>
    %115 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %116 = stablehlo.add %112, %115 : tensor<8x197x1xf32>
    %117 = stablehlo.sqrt %116 : tensor<8x197x1xf32>
    %118 = stablehlo.broadcast_in_dim %117, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %119 = stablehlo.divide %114, %118 : tensor<8x197x768xf32>
    %120 = stablehlo.broadcast_in_dim %arg18, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %121 = stablehlo.broadcast_in_dim %120, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %122 = stablehlo.multiply %119, %121 : tensor<8x197x768xf32>
    %123 = stablehlo.broadcast_in_dim %arg19, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %124 = stablehlo.broadcast_in_dim %123, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %125 = stablehlo.add %122, %124 : tensor<8x197x768xf32>
    %126 = stablehlo.dot_general %125, %arg14, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %127 = stablehlo.dot_general %125, %arg15, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %128 = stablehlo.dot_general %125, %arg16, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %129 = stablehlo.reshape %126 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %130 = stablehlo.transpose %129, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %131 = stablehlo.reshape %127 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %132 = stablehlo.transpose %131, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %133 = stablehlo.reshape %128 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %134 = stablehlo.transpose %133, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %135 = stablehlo.transpose %132, dims = [0, 1, 3, 2] : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
    %136 = stablehlo.dot_general %130, %135, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
    %137 = stablehlo.sqrt %cst_3 : tensor<f32>
    %138 = stablehlo.convert %137 : tensor<f32>
    %139 = stablehlo.broadcast_in_dim %138, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
    %140 = stablehlo.divide %136, %139 : tensor<8x12x197x197xf32>
    %cst_15 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %141 = stablehlo.reduce(%140 init: %cst_15) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %142 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
    %143 = stablehlo.maximum %142, %141 : tensor<8x12x197xf32>
    %144 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %145 = stablehlo.broadcast_in_dim %144, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %146 = stablehlo.subtract %140, %145 : tensor<8x12x197x197xf32>
    %147 = stablehlo.exponential %146 : tensor<8x12x197x197xf32>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %148 = stablehlo.reduce(%147 init: %cst_16) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %150 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %151 = stablehlo.divide %147, %150 : tensor<8x12x197x197xf32>
    %152 = stablehlo.dot_general %151, %134, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
    %153 = stablehlo.transpose %152, dims = [0, 2, 1, 3] : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
    %154 = stablehlo.reshape %153 : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
    %155 = stablehlo.dot_general %154, %arg17, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %156 = stablehlo.add %101, %155 : tensor<8x197x768xf32>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %157 = stablehlo.reduce(%156 init: %cst_17) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %158 = stablehlo.broadcast_in_dim %157, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %159 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %160 = stablehlo.divide %158, %159 : tensor<8x197x1xf32>
    %161 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %162 = stablehlo.subtract %156, %161 : tensor<8x197x768xf32>
    %163 = stablehlo.multiply %162, %162 : tensor<8x197x768xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %164 = stablehlo.reduce(%163 init: %cst_18) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %165 = stablehlo.broadcast_in_dim %164, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %166 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %167 = stablehlo.divide %165, %166 : tensor<8x197x1xf32>
    %168 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %169 = stablehlo.subtract %156, %168 : tensor<8x197x768xf32>
    %170 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %171 = stablehlo.add %167, %170 : tensor<8x197x1xf32>
    %172 = stablehlo.sqrt %171 : tensor<8x197x1xf32>
    %173 = stablehlo.broadcast_in_dim %172, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %174 = stablehlo.divide %169, %173 : tensor<8x197x768xf32>
    %175 = stablehlo.broadcast_in_dim %arg22, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %176 = stablehlo.broadcast_in_dim %175, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %177 = stablehlo.multiply %174, %176 : tensor<8x197x768xf32>
    %178 = stablehlo.broadcast_in_dim %arg23, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %179 = stablehlo.broadcast_in_dim %178, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %180 = stablehlo.add %177, %179 : tensor<8x197x768xf32>
    %181 = stablehlo.dot_general %180, %arg20, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
    %182 = stablehlo.multiply %181, %181 : tensor<8x197x3072xf32>
    %183 = stablehlo.multiply %182, %181 : tensor<8x197x3072xf32>
    %184 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %185 = stablehlo.multiply %184, %183 : tensor<8x197x3072xf32>
    %186 = stablehlo.add %181, %185 : tensor<8x197x3072xf32>
    %187 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %188 = stablehlo.multiply %187, %186 : tensor<8x197x3072xf32>
    %189 = stablehlo.tanh %188 : tensor<8x197x3072xf32>
    %190 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %191 = stablehlo.add %190, %189 : tensor<8x197x3072xf32>
    %192 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %193 = stablehlo.multiply %192, %191 : tensor<8x197x3072xf32>
    %194 = stablehlo.multiply %181, %193 : tensor<8x197x3072xf32>
    %195 = stablehlo.dot_general %194, %arg21, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
    %196 = stablehlo.add %156, %195 : tensor<8x197x768xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %197 = stablehlo.reduce(%196 init: %cst_19) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %199 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %200 = stablehlo.divide %198, %199 : tensor<8x197x1xf32>
    %201 = stablehlo.broadcast_in_dim %200, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %202 = stablehlo.subtract %196, %201 : tensor<8x197x768xf32>
    %203 = stablehlo.multiply %202, %202 : tensor<8x197x768xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %204 = stablehlo.reduce(%203 init: %cst_20) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %205 = stablehlo.broadcast_in_dim %204, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %206 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %207 = stablehlo.divide %205, %206 : tensor<8x197x1xf32>
    %208 = stablehlo.broadcast_in_dim %200, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %209 = stablehlo.subtract %196, %208 : tensor<8x197x768xf32>
    %210 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %211 = stablehlo.add %207, %210 : tensor<8x197x1xf32>
    %212 = stablehlo.sqrt %211 : tensor<8x197x1xf32>
    %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %214 = stablehlo.divide %209, %213 : tensor<8x197x768xf32>
    %215 = stablehlo.broadcast_in_dim %arg28, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %216 = stablehlo.broadcast_in_dim %215, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %217 = stablehlo.multiply %214, %216 : tensor<8x197x768xf32>
    %218 = stablehlo.broadcast_in_dim %arg29, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %219 = stablehlo.broadcast_in_dim %218, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %220 = stablehlo.add %217, %219 : tensor<8x197x768xf32>
    %221 = stablehlo.dot_general %220, %arg24, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %222 = stablehlo.dot_general %220, %arg25, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %223 = stablehlo.dot_general %220, %arg26, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %224 = stablehlo.reshape %221 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %225 = stablehlo.transpose %224, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %226 = stablehlo.reshape %222 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %227 = stablehlo.transpose %226, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %228 = stablehlo.reshape %223 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %229 = stablehlo.transpose %228, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %230 = stablehlo.transpose %227, dims = [0, 1, 3, 2] : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
    %231 = stablehlo.dot_general %225, %230, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
    %232 = stablehlo.sqrt %cst_3 : tensor<f32>
    %233 = stablehlo.convert %232 : tensor<f32>
    %234 = stablehlo.broadcast_in_dim %233, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
    %235 = stablehlo.divide %231, %234 : tensor<8x12x197x197xf32>
    %cst_21 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %236 = stablehlo.reduce(%235 init: %cst_21) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %237 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
    %238 = stablehlo.maximum %237, %236 : tensor<8x12x197xf32>
    %239 = stablehlo.broadcast_in_dim %238, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %240 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %241 = stablehlo.subtract %235, %240 : tensor<8x12x197x197xf32>
    %242 = stablehlo.exponential %241 : tensor<8x12x197x197xf32>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %243 = stablehlo.reduce(%242 init: %cst_22) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %244 = stablehlo.broadcast_in_dim %243, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %245 = stablehlo.broadcast_in_dim %244, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %246 = stablehlo.divide %242, %245 : tensor<8x12x197x197xf32>
    %247 = stablehlo.dot_general %246, %229, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
    %248 = stablehlo.transpose %247, dims = [0, 2, 1, 3] : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
    %249 = stablehlo.reshape %248 : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
    %250 = stablehlo.dot_general %249, %arg27, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %251 = stablehlo.add %196, %250 : tensor<8x197x768xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %252 = stablehlo.reduce(%251 init: %cst_23) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %253 = stablehlo.broadcast_in_dim %252, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %254 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %255 = stablehlo.divide %253, %254 : tensor<8x197x1xf32>
    %256 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %257 = stablehlo.subtract %251, %256 : tensor<8x197x768xf32>
    %258 = stablehlo.multiply %257, %257 : tensor<8x197x768xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %259 = stablehlo.reduce(%258 init: %cst_24) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %260 = stablehlo.broadcast_in_dim %259, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %261 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %262 = stablehlo.divide %260, %261 : tensor<8x197x1xf32>
    %263 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %264 = stablehlo.subtract %251, %263 : tensor<8x197x768xf32>
    %265 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %266 = stablehlo.add %262, %265 : tensor<8x197x1xf32>
    %267 = stablehlo.sqrt %266 : tensor<8x197x1xf32>
    %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %269 = stablehlo.divide %264, %268 : tensor<8x197x768xf32>
    %270 = stablehlo.broadcast_in_dim %arg32, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %271 = stablehlo.broadcast_in_dim %270, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %272 = stablehlo.multiply %269, %271 : tensor<8x197x768xf32>
    %273 = stablehlo.broadcast_in_dim %arg33, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %274 = stablehlo.broadcast_in_dim %273, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %275 = stablehlo.add %272, %274 : tensor<8x197x768xf32>
    %276 = stablehlo.dot_general %275, %arg30, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
    %277 = stablehlo.multiply %276, %276 : tensor<8x197x3072xf32>
    %278 = stablehlo.multiply %277, %276 : tensor<8x197x3072xf32>
    %279 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %280 = stablehlo.multiply %279, %278 : tensor<8x197x3072xf32>
    %281 = stablehlo.add %276, %280 : tensor<8x197x3072xf32>
    %282 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %283 = stablehlo.multiply %282, %281 : tensor<8x197x3072xf32>
    %284 = stablehlo.tanh %283 : tensor<8x197x3072xf32>
    %285 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %286 = stablehlo.add %285, %284 : tensor<8x197x3072xf32>
    %287 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %288 = stablehlo.multiply %287, %286 : tensor<8x197x3072xf32>
    %289 = stablehlo.multiply %276, %288 : tensor<8x197x3072xf32>
    %290 = stablehlo.dot_general %289, %arg31, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
    %291 = stablehlo.add %251, %290 : tensor<8x197x768xf32>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %292 = stablehlo.reduce(%291 init: %cst_25) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %294 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %295 = stablehlo.divide %293, %294 : tensor<8x197x1xf32>
    %296 = stablehlo.broadcast_in_dim %295, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %297 = stablehlo.subtract %291, %296 : tensor<8x197x768xf32>
    %298 = stablehlo.multiply %297, %297 : tensor<8x197x768xf32>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %299 = stablehlo.reduce(%298 init: %cst_26) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %301 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %302 = stablehlo.divide %300, %301 : tensor<8x197x1xf32>
    %303 = stablehlo.broadcast_in_dim %295, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %304 = stablehlo.subtract %291, %303 : tensor<8x197x768xf32>
    %305 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %306 = stablehlo.add %302, %305 : tensor<8x197x1xf32>
    %307 = stablehlo.sqrt %306 : tensor<8x197x1xf32>
    %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %309 = stablehlo.divide %304, %308 : tensor<8x197x768xf32>
    %310 = stablehlo.broadcast_in_dim %arg38, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %311 = stablehlo.broadcast_in_dim %310, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %312 = stablehlo.multiply %309, %311 : tensor<8x197x768xf32>
    %313 = stablehlo.broadcast_in_dim %arg39, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %315 = stablehlo.add %312, %314 : tensor<8x197x768xf32>
    %316 = stablehlo.dot_general %315, %arg34, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %317 = stablehlo.dot_general %315, %arg35, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %318 = stablehlo.dot_general %315, %arg36, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %319 = stablehlo.reshape %316 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %320 = stablehlo.transpose %319, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %321 = stablehlo.reshape %317 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %322 = stablehlo.transpose %321, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %323 = stablehlo.reshape %318 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %324 = stablehlo.transpose %323, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %325 = stablehlo.transpose %322, dims = [0, 1, 3, 2] : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
    %326 = stablehlo.dot_general %320, %325, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
    %327 = stablehlo.sqrt %cst_3 : tensor<f32>
    %328 = stablehlo.convert %327 : tensor<f32>
    %329 = stablehlo.broadcast_in_dim %328, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
    %330 = stablehlo.divide %326, %329 : tensor<8x12x197x197xf32>
    %cst_27 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %331 = stablehlo.reduce(%330 init: %cst_27) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %332 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
    %333 = stablehlo.maximum %332, %331 : tensor<8x12x197xf32>
    %334 = stablehlo.broadcast_in_dim %333, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %335 = stablehlo.broadcast_in_dim %334, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %336 = stablehlo.subtract %330, %335 : tensor<8x12x197x197xf32>
    %337 = stablehlo.exponential %336 : tensor<8x12x197x197xf32>
    %cst_28 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %338 = stablehlo.reduce(%337 init: %cst_28) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %339 = stablehlo.broadcast_in_dim %338, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %340 = stablehlo.broadcast_in_dim %339, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %341 = stablehlo.divide %337, %340 : tensor<8x12x197x197xf32>
    %342 = stablehlo.dot_general %341, %324, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
    %343 = stablehlo.transpose %342, dims = [0, 2, 1, 3] : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
    %344 = stablehlo.reshape %343 : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
    %345 = stablehlo.dot_general %344, %arg37, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %346 = stablehlo.add %291, %345 : tensor<8x197x768xf32>
    %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %347 = stablehlo.reduce(%346 init: %cst_29) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %348 = stablehlo.broadcast_in_dim %347, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %350 = stablehlo.divide %348, %349 : tensor<8x197x1xf32>
    %351 = stablehlo.broadcast_in_dim %350, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %352 = stablehlo.subtract %346, %351 : tensor<8x197x768xf32>
    %353 = stablehlo.multiply %352, %352 : tensor<8x197x768xf32>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %354 = stablehlo.reduce(%353 init: %cst_30) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %355 = stablehlo.broadcast_in_dim %354, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %356 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %357 = stablehlo.divide %355, %356 : tensor<8x197x1xf32>
    %358 = stablehlo.broadcast_in_dim %350, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %359 = stablehlo.subtract %346, %358 : tensor<8x197x768xf32>
    %360 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %361 = stablehlo.add %357, %360 : tensor<8x197x1xf32>
    %362 = stablehlo.sqrt %361 : tensor<8x197x1xf32>
    %363 = stablehlo.broadcast_in_dim %362, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %364 = stablehlo.divide %359, %363 : tensor<8x197x768xf32>
    %365 = stablehlo.broadcast_in_dim %arg42, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %366 = stablehlo.broadcast_in_dim %365, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %367 = stablehlo.multiply %364, %366 : tensor<8x197x768xf32>
    %368 = stablehlo.broadcast_in_dim %arg43, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %369 = stablehlo.broadcast_in_dim %368, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %370 = stablehlo.add %367, %369 : tensor<8x197x768xf32>
    %371 = stablehlo.dot_general %370, %arg40, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
    %372 = stablehlo.multiply %371, %371 : tensor<8x197x3072xf32>
    %373 = stablehlo.multiply %372, %371 : tensor<8x197x3072xf32>
    %374 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %375 = stablehlo.multiply %374, %373 : tensor<8x197x3072xf32>
    %376 = stablehlo.add %371, %375 : tensor<8x197x3072xf32>
    %377 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %378 = stablehlo.multiply %377, %376 : tensor<8x197x3072xf32>
    %379 = stablehlo.tanh %378 : tensor<8x197x3072xf32>
    %380 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %381 = stablehlo.add %380, %379 : tensor<8x197x3072xf32>
    %382 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %383 = stablehlo.multiply %382, %381 : tensor<8x197x3072xf32>
    %384 = stablehlo.multiply %371, %383 : tensor<8x197x3072xf32>
    %385 = stablehlo.dot_general %384, %arg41, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
    %386 = stablehlo.add %346, %385 : tensor<8x197x768xf32>
    %cst_31 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %387 = stablehlo.reduce(%386 init: %cst_31) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %388 = stablehlo.broadcast_in_dim %387, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %389 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %390 = stablehlo.divide %388, %389 : tensor<8x197x1xf32>
    %391 = stablehlo.broadcast_in_dim %390, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %392 = stablehlo.subtract %386, %391 : tensor<8x197x768xf32>
    %393 = stablehlo.multiply %392, %392 : tensor<8x197x768xf32>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %394 = stablehlo.reduce(%393 init: %cst_32) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %395 = stablehlo.broadcast_in_dim %394, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %397 = stablehlo.divide %395, %396 : tensor<8x197x1xf32>
    %398 = stablehlo.broadcast_in_dim %390, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %399 = stablehlo.subtract %386, %398 : tensor<8x197x768xf32>
    %400 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %401 = stablehlo.add %397, %400 : tensor<8x197x1xf32>
    %402 = stablehlo.sqrt %401 : tensor<8x197x1xf32>
    %403 = stablehlo.broadcast_in_dim %402, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %404 = stablehlo.divide %399, %403 : tensor<8x197x768xf32>
    %405 = stablehlo.broadcast_in_dim %arg48, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %406 = stablehlo.broadcast_in_dim %405, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %407 = stablehlo.multiply %404, %406 : tensor<8x197x768xf32>
    %408 = stablehlo.broadcast_in_dim %arg49, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %410 = stablehlo.add %407, %409 : tensor<8x197x768xf32>
    %411 = stablehlo.dot_general %410, %arg44, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %412 = stablehlo.dot_general %410, %arg45, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %413 = stablehlo.dot_general %410, %arg46, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %414 = stablehlo.reshape %411 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %415 = stablehlo.transpose %414, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %416 = stablehlo.reshape %412 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %417 = stablehlo.transpose %416, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %418 = stablehlo.reshape %413 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %419 = stablehlo.transpose %418, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %420 = stablehlo.transpose %417, dims = [0, 1, 3, 2] : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
    %421 = stablehlo.dot_general %415, %420, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
    %422 = stablehlo.sqrt %cst_3 : tensor<f32>
    %423 = stablehlo.convert %422 : tensor<f32>
    %424 = stablehlo.broadcast_in_dim %423, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
    %425 = stablehlo.divide %421, %424 : tensor<8x12x197x197xf32>
    %cst_33 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %426 = stablehlo.reduce(%425 init: %cst_33) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %427 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
    %428 = stablehlo.maximum %427, %426 : tensor<8x12x197xf32>
    %429 = stablehlo.broadcast_in_dim %428, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %430 = stablehlo.broadcast_in_dim %429, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %431 = stablehlo.subtract %425, %430 : tensor<8x12x197x197xf32>
    %432 = stablehlo.exponential %431 : tensor<8x12x197x197xf32>
    %cst_34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %433 = stablehlo.reduce(%432 init: %cst_34) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %434 = stablehlo.broadcast_in_dim %433, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %436 = stablehlo.divide %432, %435 : tensor<8x12x197x197xf32>
    %437 = stablehlo.dot_general %436, %419, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
    %438 = stablehlo.transpose %437, dims = [0, 2, 1, 3] : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
    %439 = stablehlo.reshape %438 : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
    %440 = stablehlo.dot_general %439, %arg47, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %441 = stablehlo.add %386, %440 : tensor<8x197x768xf32>
    %cst_35 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %442 = stablehlo.reduce(%441 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %443 = stablehlo.broadcast_in_dim %442, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %444 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %445 = stablehlo.divide %443, %444 : tensor<8x197x1xf32>
    %446 = stablehlo.broadcast_in_dim %445, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %447 = stablehlo.subtract %441, %446 : tensor<8x197x768xf32>
    %448 = stablehlo.multiply %447, %447 : tensor<8x197x768xf32>
    %cst_36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %449 = stablehlo.reduce(%448 init: %cst_36) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %450 = stablehlo.broadcast_in_dim %449, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %451 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %452 = stablehlo.divide %450, %451 : tensor<8x197x1xf32>
    %453 = stablehlo.broadcast_in_dim %445, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %454 = stablehlo.subtract %441, %453 : tensor<8x197x768xf32>
    %455 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %456 = stablehlo.add %452, %455 : tensor<8x197x1xf32>
    %457 = stablehlo.sqrt %456 : tensor<8x197x1xf32>
    %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %459 = stablehlo.divide %454, %458 : tensor<8x197x768xf32>
    %460 = stablehlo.broadcast_in_dim %arg52, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %461 = stablehlo.broadcast_in_dim %460, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %462 = stablehlo.multiply %459, %461 : tensor<8x197x768xf32>
    %463 = stablehlo.broadcast_in_dim %arg53, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %465 = stablehlo.add %462, %464 : tensor<8x197x768xf32>
    %466 = stablehlo.dot_general %465, %arg50, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
    %467 = stablehlo.multiply %466, %466 : tensor<8x197x3072xf32>
    %468 = stablehlo.multiply %467, %466 : tensor<8x197x3072xf32>
    %469 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %470 = stablehlo.multiply %469, %468 : tensor<8x197x3072xf32>
    %471 = stablehlo.add %466, %470 : tensor<8x197x3072xf32>
    %472 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %473 = stablehlo.multiply %472, %471 : tensor<8x197x3072xf32>
    %474 = stablehlo.tanh %473 : tensor<8x197x3072xf32>
    %475 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %476 = stablehlo.add %475, %474 : tensor<8x197x3072xf32>
    %477 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %478 = stablehlo.multiply %477, %476 : tensor<8x197x3072xf32>
    %479 = stablehlo.multiply %466, %478 : tensor<8x197x3072xf32>
    %480 = stablehlo.dot_general %479, %arg51, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
    %481 = stablehlo.add %441, %480 : tensor<8x197x768xf32>
    %cst_37 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %482 = stablehlo.reduce(%481 init: %cst_37) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %483 = stablehlo.broadcast_in_dim %482, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %484 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %485 = stablehlo.divide %483, %484 : tensor<8x197x1xf32>
    %486 = stablehlo.broadcast_in_dim %485, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %487 = stablehlo.subtract %481, %486 : tensor<8x197x768xf32>
    %488 = stablehlo.multiply %487, %487 : tensor<8x197x768xf32>
    %cst_38 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %489 = stablehlo.reduce(%488 init: %cst_38) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %490 = stablehlo.broadcast_in_dim %489, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %491 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %492 = stablehlo.divide %490, %491 : tensor<8x197x1xf32>
    %493 = stablehlo.broadcast_in_dim %485, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %494 = stablehlo.subtract %481, %493 : tensor<8x197x768xf32>
    %495 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %496 = stablehlo.add %492, %495 : tensor<8x197x1xf32>
    %497 = stablehlo.sqrt %496 : tensor<8x197x1xf32>
    %498 = stablehlo.broadcast_in_dim %497, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %499 = stablehlo.divide %494, %498 : tensor<8x197x768xf32>
    %500 = stablehlo.broadcast_in_dim %arg58, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %501 = stablehlo.broadcast_in_dim %500, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %502 = stablehlo.multiply %499, %501 : tensor<8x197x768xf32>
    %503 = stablehlo.broadcast_in_dim %arg59, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %504 = stablehlo.broadcast_in_dim %503, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %505 = stablehlo.add %502, %504 : tensor<8x197x768xf32>
    %506 = stablehlo.dot_general %505, %arg54, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %507 = stablehlo.dot_general %505, %arg55, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %508 = stablehlo.dot_general %505, %arg56, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %509 = stablehlo.reshape %506 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %510 = stablehlo.transpose %509, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %511 = stablehlo.reshape %507 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %512 = stablehlo.transpose %511, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %513 = stablehlo.reshape %508 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %514 = stablehlo.transpose %513, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %515 = stablehlo.transpose %512, dims = [0, 1, 3, 2] : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
    %516 = stablehlo.dot_general %510, %515, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
    %517 = stablehlo.sqrt %cst_3 : tensor<f32>
    %518 = stablehlo.convert %517 : tensor<f32>
    %519 = stablehlo.broadcast_in_dim %518, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
    %520 = stablehlo.divide %516, %519 : tensor<8x12x197x197xf32>
    %cst_39 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %521 = stablehlo.reduce(%520 init: %cst_39) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %522 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
    %523 = stablehlo.maximum %522, %521 : tensor<8x12x197xf32>
    %524 = stablehlo.broadcast_in_dim %523, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %525 = stablehlo.broadcast_in_dim %524, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %526 = stablehlo.subtract %520, %525 : tensor<8x12x197x197xf32>
    %527 = stablehlo.exponential %526 : tensor<8x12x197x197xf32>
    %cst_40 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %528 = stablehlo.reduce(%527 init: %cst_40) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %529 = stablehlo.broadcast_in_dim %528, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %530 = stablehlo.broadcast_in_dim %529, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %531 = stablehlo.divide %527, %530 : tensor<8x12x197x197xf32>
    %532 = stablehlo.dot_general %531, %514, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
    %533 = stablehlo.transpose %532, dims = [0, 2, 1, 3] : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
    %534 = stablehlo.reshape %533 : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
    %535 = stablehlo.dot_general %534, %arg57, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %536 = stablehlo.add %481, %535 : tensor<8x197x768xf32>
    %cst_41 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %537 = stablehlo.reduce(%536 init: %cst_41) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %538 = stablehlo.broadcast_in_dim %537, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %539 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %540 = stablehlo.divide %538, %539 : tensor<8x197x1xf32>
    %541 = stablehlo.broadcast_in_dim %540, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %542 = stablehlo.subtract %536, %541 : tensor<8x197x768xf32>
    %543 = stablehlo.multiply %542, %542 : tensor<8x197x768xf32>
    %cst_42 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %544 = stablehlo.reduce(%543 init: %cst_42) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %545 = stablehlo.broadcast_in_dim %544, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %546 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %547 = stablehlo.divide %545, %546 : tensor<8x197x1xf32>
    %548 = stablehlo.broadcast_in_dim %540, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %549 = stablehlo.subtract %536, %548 : tensor<8x197x768xf32>
    %550 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %551 = stablehlo.add %547, %550 : tensor<8x197x1xf32>
    %552 = stablehlo.sqrt %551 : tensor<8x197x1xf32>
    %553 = stablehlo.broadcast_in_dim %552, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %554 = stablehlo.divide %549, %553 : tensor<8x197x768xf32>
    %555 = stablehlo.broadcast_in_dim %arg62, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %556 = stablehlo.broadcast_in_dim %555, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %557 = stablehlo.multiply %554, %556 : tensor<8x197x768xf32>
    %558 = stablehlo.broadcast_in_dim %arg63, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %559 = stablehlo.broadcast_in_dim %558, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %560 = stablehlo.add %557, %559 : tensor<8x197x768xf32>
    %561 = stablehlo.dot_general %560, %arg60, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
    %562 = stablehlo.multiply %561, %561 : tensor<8x197x3072xf32>
    %563 = stablehlo.multiply %562, %561 : tensor<8x197x3072xf32>
    %564 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %565 = stablehlo.multiply %564, %563 : tensor<8x197x3072xf32>
    %566 = stablehlo.add %561, %565 : tensor<8x197x3072xf32>
    %567 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %568 = stablehlo.multiply %567, %566 : tensor<8x197x3072xf32>
    %569 = stablehlo.tanh %568 : tensor<8x197x3072xf32>
    %570 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %571 = stablehlo.add %570, %569 : tensor<8x197x3072xf32>
    %572 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %573 = stablehlo.multiply %572, %571 : tensor<8x197x3072xf32>
    %574 = stablehlo.multiply %561, %573 : tensor<8x197x3072xf32>
    %575 = stablehlo.dot_general %574, %arg61, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
    %576 = stablehlo.add %536, %575 : tensor<8x197x768xf32>
    %cst_43 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %577 = stablehlo.reduce(%576 init: %cst_43) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %578 = stablehlo.broadcast_in_dim %577, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %579 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %580 = stablehlo.divide %578, %579 : tensor<8x197x1xf32>
    %581 = stablehlo.broadcast_in_dim %580, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %582 = stablehlo.subtract %576, %581 : tensor<8x197x768xf32>
    %583 = stablehlo.multiply %582, %582 : tensor<8x197x768xf32>
    %cst_44 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %584 = stablehlo.reduce(%583 init: %cst_44) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %585 = stablehlo.broadcast_in_dim %584, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %586 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %587 = stablehlo.divide %585, %586 : tensor<8x197x1xf32>
    %588 = stablehlo.broadcast_in_dim %580, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %589 = stablehlo.subtract %576, %588 : tensor<8x197x768xf32>
    %590 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %591 = stablehlo.add %587, %590 : tensor<8x197x1xf32>
    %592 = stablehlo.sqrt %591 : tensor<8x197x1xf32>
    %593 = stablehlo.broadcast_in_dim %592, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %594 = stablehlo.divide %589, %593 : tensor<8x197x768xf32>
    %595 = stablehlo.broadcast_in_dim %arg68, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %596 = stablehlo.broadcast_in_dim %595, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %597 = stablehlo.multiply %594, %596 : tensor<8x197x768xf32>
    %598 = stablehlo.broadcast_in_dim %arg69, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %599 = stablehlo.broadcast_in_dim %598, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %600 = stablehlo.add %597, %599 : tensor<8x197x768xf32>
    %601 = stablehlo.dot_general %600, %arg64, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %602 = stablehlo.dot_general %600, %arg65, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %603 = stablehlo.dot_general %600, %arg66, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %604 = stablehlo.reshape %601 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %605 = stablehlo.transpose %604, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %606 = stablehlo.reshape %602 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %607 = stablehlo.transpose %606, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %608 = stablehlo.reshape %603 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %609 = stablehlo.transpose %608, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %610 = stablehlo.transpose %607, dims = [0, 1, 3, 2] : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
    %611 = stablehlo.dot_general %605, %610, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
    %612 = stablehlo.sqrt %cst_3 : tensor<f32>
    %613 = stablehlo.convert %612 : tensor<f32>
    %614 = stablehlo.broadcast_in_dim %613, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
    %615 = stablehlo.divide %611, %614 : tensor<8x12x197x197xf32>
    %cst_45 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %616 = stablehlo.reduce(%615 init: %cst_45) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %617 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
    %618 = stablehlo.maximum %617, %616 : tensor<8x12x197xf32>
    %619 = stablehlo.broadcast_in_dim %618, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %620 = stablehlo.broadcast_in_dim %619, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %621 = stablehlo.subtract %615, %620 : tensor<8x12x197x197xf32>
    %622 = stablehlo.exponential %621 : tensor<8x12x197x197xf32>
    %cst_46 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %623 = stablehlo.reduce(%622 init: %cst_46) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %624 = stablehlo.broadcast_in_dim %623, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %625 = stablehlo.broadcast_in_dim %624, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %626 = stablehlo.divide %622, %625 : tensor<8x12x197x197xf32>
    %627 = stablehlo.dot_general %626, %609, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
    %628 = stablehlo.transpose %627, dims = [0, 2, 1, 3] : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
    %629 = stablehlo.reshape %628 : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
    %630 = stablehlo.dot_general %629, %arg67, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %631 = stablehlo.add %576, %630 : tensor<8x197x768xf32>
    %cst_47 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %632 = stablehlo.reduce(%631 init: %cst_47) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %633 = stablehlo.broadcast_in_dim %632, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %634 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %635 = stablehlo.divide %633, %634 : tensor<8x197x1xf32>
    %636 = stablehlo.broadcast_in_dim %635, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %637 = stablehlo.subtract %631, %636 : tensor<8x197x768xf32>
    %638 = stablehlo.multiply %637, %637 : tensor<8x197x768xf32>
    %cst_48 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %639 = stablehlo.reduce(%638 init: %cst_48) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %640 = stablehlo.broadcast_in_dim %639, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %641 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %642 = stablehlo.divide %640, %641 : tensor<8x197x1xf32>
    %643 = stablehlo.broadcast_in_dim %635, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %644 = stablehlo.subtract %631, %643 : tensor<8x197x768xf32>
    %645 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %646 = stablehlo.add %642, %645 : tensor<8x197x1xf32>
    %647 = stablehlo.sqrt %646 : tensor<8x197x1xf32>
    %648 = stablehlo.broadcast_in_dim %647, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %649 = stablehlo.divide %644, %648 : tensor<8x197x768xf32>
    %650 = stablehlo.broadcast_in_dim %arg72, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %651 = stablehlo.broadcast_in_dim %650, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %652 = stablehlo.multiply %649, %651 : tensor<8x197x768xf32>
    %653 = stablehlo.broadcast_in_dim %arg73, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %654 = stablehlo.broadcast_in_dim %653, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %655 = stablehlo.add %652, %654 : tensor<8x197x768xf32>
    %656 = stablehlo.dot_general %655, %arg70, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
    %657 = stablehlo.multiply %656, %656 : tensor<8x197x3072xf32>
    %658 = stablehlo.multiply %657, %656 : tensor<8x197x3072xf32>
    %659 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %660 = stablehlo.multiply %659, %658 : tensor<8x197x3072xf32>
    %661 = stablehlo.add %656, %660 : tensor<8x197x3072xf32>
    %662 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %663 = stablehlo.multiply %662, %661 : tensor<8x197x3072xf32>
    %664 = stablehlo.tanh %663 : tensor<8x197x3072xf32>
    %665 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %666 = stablehlo.add %665, %664 : tensor<8x197x3072xf32>
    %667 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %668 = stablehlo.multiply %667, %666 : tensor<8x197x3072xf32>
    %669 = stablehlo.multiply %656, %668 : tensor<8x197x3072xf32>
    %670 = stablehlo.dot_general %669, %arg71, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
    %671 = stablehlo.add %631, %670 : tensor<8x197x768xf32>
    %cst_49 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %672 = stablehlo.reduce(%671 init: %cst_49) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %673 = stablehlo.broadcast_in_dim %672, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %674 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %675 = stablehlo.divide %673, %674 : tensor<8x197x1xf32>
    %676 = stablehlo.broadcast_in_dim %675, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %677 = stablehlo.subtract %671, %676 : tensor<8x197x768xf32>
    %678 = stablehlo.multiply %677, %677 : tensor<8x197x768xf32>
    %cst_50 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %679 = stablehlo.reduce(%678 init: %cst_50) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %680 = stablehlo.broadcast_in_dim %679, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %681 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %682 = stablehlo.divide %680, %681 : tensor<8x197x1xf32>
    %683 = stablehlo.broadcast_in_dim %675, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %684 = stablehlo.subtract %671, %683 : tensor<8x197x768xf32>
    %685 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %686 = stablehlo.add %682, %685 : tensor<8x197x1xf32>
    %687 = stablehlo.sqrt %686 : tensor<8x197x1xf32>
    %688 = stablehlo.broadcast_in_dim %687, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %689 = stablehlo.divide %684, %688 : tensor<8x197x768xf32>
    %690 = stablehlo.broadcast_in_dim %arg78, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %691 = stablehlo.broadcast_in_dim %690, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %692 = stablehlo.multiply %689, %691 : tensor<8x197x768xf32>
    %693 = stablehlo.broadcast_in_dim %arg79, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %694 = stablehlo.broadcast_in_dim %693, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %695 = stablehlo.add %692, %694 : tensor<8x197x768xf32>
    %696 = stablehlo.dot_general %695, %arg74, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %697 = stablehlo.dot_general %695, %arg75, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %698 = stablehlo.dot_general %695, %arg76, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %699 = stablehlo.reshape %696 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %700 = stablehlo.transpose %699, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %701 = stablehlo.reshape %697 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %702 = stablehlo.transpose %701, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %703 = stablehlo.reshape %698 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %704 = stablehlo.transpose %703, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %705 = stablehlo.transpose %702, dims = [0, 1, 3, 2] : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
    %706 = stablehlo.dot_general %700, %705, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
    %707 = stablehlo.sqrt %cst_3 : tensor<f32>
    %708 = stablehlo.convert %707 : tensor<f32>
    %709 = stablehlo.broadcast_in_dim %708, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
    %710 = stablehlo.divide %706, %709 : tensor<8x12x197x197xf32>
    %cst_51 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %711 = stablehlo.reduce(%710 init: %cst_51) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %712 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
    %713 = stablehlo.maximum %712, %711 : tensor<8x12x197xf32>
    %714 = stablehlo.broadcast_in_dim %713, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %715 = stablehlo.broadcast_in_dim %714, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %716 = stablehlo.subtract %710, %715 : tensor<8x12x197x197xf32>
    %717 = stablehlo.exponential %716 : tensor<8x12x197x197xf32>
    %cst_52 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %718 = stablehlo.reduce(%717 init: %cst_52) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %719 = stablehlo.broadcast_in_dim %718, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %720 = stablehlo.broadcast_in_dim %719, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %721 = stablehlo.divide %717, %720 : tensor<8x12x197x197xf32>
    %722 = stablehlo.dot_general %721, %704, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
    %723 = stablehlo.transpose %722, dims = [0, 2, 1, 3] : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
    %724 = stablehlo.reshape %723 : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
    %725 = stablehlo.dot_general %724, %arg77, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %726 = stablehlo.add %671, %725 : tensor<8x197x768xf32>
    %cst_53 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %727 = stablehlo.reduce(%726 init: %cst_53) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %728 = stablehlo.broadcast_in_dim %727, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %729 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %730 = stablehlo.divide %728, %729 : tensor<8x197x1xf32>
    %731 = stablehlo.broadcast_in_dim %730, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %732 = stablehlo.subtract %726, %731 : tensor<8x197x768xf32>
    %733 = stablehlo.multiply %732, %732 : tensor<8x197x768xf32>
    %cst_54 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %734 = stablehlo.reduce(%733 init: %cst_54) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %735 = stablehlo.broadcast_in_dim %734, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %736 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %737 = stablehlo.divide %735, %736 : tensor<8x197x1xf32>
    %738 = stablehlo.broadcast_in_dim %730, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %739 = stablehlo.subtract %726, %738 : tensor<8x197x768xf32>
    %740 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %741 = stablehlo.add %737, %740 : tensor<8x197x1xf32>
    %742 = stablehlo.sqrt %741 : tensor<8x197x1xf32>
    %743 = stablehlo.broadcast_in_dim %742, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %744 = stablehlo.divide %739, %743 : tensor<8x197x768xf32>
    %745 = stablehlo.broadcast_in_dim %arg82, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %746 = stablehlo.broadcast_in_dim %745, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %747 = stablehlo.multiply %744, %746 : tensor<8x197x768xf32>
    %748 = stablehlo.broadcast_in_dim %arg83, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %749 = stablehlo.broadcast_in_dim %748, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %750 = stablehlo.add %747, %749 : tensor<8x197x768xf32>
    %751 = stablehlo.dot_general %750, %arg80, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
    %752 = stablehlo.multiply %751, %751 : tensor<8x197x3072xf32>
    %753 = stablehlo.multiply %752, %751 : tensor<8x197x3072xf32>
    %754 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %755 = stablehlo.multiply %754, %753 : tensor<8x197x3072xf32>
    %756 = stablehlo.add %751, %755 : tensor<8x197x3072xf32>
    %757 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %758 = stablehlo.multiply %757, %756 : tensor<8x197x3072xf32>
    %759 = stablehlo.tanh %758 : tensor<8x197x3072xf32>
    %760 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %761 = stablehlo.add %760, %759 : tensor<8x197x3072xf32>
    %762 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %763 = stablehlo.multiply %762, %761 : tensor<8x197x3072xf32>
    %764 = stablehlo.multiply %751, %763 : tensor<8x197x3072xf32>
    %765 = stablehlo.dot_general %764, %arg81, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
    %766 = stablehlo.add %726, %765 : tensor<8x197x768xf32>
    %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %767 = stablehlo.reduce(%766 init: %cst_55) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %768 = stablehlo.broadcast_in_dim %767, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %769 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %770 = stablehlo.divide %768, %769 : tensor<8x197x1xf32>
    %771 = stablehlo.broadcast_in_dim %770, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %772 = stablehlo.subtract %766, %771 : tensor<8x197x768xf32>
    %773 = stablehlo.multiply %772, %772 : tensor<8x197x768xf32>
    %cst_56 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %774 = stablehlo.reduce(%773 init: %cst_56) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %775 = stablehlo.broadcast_in_dim %774, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %776 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %777 = stablehlo.divide %775, %776 : tensor<8x197x1xf32>
    %778 = stablehlo.broadcast_in_dim %770, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %779 = stablehlo.subtract %766, %778 : tensor<8x197x768xf32>
    %780 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %781 = stablehlo.add %777, %780 : tensor<8x197x1xf32>
    %782 = stablehlo.sqrt %781 : tensor<8x197x1xf32>
    %783 = stablehlo.broadcast_in_dim %782, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %784 = stablehlo.divide %779, %783 : tensor<8x197x768xf32>
    %785 = stablehlo.broadcast_in_dim %arg88, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %786 = stablehlo.broadcast_in_dim %785, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %787 = stablehlo.multiply %784, %786 : tensor<8x197x768xf32>
    %788 = stablehlo.broadcast_in_dim %arg89, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %789 = stablehlo.broadcast_in_dim %788, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %790 = stablehlo.add %787, %789 : tensor<8x197x768xf32>
    %791 = stablehlo.dot_general %790, %arg84, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %792 = stablehlo.dot_general %790, %arg85, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %793 = stablehlo.dot_general %790, %arg86, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %794 = stablehlo.reshape %791 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %795 = stablehlo.transpose %794, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %796 = stablehlo.reshape %792 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %797 = stablehlo.transpose %796, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %798 = stablehlo.reshape %793 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %799 = stablehlo.transpose %798, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %800 = stablehlo.transpose %797, dims = [0, 1, 3, 2] : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
    %801 = stablehlo.dot_general %795, %800, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
    %802 = stablehlo.sqrt %cst_3 : tensor<f32>
    %803 = stablehlo.convert %802 : tensor<f32>
    %804 = stablehlo.broadcast_in_dim %803, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
    %805 = stablehlo.divide %801, %804 : tensor<8x12x197x197xf32>
    %cst_57 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %806 = stablehlo.reduce(%805 init: %cst_57) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %807 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
    %808 = stablehlo.maximum %807, %806 : tensor<8x12x197xf32>
    %809 = stablehlo.broadcast_in_dim %808, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %810 = stablehlo.broadcast_in_dim %809, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %811 = stablehlo.subtract %805, %810 : tensor<8x12x197x197xf32>
    %812 = stablehlo.exponential %811 : tensor<8x12x197x197xf32>
    %cst_58 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %813 = stablehlo.reduce(%812 init: %cst_58) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %814 = stablehlo.broadcast_in_dim %813, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %815 = stablehlo.broadcast_in_dim %814, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %816 = stablehlo.divide %812, %815 : tensor<8x12x197x197xf32>
    %817 = stablehlo.dot_general %816, %799, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
    %818 = stablehlo.transpose %817, dims = [0, 2, 1, 3] : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
    %819 = stablehlo.reshape %818 : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
    %820 = stablehlo.dot_general %819, %arg87, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %821 = stablehlo.add %766, %820 : tensor<8x197x768xf32>
    %cst_59 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %822 = stablehlo.reduce(%821 init: %cst_59) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %823 = stablehlo.broadcast_in_dim %822, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %824 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %825 = stablehlo.divide %823, %824 : tensor<8x197x1xf32>
    %826 = stablehlo.broadcast_in_dim %825, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %827 = stablehlo.subtract %821, %826 : tensor<8x197x768xf32>
    %828 = stablehlo.multiply %827, %827 : tensor<8x197x768xf32>
    %cst_60 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %829 = stablehlo.reduce(%828 init: %cst_60) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %830 = stablehlo.broadcast_in_dim %829, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %831 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %832 = stablehlo.divide %830, %831 : tensor<8x197x1xf32>
    %833 = stablehlo.broadcast_in_dim %825, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %834 = stablehlo.subtract %821, %833 : tensor<8x197x768xf32>
    %835 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %836 = stablehlo.add %832, %835 : tensor<8x197x1xf32>
    %837 = stablehlo.sqrt %836 : tensor<8x197x1xf32>
    %838 = stablehlo.broadcast_in_dim %837, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %839 = stablehlo.divide %834, %838 : tensor<8x197x768xf32>
    %840 = stablehlo.broadcast_in_dim %arg92, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %841 = stablehlo.broadcast_in_dim %840, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %842 = stablehlo.multiply %839, %841 : tensor<8x197x768xf32>
    %843 = stablehlo.broadcast_in_dim %arg93, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %844 = stablehlo.broadcast_in_dim %843, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %845 = stablehlo.add %842, %844 : tensor<8x197x768xf32>
    %846 = stablehlo.dot_general %845, %arg90, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
    %847 = stablehlo.multiply %846, %846 : tensor<8x197x3072xf32>
    %848 = stablehlo.multiply %847, %846 : tensor<8x197x3072xf32>
    %849 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %850 = stablehlo.multiply %849, %848 : tensor<8x197x3072xf32>
    %851 = stablehlo.add %846, %850 : tensor<8x197x3072xf32>
    %852 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %853 = stablehlo.multiply %852, %851 : tensor<8x197x3072xf32>
    %854 = stablehlo.tanh %853 : tensor<8x197x3072xf32>
    %855 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %856 = stablehlo.add %855, %854 : tensor<8x197x3072xf32>
    %857 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %858 = stablehlo.multiply %857, %856 : tensor<8x197x3072xf32>
    %859 = stablehlo.multiply %846, %858 : tensor<8x197x3072xf32>
    %860 = stablehlo.dot_general %859, %arg91, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
    %861 = stablehlo.add %821, %860 : tensor<8x197x768xf32>
    %cst_61 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %862 = stablehlo.reduce(%861 init: %cst_61) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %863 = stablehlo.broadcast_in_dim %862, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %864 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %865 = stablehlo.divide %863, %864 : tensor<8x197x1xf32>
    %866 = stablehlo.broadcast_in_dim %865, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %867 = stablehlo.subtract %861, %866 : tensor<8x197x768xf32>
    %868 = stablehlo.multiply %867, %867 : tensor<8x197x768xf32>
    %cst_62 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %869 = stablehlo.reduce(%868 init: %cst_62) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %870 = stablehlo.broadcast_in_dim %869, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %871 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %872 = stablehlo.divide %870, %871 : tensor<8x197x1xf32>
    %873 = stablehlo.broadcast_in_dim %865, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %874 = stablehlo.subtract %861, %873 : tensor<8x197x768xf32>
    %875 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %876 = stablehlo.add %872, %875 : tensor<8x197x1xf32>
    %877 = stablehlo.sqrt %876 : tensor<8x197x1xf32>
    %878 = stablehlo.broadcast_in_dim %877, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %879 = stablehlo.divide %874, %878 : tensor<8x197x768xf32>
    %880 = stablehlo.broadcast_in_dim %arg98, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %881 = stablehlo.broadcast_in_dim %880, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %882 = stablehlo.multiply %879, %881 : tensor<8x197x768xf32>
    %883 = stablehlo.broadcast_in_dim %arg99, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %884 = stablehlo.broadcast_in_dim %883, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %885 = stablehlo.add %882, %884 : tensor<8x197x768xf32>
    %886 = stablehlo.dot_general %885, %arg94, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %887 = stablehlo.dot_general %885, %arg95, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %888 = stablehlo.dot_general %885, %arg96, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %889 = stablehlo.reshape %886 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %890 = stablehlo.transpose %889, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %891 = stablehlo.reshape %887 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %892 = stablehlo.transpose %891, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %893 = stablehlo.reshape %888 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %894 = stablehlo.transpose %893, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %895 = stablehlo.transpose %892, dims = [0, 1, 3, 2] : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
    %896 = stablehlo.dot_general %890, %895, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
    %897 = stablehlo.sqrt %cst_3 : tensor<f32>
    %898 = stablehlo.convert %897 : tensor<f32>
    %899 = stablehlo.broadcast_in_dim %898, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
    %900 = stablehlo.divide %896, %899 : tensor<8x12x197x197xf32>
    %cst_63 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %901 = stablehlo.reduce(%900 init: %cst_63) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %902 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
    %903 = stablehlo.maximum %902, %901 : tensor<8x12x197xf32>
    %904 = stablehlo.broadcast_in_dim %903, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %905 = stablehlo.broadcast_in_dim %904, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %906 = stablehlo.subtract %900, %905 : tensor<8x12x197x197xf32>
    %907 = stablehlo.exponential %906 : tensor<8x12x197x197xf32>
    %cst_64 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %908 = stablehlo.reduce(%907 init: %cst_64) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %909 = stablehlo.broadcast_in_dim %908, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %910 = stablehlo.broadcast_in_dim %909, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %911 = stablehlo.divide %907, %910 : tensor<8x12x197x197xf32>
    %912 = stablehlo.dot_general %911, %894, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
    %913 = stablehlo.transpose %912, dims = [0, 2, 1, 3] : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
    %914 = stablehlo.reshape %913 : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
    %915 = stablehlo.dot_general %914, %arg97, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %916 = stablehlo.add %861, %915 : tensor<8x197x768xf32>
    %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %917 = stablehlo.reduce(%916 init: %cst_65) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %918 = stablehlo.broadcast_in_dim %917, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %919 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %920 = stablehlo.divide %918, %919 : tensor<8x197x1xf32>
    %921 = stablehlo.broadcast_in_dim %920, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %922 = stablehlo.subtract %916, %921 : tensor<8x197x768xf32>
    %923 = stablehlo.multiply %922, %922 : tensor<8x197x768xf32>
    %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %924 = stablehlo.reduce(%923 init: %cst_66) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %925 = stablehlo.broadcast_in_dim %924, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %926 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %927 = stablehlo.divide %925, %926 : tensor<8x197x1xf32>
    %928 = stablehlo.broadcast_in_dim %920, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %929 = stablehlo.subtract %916, %928 : tensor<8x197x768xf32>
    %930 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %931 = stablehlo.add %927, %930 : tensor<8x197x1xf32>
    %932 = stablehlo.sqrt %931 : tensor<8x197x1xf32>
    %933 = stablehlo.broadcast_in_dim %932, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %934 = stablehlo.divide %929, %933 : tensor<8x197x768xf32>
    %935 = stablehlo.broadcast_in_dim %arg102, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %936 = stablehlo.broadcast_in_dim %935, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %937 = stablehlo.multiply %934, %936 : tensor<8x197x768xf32>
    %938 = stablehlo.broadcast_in_dim %arg103, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %939 = stablehlo.broadcast_in_dim %938, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %940 = stablehlo.add %937, %939 : tensor<8x197x768xf32>
    %941 = stablehlo.dot_general %940, %arg100, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
    %942 = stablehlo.multiply %941, %941 : tensor<8x197x3072xf32>
    %943 = stablehlo.multiply %942, %941 : tensor<8x197x3072xf32>
    %944 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %945 = stablehlo.multiply %944, %943 : tensor<8x197x3072xf32>
    %946 = stablehlo.add %941, %945 : tensor<8x197x3072xf32>
    %947 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %948 = stablehlo.multiply %947, %946 : tensor<8x197x3072xf32>
    %949 = stablehlo.tanh %948 : tensor<8x197x3072xf32>
    %950 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %951 = stablehlo.add %950, %949 : tensor<8x197x3072xf32>
    %952 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %953 = stablehlo.multiply %952, %951 : tensor<8x197x3072xf32>
    %954 = stablehlo.multiply %941, %953 : tensor<8x197x3072xf32>
    %955 = stablehlo.dot_general %954, %arg101, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
    %956 = stablehlo.add %916, %955 : tensor<8x197x768xf32>
    %cst_67 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %957 = stablehlo.reduce(%956 init: %cst_67) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %958 = stablehlo.broadcast_in_dim %957, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %959 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %960 = stablehlo.divide %958, %959 : tensor<8x197x1xf32>
    %961 = stablehlo.broadcast_in_dim %960, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %962 = stablehlo.subtract %956, %961 : tensor<8x197x768xf32>
    %963 = stablehlo.multiply %962, %962 : tensor<8x197x768xf32>
    %cst_68 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %964 = stablehlo.reduce(%963 init: %cst_68) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %965 = stablehlo.broadcast_in_dim %964, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %966 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %967 = stablehlo.divide %965, %966 : tensor<8x197x1xf32>
    %968 = stablehlo.broadcast_in_dim %960, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %969 = stablehlo.subtract %956, %968 : tensor<8x197x768xf32>
    %970 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %971 = stablehlo.add %967, %970 : tensor<8x197x1xf32>
    %972 = stablehlo.sqrt %971 : tensor<8x197x1xf32>
    %973 = stablehlo.broadcast_in_dim %972, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %974 = stablehlo.divide %969, %973 : tensor<8x197x768xf32>
    %975 = stablehlo.broadcast_in_dim %arg108, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %976 = stablehlo.broadcast_in_dim %975, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %977 = stablehlo.multiply %974, %976 : tensor<8x197x768xf32>
    %978 = stablehlo.broadcast_in_dim %arg109, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %979 = stablehlo.broadcast_in_dim %978, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %980 = stablehlo.add %977, %979 : tensor<8x197x768xf32>
    %981 = stablehlo.dot_general %980, %arg104, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %982 = stablehlo.dot_general %980, %arg105, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %983 = stablehlo.dot_general %980, %arg106, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %984 = stablehlo.reshape %981 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %985 = stablehlo.transpose %984, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %986 = stablehlo.reshape %982 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %987 = stablehlo.transpose %986, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %988 = stablehlo.reshape %983 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %989 = stablehlo.transpose %988, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %990 = stablehlo.transpose %987, dims = [0, 1, 3, 2] : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
    %991 = stablehlo.dot_general %985, %990, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
    %992 = stablehlo.sqrt %cst_3 : tensor<f32>
    %993 = stablehlo.convert %992 : tensor<f32>
    %994 = stablehlo.broadcast_in_dim %993, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
    %995 = stablehlo.divide %991, %994 : tensor<8x12x197x197xf32>
    %cst_69 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %996 = stablehlo.reduce(%995 init: %cst_69) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %997 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
    %998 = stablehlo.maximum %997, %996 : tensor<8x12x197xf32>
    %999 = stablehlo.broadcast_in_dim %998, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %1000 = stablehlo.broadcast_in_dim %999, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %1001 = stablehlo.subtract %995, %1000 : tensor<8x12x197x197xf32>
    %1002 = stablehlo.exponential %1001 : tensor<8x12x197x197xf32>
    %cst_70 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1003 = stablehlo.reduce(%1002 init: %cst_70) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %1004 = stablehlo.broadcast_in_dim %1003, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %1005 = stablehlo.broadcast_in_dim %1004, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %1006 = stablehlo.divide %1002, %1005 : tensor<8x12x197x197xf32>
    %1007 = stablehlo.dot_general %1006, %989, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
    %1008 = stablehlo.transpose %1007, dims = [0, 2, 1, 3] : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
    %1009 = stablehlo.reshape %1008 : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
    %1010 = stablehlo.dot_general %1009, %arg107, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %1011 = stablehlo.add %956, %1010 : tensor<8x197x768xf32>
    %cst_71 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1012 = stablehlo.reduce(%1011 init: %cst_71) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %1013 = stablehlo.broadcast_in_dim %1012, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %1014 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %1015 = stablehlo.divide %1013, %1014 : tensor<8x197x1xf32>
    %1016 = stablehlo.broadcast_in_dim %1015, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %1017 = stablehlo.subtract %1011, %1016 : tensor<8x197x768xf32>
    %1018 = stablehlo.multiply %1017, %1017 : tensor<8x197x768xf32>
    %cst_72 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1019 = stablehlo.reduce(%1018 init: %cst_72) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %1020 = stablehlo.broadcast_in_dim %1019, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %1021 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %1022 = stablehlo.divide %1020, %1021 : tensor<8x197x1xf32>
    %1023 = stablehlo.broadcast_in_dim %1015, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %1024 = stablehlo.subtract %1011, %1023 : tensor<8x197x768xf32>
    %1025 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %1026 = stablehlo.add %1022, %1025 : tensor<8x197x1xf32>
    %1027 = stablehlo.sqrt %1026 : tensor<8x197x1xf32>
    %1028 = stablehlo.broadcast_in_dim %1027, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %1029 = stablehlo.divide %1024, %1028 : tensor<8x197x768xf32>
    %1030 = stablehlo.broadcast_in_dim %arg112, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1031 = stablehlo.broadcast_in_dim %1030, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %1032 = stablehlo.multiply %1029, %1031 : tensor<8x197x768xf32>
    %1033 = stablehlo.broadcast_in_dim %arg113, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1034 = stablehlo.broadcast_in_dim %1033, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %1035 = stablehlo.add %1032, %1034 : tensor<8x197x768xf32>
    %1036 = stablehlo.dot_general %1035, %arg110, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
    %1037 = stablehlo.multiply %1036, %1036 : tensor<8x197x3072xf32>
    %1038 = stablehlo.multiply %1037, %1036 : tensor<8x197x3072xf32>
    %1039 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %1040 = stablehlo.multiply %1039, %1038 : tensor<8x197x3072xf32>
    %1041 = stablehlo.add %1036, %1040 : tensor<8x197x3072xf32>
    %1042 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %1043 = stablehlo.multiply %1042, %1041 : tensor<8x197x3072xf32>
    %1044 = stablehlo.tanh %1043 : tensor<8x197x3072xf32>
    %1045 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %1046 = stablehlo.add %1045, %1044 : tensor<8x197x3072xf32>
    %1047 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %1048 = stablehlo.multiply %1047, %1046 : tensor<8x197x3072xf32>
    %1049 = stablehlo.multiply %1036, %1048 : tensor<8x197x3072xf32>
    %1050 = stablehlo.dot_general %1049, %arg111, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
    %1051 = stablehlo.add %1011, %1050 : tensor<8x197x768xf32>
    %cst_73 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1052 = stablehlo.reduce(%1051 init: %cst_73) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %1053 = stablehlo.broadcast_in_dim %1052, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %1054 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %1055 = stablehlo.divide %1053, %1054 : tensor<8x197x1xf32>
    %1056 = stablehlo.broadcast_in_dim %1055, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %1057 = stablehlo.subtract %1051, %1056 : tensor<8x197x768xf32>
    %1058 = stablehlo.multiply %1057, %1057 : tensor<8x197x768xf32>
    %cst_74 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1059 = stablehlo.reduce(%1058 init: %cst_74) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %1060 = stablehlo.broadcast_in_dim %1059, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %1061 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %1062 = stablehlo.divide %1060, %1061 : tensor<8x197x1xf32>
    %1063 = stablehlo.broadcast_in_dim %1055, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %1064 = stablehlo.subtract %1051, %1063 : tensor<8x197x768xf32>
    %1065 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %1066 = stablehlo.add %1062, %1065 : tensor<8x197x1xf32>
    %1067 = stablehlo.sqrt %1066 : tensor<8x197x1xf32>
    %1068 = stablehlo.broadcast_in_dim %1067, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %1069 = stablehlo.divide %1064, %1068 : tensor<8x197x768xf32>
    %1070 = stablehlo.broadcast_in_dim %arg118, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1071 = stablehlo.broadcast_in_dim %1070, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %1072 = stablehlo.multiply %1069, %1071 : tensor<8x197x768xf32>
    %1073 = stablehlo.broadcast_in_dim %arg119, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1074 = stablehlo.broadcast_in_dim %1073, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %1075 = stablehlo.add %1072, %1074 : tensor<8x197x768xf32>
    %1076 = stablehlo.dot_general %1075, %arg114, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %1077 = stablehlo.dot_general %1075, %arg115, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %1078 = stablehlo.dot_general %1075, %arg116, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %1079 = stablehlo.reshape %1076 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %1080 = stablehlo.transpose %1079, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %1081 = stablehlo.reshape %1077 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %1082 = stablehlo.transpose %1081, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %1083 = stablehlo.reshape %1078 : (tensor<8x197x768xf32>) -> tensor<8x197x12x64xf32>
    %1084 = stablehlo.transpose %1083, dims = [0, 2, 1, 3] : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
    %1085 = stablehlo.transpose %1082, dims = [0, 1, 3, 2] : (tensor<8x12x197x64xf32>) -> tensor<8x12x64x197xf32>
    %1086 = stablehlo.dot_general %1080, %1085, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x64xf32>, tensor<8x12x64x197xf32>) -> tensor<8x12x197x197xf32>
    %1087 = stablehlo.sqrt %cst_3 : tensor<f32>
    %1088 = stablehlo.convert %1087 : tensor<f32>
    %1089 = stablehlo.broadcast_in_dim %1088, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
    %1090 = stablehlo.divide %1086, %1089 : tensor<8x12x197x197xf32>
    %cst_75 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1091 = stablehlo.reduce(%1090 init: %cst_75) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %1092 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
    %1093 = stablehlo.maximum %1092, %1091 : tensor<8x12x197xf32>
    %1094 = stablehlo.broadcast_in_dim %1093, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %1095 = stablehlo.broadcast_in_dim %1094, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %1096 = stablehlo.subtract %1090, %1095 : tensor<8x12x197x197xf32>
    %1097 = stablehlo.exponential %1096 : tensor<8x12x197x197xf32>
    %cst_76 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1098 = stablehlo.reduce(%1097 init: %cst_76) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %1099 = stablehlo.broadcast_in_dim %1098, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x1xf32>
    %1100 = stablehlo.broadcast_in_dim %1099, dims = [0, 1, 2, 3] : (tensor<8x12x197x1xf32>) -> tensor<8x12x197x197xf32>
    %1101 = stablehlo.divide %1097, %1100 : tensor<8x12x197x197xf32>
    %1102 = stablehlo.dot_general %1101, %1084, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x12x197x197xf32>, tensor<8x12x197x64xf32>) -> tensor<8x12x197x64xf32>
    %1103 = stablehlo.transpose %1102, dims = [0, 2, 1, 3] : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
    %1104 = stablehlo.reshape %1103 : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
    %1105 = stablehlo.dot_general %1104, %arg117, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x768xf32>) -> tensor<8x197x768xf32>
    %1106 = stablehlo.add %1051, %1105 : tensor<8x197x768xf32>
    %cst_77 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1107 = stablehlo.reduce(%1106 init: %cst_77) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %1108 = stablehlo.broadcast_in_dim %1107, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %1109 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %1110 = stablehlo.divide %1108, %1109 : tensor<8x197x1xf32>
    %1111 = stablehlo.broadcast_in_dim %1110, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %1112 = stablehlo.subtract %1106, %1111 : tensor<8x197x768xf32>
    %1113 = stablehlo.multiply %1112, %1112 : tensor<8x197x768xf32>
    %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1114 = stablehlo.reduce(%1113 init: %cst_78) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %1115 = stablehlo.broadcast_in_dim %1114, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x1xf32>
    %1116 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %1117 = stablehlo.divide %1115, %1116 : tensor<8x197x1xf32>
    %1118 = stablehlo.broadcast_in_dim %1110, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %1119 = stablehlo.subtract %1106, %1118 : tensor<8x197x768xf32>
    %1120 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197x1xf32>
    %1121 = stablehlo.add %1117, %1120 : tensor<8x197x1xf32>
    %1122 = stablehlo.sqrt %1121 : tensor<8x197x1xf32>
    %1123 = stablehlo.broadcast_in_dim %1122, dims = [0, 1, 2] : (tensor<8x197x1xf32>) -> tensor<8x197x768xf32>
    %1124 = stablehlo.divide %1119, %1123 : tensor<8x197x768xf32>
    %1125 = stablehlo.broadcast_in_dim %arg122, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1126 = stablehlo.broadcast_in_dim %1125, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %1127 = stablehlo.multiply %1124, %1126 : tensor<8x197x768xf32>
    %1128 = stablehlo.broadcast_in_dim %arg123, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1129 = stablehlo.broadcast_in_dim %1128, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %1130 = stablehlo.add %1127, %1129 : tensor<8x197x768xf32>
    %1131 = stablehlo.dot_general %1130, %arg120, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<768x3072xf32>) -> tensor<8x197x3072xf32>
    %1132 = stablehlo.multiply %1131, %1131 : tensor<8x197x3072xf32>
    %1133 = stablehlo.multiply %1132, %1131 : tensor<8x197x3072xf32>
    %1134 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %1135 = stablehlo.multiply %1134, %1133 : tensor<8x197x3072xf32>
    %1136 = stablehlo.add %1131, %1135 : tensor<8x197x3072xf32>
    %1137 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %1138 = stablehlo.multiply %1137, %1136 : tensor<8x197x3072xf32>
    %1139 = stablehlo.tanh %1138 : tensor<8x197x3072xf32>
    %1140 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %1141 = stablehlo.add %1140, %1139 : tensor<8x197x3072xf32>
    %1142 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
    %1143 = stablehlo.multiply %1142, %1141 : tensor<8x197x3072xf32>
    %1144 = stablehlo.multiply %1131, %1143 : tensor<8x197x3072xf32>
    %1145 = stablehlo.dot_general %1144, %arg121, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x197x3072xf32>, tensor<3072x768xf32>) -> tensor<8x197x768xf32>
    %1146 = stablehlo.add %1106, %1145 : tensor<8x197x768xf32>
    return %1146 : tensor<8x197x768xf32>
  }
}
