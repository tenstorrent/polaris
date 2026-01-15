#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
from turtle import shape
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../'))
import ttsim.front.ttnn as ttnn
import numpy as np
from workloads.ttnn.vadv2.tt.tt_head import TtVADHead
from loguru import logger

def test_vadv2_head():
    device = ttnn.open_device(device_id=0)
    
    def tp(shape):
        return ttnn._rand(shape=shape, dtype=ttnn.bfloat16, device=device)

    parameter = {

    "head": {

        "positional_encoding": {
            "row_embed": {"weight": tp((100,128))},
            "col_embed": {"weight": tp((100,128))},
        },

        # ----------------------------
        # MOTION DECODER
        # ----------------------------
        "motion_decoder": {
            "layers": {
                "layer0": {
                    "attentions": {
                        "attn0": {
                            "in_proj": {
                                "weight": tp((256,768)),
                                "bias": tp((1,768)),
                            },
                            "out_proj": {
                                "weight": tp((256,256)),
                                "bias": tp((1,256)),
                            }
                        }
                    },
                    "ffn": {
                        "ffn0": {
                            "linear1": {
                                "weight": tp((256,512)),
                                "bias": tp((1,512)),
                            },
                            "linear2": {
                                "weight": tp((512,256)),
                                "bias": tp((1,256)),
                            },
                        }
                    },
                    "norms": {
                        "norm0": {
                            "weight": tp((1,256)),
                            "bias": tp((1,256)),
                        },
                        "norm1": {
                            "weight": tp((1,256)),
                            "bias": tp((1,256)),
                        },
                    },
                }
            }
        },

        # ----------------------------
        # MOTION MAP DECODER
        # ----------------------------
        "motion_map_decoder": {
            "layers": {
                "layer0": {
                    "attentions": {
                        "attn0": {
                            "in_proj": {
                                "weight": tp((256,768)),
                                "bias": tp((1,768)),
                            },
                            "out_proj": {
                                "weight": tp((256,256)),
                                "bias": tp((1,256)),
                            }
                        }
                    },
                    "ffn": {
                        "ffn0": {
                            "linear1": {"weight": tp((256,512)), "bias": tp((1,512))},
                            "linear2": {"weight": tp((512,256)), "bias": tp((1,256))},
                        }
                    },
                    "norms": {
                        "norm0": {"weight": tp((1,256)), "bias": tp((1,256))},
                        "norm1": {"weight": tp((1,256)), "bias": tp((1,256))},
                    },
                }
            }
        },

        # ----------------------------
        # EGO MAP DECODER
        # ----------------------------
        "ego_map_decoder": {
            "layers": {
                "layer0": {
                    "attentions": {
                        "attn0": {
                            "in_proj": {"weight": tp((256,768)),"bias": tp((1,768))},
                            "out_proj": {"weight": tp((256,256)), "bias": tp((1,256))}
                        }
                    },
                    "ffn": {
                        "ffn0": {
                            "linear1": {"weight": tp((256,512)), "bias": tp((1,512))},
                            "linear2": {"weight": tp((512,256)), "bias": tp((1,256))},
                        }
                    },
                    "norms": {
                        "norm0": {"weight": tp((1,256)), "bias": tp((1,256))},
                        "norm1": {"weight": tp((1,256)), "bias": tp((1,256))},
                    }
                }
            }
        },

        # ----------------------------
        # EGO AGENT DECODER
        # ----------------------------
        "ego_agent_decoder": {
            "layers": {
                "layer0": {
                    "attentions": {
                        "attn0": {
                            "in_proj": {"weight": tp((256,768)), "bias": tp((1,768))},
                            "out_proj": {"weight": tp((256,256)), "bias": tp((1,256))}
                        }
                    },
                    "ffn": {
                        "ffn0": {
                            "linear1": {"weight": tp((256,512)), "bias": tp((1,512))},
                            "linear2": {"weight": tp((512,256)), "bias": tp((1,256))},
                        }
                    },
                    "norms": {
                        "norm0": {"weight": tp((1,256)), "bias": tp((1,256))},
                        "norm1": {"weight": tp((1,256)), "bias": tp((1,256))},
                    }
                }
            }
        },

        # ----------------------------
        # LANE ENCODER
        # ----------------------------
        "lane_encoder": {
            f"lmlp_{i}": {
                "linear": {
                    "weight": tp((256,128)),
                    "bias": tp((1,128)),
                },
                "norm": {
                    "weight": tp((1,128)),
                    "bias": tp((1,128)),
                }
            }
            for i in range(3)
        },

        # --------------------------------------------------------
        # TRANSFORMER (encoder 0–2, decoder 0–2, map_decoder 0–2)
        # --------------------------------------------------------
        "transformer": {
            "encoder": {
                "layers": {
                    f"layer{k}": {
                        "attentions": {
                            "attn0": {
                                "sampling_offsets": {"weight": tp((512,128)), "bias": tp((1,128))},
                                "attention_weights": {"weight": tp((512,64)), "bias": tp((1,64))},
                                "value_proj": {"weight": tp((256,256)), "bias": tp((1,256))},
                                "output_proj": {"weight": tp((256,256)), "bias": tp((1,256))}
                            },
                            "attn1": {
                                "sampling_offsets": {"weight": tp((256,128)), "bias": tp((1,128))},
                                "attention_weights": {"weight": tp((256,64)), "bias": tp((1,64))},
                                "value_proj": {"weight": tp((256,256)), "bias": tp((1,256))},
                                "output_proj": {"weight": tp((256,256)), "bias": tp((1,256))}
                            }
                        },
                        "ffn": {
                            "ffn0": {
                                "linear1": {"weight": tp((256,512)), "bias": tp((1,512))},
                                "linear2": {"weight": tp((512,256)), "bias": tp((1,256))}
                            }
                        },
                        "norms": {
                            "norm0": {"weight": tp((1,256)), "bias": tp((1,256))},
                            "norm1": {"weight": tp((1,256)), "bias": tp((1,256))},
                            "norm2": {"weight": tp((1,256)), "bias": tp((1,256))}
                        }
                    }
                    for k in range(3)
                }
            },

            "decoder": {
                "layers": {
                    f"layer{k}": {
                        "attentions": {
                            "attn0": {
                                "in_proj": {"weight": tp((256,768)), "bias": tp((1,768))},
                                "out_proj": {"weight": tp((256,256)), "bias": tp((1,256))}
                            },
                            "attn1": {
                                "sampling_offsets": {"weight": tp((256,64)), "bias": tp((1,64))},
                                "attention_weights": {"weight": tp((256,32)), "bias": tp((1,32))},
                                "value_proj": {"weight": tp((256,256)), "bias": tp((1,256))},
                                "output_proj": {"weight": tp((256,256)), "bias": tp((1,256))}
                            }
                        },
                        "ffn": {
                            "ffn0": {
                                "linear1": {"weight": tp((256,512)), "bias": tp((1,512))},
                                "linear2": {"weight": tp((512,256)), "bias": tp((1,256))}
                            }
                        },
                        "norms": {
                            "norm0": {"weight": tp((1,256)), "bias": tp((1,256))},
                            "norm1": {"weight": tp((1,256)), "bias": tp((1,256))},
                            "norm2": {"weight": tp((1,256)), "bias": tp((1,256))}
                        }
                    }
                    for k in range(3)
                }
            },

            "map_decoder": {
                "layers": {
                    f"layer{k}": {
                        "attentions": {
                            "attn0": {
                                "in_proj": {"weight": tp((256,768)), "bias": tp((1,768))},
                                "out_proj": {"weight": tp((256,256)), "bias": tp((1,256))}
                            },
                            "attn1": {
                                "sampling_offsets": {"weight": tp((256,64)), "bias": tp((1,64))},
                                "attention_weights": {"weight": tp((256,32)), "bias": tp((1,32))},
                                "value_proj": {"weight": tp((256,256)), "bias": tp((1,256))},
                                "output_proj": {"weight": tp((256,256)), "bias": tp((1,256))}
                            }
                        },
                        "ffn": {
                            "ffn0": {
                                "linear1": {"weight": tp((256,512)), "bias": tp((1,512))},
                                "linear2": {"weight": tp((512,256)), "bias": tp((1,256))}
                            }
                        },
                        "norms": {
                            "norm0": {"weight": tp((1,256)), "bias": tp((1,256))},
                            "norm1": {"weight": tp((1,256)), "bias": tp((1,256))},
                            "norm2": {"weight": tp((1,256)), "bias": tp((1,256))}
                        }
                    }
                    for k in range(3)
                }
            },

            "reference_points": {"weight": tp((256,3)), "bias": tp((1,3))},
            "map_reference_points": {"weight": tp((256,2)), "bias": tp((1,2))},

            "can_bus_mlp": {
                "0": {"weight": tp((18,128)), "bias": tp((1,128))},
                "1": {"weight": tp((128,256)), "bias": tp((1,256))},
                "norm": {"weight": tp((1,256)), "bias": tp((1,256))}
            },

            "level_embeds": tp((4,256)),
            "cams_embeds": tp((6,256)),
        },

        # ----------------------------
        # POSITIONAL MLPs + EMBEDDINGS
        # ----------------------------
        "pos_mlp_sa": {"weight": tp((2,256)), "bias": tp((1,256))},
        "pos_mlp": {"weight": tp((2,256)), "bias": tp((1,256))},
        "ego_agent_pos_mlp": {"weight": tp((2,256)), "bias": tp((1,256))},
        "ego_map_pos_mlp": {"weight": tp((2,256)), "bias": tp((1,256))},

        "bev_embedding": {"weight": tp((10000,256))},
        "query_embedding": {"weight": tp((300,512))},
        "map_instance_embedding": {"weight": tp((100,512))},
        "map_pts_embedding": {"weight": tp((20,512))},

        "motion_mode_query": {"weight": tp((6,256))},
        "ego_query": {"weight": tp((1,256))},

        # ----------------------------
        # BRANCHES (cls/reg/traj/map)
        # ----------------------------
        "branches": {
            "cls_branches": {
                str(i): {
                    "0": {"weight": tp((256,256)), "bias": tp((1,256))},
                    "1_norm": {"weight": tp((1,256)), "bias": tp((1,256))},
                    "2": {"weight": tp((256,256)), "bias": tp((1,256))},
                    "3_norm": {"weight": tp((1,256)), "bias": tp((1,256))},
                    "4": {"weight": tp((256,10)), "bias": tp((1,10))}
                }
                for i in range(3)
            },

            "reg_branches": {
                str(i): {
                    "0": {"weight": tp((256,256)), "bias": tp((1,256))},
                    "1": {"weight": tp((256,256)), "bias": tp((1,256))},
                    "2": {"weight": tp((256,10)), "bias": tp((1,10))}
                }
                for i in range(3)
            },

            "traj_branches": {
                "0": {
                    "0": {"weight": tp((512,512)), "bias": tp((1,512))},
                    "1": {"weight": tp((512,512)), "bias": tp((1,512))},
                    "2": {"weight": tp((512,12)), "bias": tp((1,12))}
                }
            },

            "map_cls_branches": {
                str(i): {
                    "0": {"weight": tp((256,256)), "bias": tp((1,256))},
                    "1_norm": {"weight": tp((1,256)), "bias": tp((1,256))},
                    "2": {"weight": tp((256,256)), "bias": tp((1,256))},
                    "3_norm": {"weight": tp((1,256)), "bias": tp((1,256))},
                    "4": {"weight": tp((256,3)), "bias": tp((1,3))}
                }
                for i in range(3)
            },

            "map_reg_branches": {
                str(i): {
                    "0": {"weight": tp((256,256)), "bias": tp((1,256))},
                    "1": {"weight": tp((256,256)), "bias": tp((1,256))},
                    "2": {"weight": tp((256,2)), "bias": tp((1,2))}
                }
                for i in range(3)
            },

            "ego_fut_decoder": {
                "0": {"0": {"weight": tp((512,512)), "bias": tp((1,512))}},
                "2": {"0": {"weight": tp((512,512)), "bias": tp((1,512))}},
                "4": {"0": {"weight": tp((512,36)), "bias": tp((1,36))}},
            },

            "agent_fus_mlp": {
                "0": {"weight": tp((3072,256)), "bias": tp((1,256))},
                "1_norm": {"weight": tp((1,256)), "bias": tp((1,256))},
                "3": {"weight": tp((256,256)), "bias": tp((1,256))}
            },

            "traj_cls_branches": {
                "0": {
                    "0": {"weight": tp((512,512)), "bias": tp((1,512))},
                    "1_norm": {"weight": tp((1,512)), "bias": tp((1,512))},
                    "2": {"weight": tp((512,512)), "bias": tp((1,512))},
                    "3_norm": {"weight": tp((1,512)), "bias": tp((1,512))},
                    "4": {"weight": tp((512,1)), "bias": tp((1,1))}
                }
            }
        }
    }
}

    tt_model = TtVADHead(
        params=parameter,
        device=device,
        with_box_refine=True,
        as_two_stage=False,
        transformer=True,
        bbox_coder={
            "type": "CustomNMSFreeCoder",
            "post_center_range": [-20, -35, -10.0, 20, 35, 10.0],
            "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            "max_num": 100,
            "voxel_size": [0.15, 0.15, 4],
            "num_classes": 10,
        },
        num_cls_fcs=2,
        code_weights=None,
        bev_h=100,
        bev_w=100,
        fut_ts=6,
        fut_mode=6,
        map_bbox_coder={
            "type": "MapNMSFreeCoder",
            "post_center_range": [-20, -35, -20, -35, 20, 35, 20, 35],
            "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            "max_num": 50,
            "voxel_size": [0.15, 0.15, 4],
            "num_classes": 3,
        },
        map_num_query=900,
        map_num_classes=3,
        map_num_vec=100,
        map_num_pts_per_vec=20,
        map_num_pts_per_gt_vec=20,
        map_query_embed_type="instance_pts",
        map_transform_method="minmax",
        map_gt_shift_pts_pattern="v2",
        map_dir_interval=1,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        tot_epoch=12,
        use_traj_lr_warmup=False,
        motion_decoder=True,
        motion_map_decoder=True,
        use_pe=True,
        motion_det_score=None,
        map_thresh=0.5,
        dis_thresh=0.2,
        pe_normalization=True,
        ego_his_encoder=None,
        ego_fut_mode=3,
        ego_agent_decoder=True,
        ego_map_decoder=True,
        query_thresh=0.0,
        query_use_fix_pad=False,
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
    )

    mlvl_feats = []
    c = ttnn._rand([1, 6, 256, 12, 20], dtype=ttnn.bfloat16, device=device)
    mlvl_feats.append(c)
    img_metas = [
        {
            "ori_shape": [(360, 640, 3), (360, 640, 3), (360, 640, 3), (360, 640, 3), (360, 640, 3), (360, 640, 3)],
            "img_shape": [(384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3)],
            "lidar2img": [
                np.array(
                    [
                        [4.97195909e02, 3.36259809e02, 1.31050214e01, -1.41740456e02],
                        [-7.28050437e00, 2.14719425e02, -4.90215017e02, -2.57883151e02],
                        [-1.17025046e-02, 9.98471159e-01, 5.40221896e-02, -4.25203639e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [5.45978616e02, -2.47705944e02, -1.61356657e01, -1.84657143e02],
                        [1.51784935e02, 1.28122911e02, -4.95917894e02, -2.77022512e02],
                        [8.43406855e-01, 5.36312055e-01, 3.21598489e-02, -6.10371854e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [1.29479337e01, 6.01261709e02, 3.10492731e01, -1.20975154e02],
                        [-1.55728079e02, 1.28176621e02, -4.94981202e02, -2.71769902e02],
                        [-8.23415292e-01, 5.65940098e-01, 4.12196894e-02, -5.29677094e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [-3.21592898e02, -3.40289545e02, -1.05750653e01, -3.48318395e02],
                        [-4.32931264e00, -1.78114385e02, -3.25958977e02, -2.83473696e02],
                        [-8.33350064e-03, -9.99200442e-01, -3.91028008e-02, -1.01645350e00],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [-4.74626444e02, 3.69304577e02, 2.13056637e01, -2.50136476e02],
                        [-1.85050206e02, -4.10162348e01, -5.00990867e02, -2.24731382e02],
                        [-9.47586752e-01, -3.19482867e-01, 3.16948959e-03, -4.32527296e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [1.14075693e02, -5.87710608e02, -2.38253717e01, -1.09040128e02],
                        [1.77894417e02, -4.91302807e01, -5.00157067e02, -2.35298447e02],
                        [9.24052925e-01, -3.82246554e-01, -3.70989150e-03, -4.64645142e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            ],
            "pad_shape": [(384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3)],
            "scale_factor": 1.0,
            "flip": False,
            "pcd_horizontal_flip": False,
            "pcd_vertical_flip": False,
            "img_norm_cfg": {
                "mean": np.array([123.675, 116.28, 103.53], dtype=np.float32),
                "std": np.array([58.395, 57.12, 57.375], dtype=np.float32),
                "to_rgb": True,
            },
            "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
            "prev_idx": "",
            "next_idx": "3950bd41f74548429c0f7700ff3d8269",
            "pcd_scale_factor": 1.0,
            "pts_filename": "./data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin",
            "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
            "can_bus": np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.9686697,
                    -0.9686697,
                    -0.9686697,
                    -0.9686697,
                    -0.60694152,
                    -0.07634412,
                    9.87149385,
                    -0.02108691,
                    -0.01243972,
                    -0.023067,
                    8.5640597,
                    0.0,
                    0.0,
                    5.78155401,
                    0.0,
                ]
            ),
        }
    ]

    ttnn_outputs = tt_model(mlvl_feats, img_metas)
    logger.debug("TTNN model executed successfully.")
    logger.debug('input tensor shape:', c.shape)
    logger.debug('bev_embed ', ttnn_outputs["bev_embed"].shape)
    logger.debug('all_cls_scores ', ttnn_outputs["all_cls_scores"].shape)
    logger.debug('all_bbox_preds ', ttnn_outputs["all_bbox_preds"].shape)
    logger.debug('all_traj_preds ', ttnn_outputs["all_traj_preds"].shape)
    logger.debug('all_traj_cls_scores ', ttnn_outputs["all_traj_cls_scores"].shape)
    logger.debug('map_all_cls_scores ', ttnn_outputs["map_all_cls_scores"].shape)
    logger.debug('map_all_bbox_preds ', ttnn_outputs["map_all_bbox_preds"].shape)

    if (c.shape == [1, 6, 256, 12, 20] and
        ttnn_outputs["bev_embed"].shape == [10000, 1, 256] and
        ttnn_outputs["all_cls_scores"].shape == [3, 1, 300, 10] and
        ttnn_outputs["all_bbox_preds"].shape == [3, 1, 300, 10] and
        ttnn_outputs["all_traj_preds"].shape == [3, 1, 300, 6, 12] and
        ttnn_outputs["all_traj_cls_scores"].shape == [3, 1, 300, 6] and
        ttnn_outputs["map_all_cls_scores"].shape == [3, 1, 100, 3] and
        ttnn_outputs["map_all_bbox_preds"].shape == [3, 1, 100, 4]):
        logger.debug("Test passed: Output shapes are as expected.")
        return 0
    else:
        raise AssertionError("Test failed: Output shapes are not as expected.")

if __name__ == "__main__":
    test_vadv2_head()
