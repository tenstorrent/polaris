#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BEVFormer Test Script
Comprehensive test and validation script for BEVFormer implementation in Polaris
"""

import pytest
from ttsim.utils.common import parse_yaml

from workloads.bevformer.BEVFormer import BEVFormer


@pytest.fixture
def bevformer():
    """Fixture to construct a BEVFormer model with input tensors."""
    bevformer_cfg = {
        'bs': 1,
        'num_sweeps': 1,
        'num_cameras': 6,
        'img_channels': 3,
        'img_height': 256,
        'img_width': 704,
        'video_test_mode': True,
        'backbone_conf_yaml': 'config/bevformer_cfgs/bevformer_backbone.yaml',
        'transformer_conf_yaml': 'config/bevformer_cfgs/bevformer_transformer.yaml',
        'head_conf_yaml': 'config/bevformer_cfgs/bevformer_head.yaml'
    }
    model = BEVFormer('bevformer', bevformer_cfg)
    model.create_input_tensors()
    return model


def test_bevformer_creation():
    """Test BEVFormer model creation and basic functionality"""
    print("=" * 60)
    print("Testing BEVFormer Model Creation")
    print("=" * 60)

    # BEVFormer configuration
    bevformer_cfg = {
        'bs': 1,
        'num_sweeps': 1,
        'num_cameras': 6,
        'img_channels': 3,
        'img_height': 256,
        'img_width': 704,
        'video_test_mode': True,
        'backbone_conf_yaml': 'config/bevformer_cfgs/bevformer_backbone.yaml',
        'transformer_conf_yaml': 'config/bevformer_cfgs/bevformer_transformer.yaml',
        'head_conf_yaml': 'config/bevformer_cfgs/bevformer_head.yaml'
    }

    # Create BEVFormer model
    bevformer = BEVFormer('bevformer', bevformer_cfg)
    print("✓ BEVFormer model created successfully!")

    # Create input tensors
    bevformer.create_input_tensors()
    print("✓ Input tensors created successfully!")

    # Print model structure
    print("\nModel Structure:")
    print(f"- Batch size: {bevformer.bs}")
    print(f"- Number of sweeps: {bevformer.num_sweeps}")
    print(f"- Number of cameras: {bevformer.num_cameras}")
    print(f"- Image size: {bevformer.img_height}x{bevformer.img_width}")
    print(f"- Video test mode: {bevformer.video_test_mode}")

    assert bevformer is not None


def test_bevformer_inference(bevformer):
    """Test BEVFormer inference with dummy data"""
    print("\n" + "=" * 60)
    print("Testing BEVFormer Inference")
    print("=" * 60)

    if bevformer is None:
        print("✗ No BEVFormer model available for inference test")
        return False

    try:
        # Use the model's input tensors instead of creating numpy arrays
        # This ensures compatibility with Polaris tensor operations
        input_tensors = bevformer.input_tensors

        # Create dummy metadata
        img_metas = [{
            'scene_token': 'test_scene_001',
            'can_bus': [0.0] * 18,  # dummy can_bus data
            'prev_bev_exists': False
        } for _ in range(bevformer.bs)]

        print("✓ Using model's input tensors for inference")

        # Test inference
        print("Running inference...")
        outputs = bevformer(
            sweep_imgs=input_tensors['sweep_imgs'],
            sensor2ego_mats=input_tensors['sensor2ego_mats'],
            intrin_mats=input_tensors['intrin_mats'],
            ida_mats=input_tensors['ida_mats'],
            sensor2sensor_mats=input_tensors['sensor2sensor_mats'],
            bda_mat=input_tensors['bda_mat'],
            can_bus=input_tensors['can_bus'],
            img_metas=img_metas
        )

        print("✓ Inference completed successfully!")
        print(f"Output type: {type(outputs)}")
        print(f"Number of batch outputs: {len(outputs)}")

        assert isinstance(outputs, (dict, list))
        return None

    except Exception as e:
        print(f"✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"inference failed: {e}"


def test_bevformer_components(bevformer):
    """Test individual BEVFormer components"""
    print("\n" + "=" * 60)
    print("Testing BEVFormer Components")
    print("=" * 60)

    if bevformer is None:
        print("✗ No BEVFormer model available for component testing")
        return False

    try:
        # Test basic component access
        print("Testing component access...")

        # Test backbone exists
        if hasattr(bevformer, 'backbone'):
            print("✓ Backbone component accessible")
        else:
            print("✗ Backbone component not found")

        # Test transformer exists
        if hasattr(bevformer.pts_bbox_head, 'transformer'):
            transformer = bevformer.pts_bbox_head.transformer
            print("✓ Transformer component accessible")
            print(f"  - embed_dims: {transformer.embed_dims}")
            print(f"  - bev_size: {transformer.bev_h}x{transformer.bev_w}")
            if hasattr(transformer, 'encoder'):
                print(f"  - encoder_layers: {transformer.encoder.num_layers}")
        else:
            print("✗ Transformer component not found")

        # Test head exists
        if hasattr(bevformer, 'pts_bbox_head'):
            print("✓ Detection head component accessible")
        else:
            print("✗ Detection head component not found")

        print("✓ All components are accessible")
        assert hasattr(bevformer, 'backbone')
        assert hasattr(bevformer, 'pts_bbox_head')
        assert hasattr(bevformer.pts_bbox_head, 'transformer')
        return None

    except Exception as e:
        print(f"✗ Error testing components: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"component checks failed: {e}"


def test_configuration_loading():
    """Test loading and parsing of configuration files"""
    print("\n" + "=" * 60)
    print("Testing Configuration Loading")
    print("=" * 60)

    try:
        # Test loading main configuration
        main_cfg = parse_yaml('config/bevformer_cfgs/bevformer_base.yaml')
        print("✓ Main configuration loaded")

        # Test loading component configurations
        backbone_cfg = parse_yaml('config/bevformer_cfgs/bevformer_backbone.yaml')
        print("✓ Backbone configuration loaded")

        transformer_cfg = parse_yaml('config/bevformer_cfgs/bevformer_transformer.yaml')
        print("✓ Transformer configuration loaded")

        head_cfg = parse_yaml('config/bevformer_cfgs/bevformer_head.yaml')
        print("✓ Head configuration loaded")

        # Print some key configuration values
        print("\nConfiguration Summary:")
        print(f"- Model: BEVFormer-base")
        print(f"- Batch size: {main_cfg.get('bs', 'N/A')}")
        print(f"- Cameras: {main_cfg.get('num_cameras', 'N/A')}")
        print(f"- Image size: {main_cfg.get('img_height', 'N/A')}x{main_cfg.get('img_width', 'N/A')}")
        classes_count = len(main_cfg.get('class_names', []))
        print(f"- Classes: {classes_count}")

        assert isinstance(main_cfg, dict)
        assert isinstance(backbone_cfg, dict)
        assert isinstance(transformer_cfg, dict)
        assert isinstance(head_cfg, dict)
        return None

    except Exception as e:
        print(f"✗ Error loading configurations: {e}")
        assert False, f"config load failed: {e}"


def main():
    """Main test function"""
    print("BEVFormer Polaris Implementation - Test Suite")
    print("Testing camera-only 3D object detection model")
    print()

    # Test configuration loading
    config_ok = test_configuration_loading()

    if not config_ok:
        print("\n✗ Configuration loading failed. Aborting tests.")
        return

    # Test model creation
    bevformer = test_bevformer_creation()

    if bevformer is None:
        print("\n✗ Model creation failed. Aborting remaining tests.")
        return

    # Test components
    components_ok = test_bevformer_components(bevformer)

    # Test inference
    inference_ok = test_bevformer_inference(bevformer)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Configuration Loading: {'✓ PASS' if config_ok else '✗ FAIL'}")
    print(f"Model Creation: {'✓ PASS' if bevformer is not None else '✗ FAIL'}")
    print(f"Component Testing: {'✓ PASS' if components_ok else '✗ FAIL'}")
    print(f"Inference Testing: {'✓ PASS' if inference_ok else '✗ FAIL'}")

    all_passed = all([config_ok, bevformer is not None, components_ok, inference_ok])
    print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    if all_passed:
        print("\n🎉 BEVFormer implementation is ready for use!")
        print("The model can now be integrated into the Polaris framework for")
        print("camera-only 3D object detection tasks.")
    else:
        print("\n⚠️  Some tests failed. Please review the implementation and fix issues.")


if __name__ == '__main__':
    main()
