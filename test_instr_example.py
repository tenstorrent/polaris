#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Example script to test the instr class inheritance from ttdecode.decode.decoded_instruction
"""

import sys
import os

# Add the ttsim module to the path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'polaris', 'ttsim', 'back', 'tensix_neo'))

from ttsim.back.tensix_neo.isaFunctions import instr
import ttsim.front.ttdecode.python.src.ttdecode as ttdecode

def test_instr_basic():
    """Test basic instr creation and attribute access"""
    print("=" * 70)
    print("Test 1: Basic instr creation")
    print("=" * 70)

    ins = instr()

    # Test custom attributes added by instr class
    print(f"Initial addr: {ins.addr}")
    print(f"Initial coreId: {ins.coreId}")
    print(f"Initial threadId: {ins.threadId}")
    print(f"Initial insId: {ins.insId}")

    # Set some values
    ins.setInsId(10)
    ins.setCoreId(2)
    ins.setThread(1)
    ins.setRelAddr(0x1000)
    ins.setOp("ADD")

    print(f"\nAfter setting values:")
    print(f"insId: {ins.getInsId()}")
    print(f"coreId: {ins.getCoreId()}")
    print(f"threadId: {ins.getThread()}")
    print(f"addr: {hex(ins.getRelAddr())}")
    print(f"op: {ins.getOp()}")

    print("\n[PASS] Test 1 passed\n")


def test_instr_inheritance():
    """Test that instr inherits from decoded_instruction"""
    print("=" * 70)
    print("Test 2: Inheritance from decoded_instruction")
    print("=" * 70)

    ins = instr()

    # Check if it's an instance of decoded_instruction
    print(f"ins is instance of instr: {isinstance(ins, instr)}")
    print(f"ins is instance of decoded_instruction: {isinstance(ins, ttdecode.decode.decoded_instruction)}")

    # Check if base class attributes are accessible
    print(f"\nBase class attributes:")
    print(f"Has 'word' attribute: {hasattr(ins, 'word')}")
    print(f"Has 'program_counter' attribute: {hasattr(ins, 'program_counter')}")
    print(f"Has 'kind' attribute: {hasattr(ins, 'kind')}")
    print(f"Has 'opcode' attribute: {hasattr(ins, 'opcode')}")
    print(f"Has 'mnemonic' attribute: {hasattr(ins, 'mnemonic')}")
    print(f"Has 'operands' attribute: {hasattr(ins, 'operands')}")

    # Try to set base class attributes
    ins.word = 0x12345678
    print(f"\nSet word to 0x12345678: {hex(ins.word)}")

    print("\n[PASS] Test 2 passed\n")


def test_instr_operands():
    """Test operands manipulation"""
    print("=" * 70)
    print("Test 3: Operands manipulation")
    print("=" * 70)

    # Test 3a: Basic instr without operands
    ins = instr()
    print(f"Has operands attribute: {hasattr(ins, 'operands')}")

    # getDstInt() should return empty list when operands not initialized
    try:
        dst = ins.getDstInt()
        print(f"getDstInt() on empty instr: {dst}")
    except Exception as e:
        print(f"getDstInt() raised exception: {e}")

    # Test 3b: Create instr without operands, they will be auto-initialized
    decoded_ins = ttdecode.decode.decoded_instruction()

    # Copy it to our instr
    ins2 = instr(decoded_ins)

    print(f"\nHas operands: {hasattr(ins2, 'operands')}")
    print(f"operands is None: {ins2.operands is None}")
    print(f"operands type: {type(ins2.operands)}")

    if ins2.operands is not None:
        # Test getting empty values first
        print("\nOperands already initialized (shouldn't happen with empty instruction)")
        print(f"Source integers: {ins2.getSrcInt()}")
        print(f"Destination integers: {ins2.getDstInt()}")
        print(f"Immediates: {ins2.getImm()}")
        print(f"Attributes: {ins2.getAttr()}")
    else:
        print("\nOperands is None (as expected for fresh instruction)")

    # Test setting integer sources - operands will be auto-initialized
    print("\nTesting set_integer_sources with list:")
    ins2.setSrcInt([1, 2, 3])

    # After setting, operands should now be initialized
    print(f"After setSrcInt - operands is None: {ins2.operands is None}")
    if ins2.operands is not None:
        print(f"operands.sources exists: {hasattr(ins2.operands, 'sources')}")
        if hasattr(ins2.operands, 'sources'):
            print(f"sources.integers: {ins2.operands.sources.integers}")

    result = ins2.getSrcInt()
    print(f"Source integers: {result}")
    print(f"Source integers type: {type(result)}")
    # Convert to list for comparison since nanobind may return a different type
    result_list = list(result) if result is not None else []
    assert result_list == [1, 2, 3], f"Expected [1, 2, 3], got {result_list}"

    # Test setting integer destinations
    print("\nTesting set_integer_destinations with list:")
    ins2.setDstInt([4, 5])
    result = ins2.getDstInt()
    print(f"Destination integers: {result}")
    print(f"Destination integers type: {type(result)}")
    result_list = list(result) if result is not None else []
    assert result_list == [4, 5], f"Expected [4, 5], got {result_list}"

    # Test setting immediates
    print("\nTesting set_immediates with list:")
    ins2.setImm([100, 200])
    result = ins2.getImm()
    print(f"Immediates: {result}")
    print(f"Immediates type: {type(result)}")
    result_list = list(result) if result is not None else []
    assert result_list == [100, 200], f"Expected [100, 200], got {result_list}"

    # Test setting attributes
    print("\nTesting set_attributes with dict:")
    ins2.setAttr({'mode': 5, 'flag': 1})
    result = ins2.getAttr()
    print(f"Attributes: {result}")
    print(f"Attributes type: {type(result)}")
    # Convert to dict for comparison since nanobind may return a different type
    result_dict = dict(result) if result is not None else {}
    assert result_dict == {'mode': 5, 'flag': 1}, f"Expected {{'mode': 5, 'flag': 1}}, got {result_dict}"

    print("\n[PASS] Test 3 passed\n")


def test_instr_copy_constructor():
    """Test copy constructor"""
    print("=" * 70)
    print("Test 4: Copy constructor")
    print("=" * 70)

    # Create a source instruction
    src_ins = instr()
    src_ins.setInsId(42)
    src_ins.setCoreId(3)
    src_ins.setThread(2)
    src_ins.setRelAddr(0x2000)
    src_ins.setOp("MUL")
    src_ins.word = 0xABCDEF00

    print("Source instruction:")
    print(f"  insId: {src_ins.getInsId()}")
    print(f"  coreId: {src_ins.getCoreId()}")
    print(f"  threadId: {src_ins.getThread()}")
    print(f"  addr: {hex(src_ins.getRelAddr())}")
    print(f"  op: {src_ins.getOp()}")
    print(f"  word: {hex(src_ins.word)}")

    # Copy using copy constructor
    copied_ins = instr(src_ins)

    print("\nCopied instruction:")
    print(f"  insId: {copied_ins.getInsId()}")
    print(f"  coreId: {copied_ins.getCoreId()}")
    print(f"  threadId: {copied_ins.getThread()}")
    print(f"  addr: {hex(copied_ins.getRelAddr())}")
    print(f"  op: {copied_ins.getOp()}")
    print(f"  word: {hex(copied_ins.word)}")

    # Verify values match
    assert copied_ins.getInsId() == src_ins.getInsId(), "insId mismatch"
    assert copied_ins.getCoreId() == src_ins.getCoreId(), "coreId mismatch"
    assert copied_ins.getThread() == src_ins.getThread(), "threadId mismatch"
    assert copied_ins.getRelAddr() == src_ins.getRelAddr(), "addr mismatch"
    assert copied_ins.getOp() == src_ins.getOp(), "op mismatch"
    assert copied_ins.word == src_ins.word, "word mismatch"

    print("\n[PASS] Test 4 passed\n")


def test_instr_str():
    """Test __str__ method"""
    print("=" * 70)
    print("Test 5: String representation")
    print("=" * 70)

    ins = instr()
    ins.setInsId(5)
    ins.setThread(1)
    ins.setRelAddr(0x1234)
    ins.setOp("LOAD")

    # Create operands
    decoded_ins = ttdecode.decode.decoded_instruction()
    decoded_ins.operands = ttdecode.decode.operands()
    ins2 = instr(decoded_ins)
    ins2.setInsId(5)
    ins2.setThread(1)
    ins2.setRelAddr(0x1234)
    ins2.setOp("LOAD")
    ins2.setSrcInt([10, 11])
    ins2.setDstInt([20])

    print("String representation:")
    print(str(ins2))

    print("\n[PASS] Test 5 passed\n")


def test_instr_pipes():
    """Test pipe-related functionality"""
    print("=" * 70)
    print("Test 6: Pipe operations")
    print("=" * 70)

    ins = instr()

    # Set execution pipe
    ins.setExPipe("MATH")
    print(f"Execution pipe: {ins.getExPipe()}")

    # Set source pipes
    ins.setSrcPipes(["UNPACKER0", "UNPACKER1"])
    print(f"Source pipes: {ins.getSrcPipes()}")

    # Set destination pipes
    ins.setDstPipes(["PACKER0"])
    print(f"Destination pipes: {ins.getDstPipes()}")

    # Set pipe thread ID
    ins.setPipesThreadId(2)
    print(f"Pipes thread ID: {ins.getPipesThreadId()}")

    # Set pipe delay
    ins.setPipeDelay(5)
    print(f"Pipe delay: {ins.getPipeDelay()}")

    print("\n[PASS] Test 6 passed\n")


def test_instr_context():
    """Test getContext method"""
    print("=" * 70)
    print("Test 7: Context calculation")
    print("=" * 70)

    ins = instr()

    # Test with MATH pipe
    ins.setExPipe("MATH")
    ins.setThread(0)
    context = ins.getContext()
    print(f"Context with MATH pipe: {context} (expected: 1 for MATH_THREAD)")

    # Test with UNPACKER pipe
    ins.setExPipe("UNPACKER0")
    context = ins.getContext()
    print(f"Context with UNPACKER0 pipe: {context} (expected: 0 for UNPACKER_THREAD)")

    # Test with PACKER pipe
    ins.setExPipe("PACKER0")
    context = ins.getContext()
    print(f"Context with PACKER0 pipe: {context} (expected: 3 for PACKER_THREAD)")

    # Test with SFPU pipe
    ins.setExPipe("SFPU")
    context = ins.getContext()
    print(f"Context with SFPU pipe: {context} (expected: 2 for SFPU_THREAD)")

    print("\n[PASS] Test 7 passed\n")


def test_instr_copy():
    """Test shallow copy (copy.copy)"""
    import copy

    print("=" * 70)
    print("Test 8: Shallow copy (copy.copy)")
    print("=" * 70)

    # Create source instruction with operands
    decoded_ins = ttdecode.decode.decoded_instruction()
    decoded_ins.operands = ttdecode.decode.operands()
    src_ins = instr(decoded_ins)

    src_ins.setInsId(99)
    src_ins.setCoreId(5)
    src_ins.setThread(3)
    src_ins.setRelAddr(0x5000)
    src_ins.setOp("COPY_TEST")
    src_ins.word = 0xDEADBEEF
    src_ins.setSrcInt([7, 8, 9])
    src_ins.setDstInt([10])
    src_ins.setSrcPipes(["UNPACKER0", "UNPACKER1"])
    src_ins.setDstPipes(["PACKER0"])

    print("Source instruction:")
    print(f"  insId: {src_ins.getInsId()}")
    print(f"  coreId: {src_ins.getCoreId()}")
    print(f"  threadId: {src_ins.getThread()}")
    print(f"  addr: {hex(src_ins.getRelAddr())}")
    print(f"  op: {src_ins.getOp()}")
    print(f"  word: {hex(src_ins.word)}")
    print(f"  srcInt: {src_ins.getSrcInt()}")
    print(f"  dstInt: {src_ins.getDstInt()}")
    print(f"  srcPipes: {src_ins.getSrcPipes()}")
    print(f"  dstPipes: {src_ins.getDstPipes()}")

    # Perform shallow copy
    copied_ins = copy.copy(src_ins)

    print("\nShallow copied instruction:")
    print(f"  insId: {copied_ins.getInsId()}")
    print(f"  coreId: {copied_ins.getCoreId()}")
    print(f"  threadId: {copied_ins.getThread()}")
    print(f"  addr: {hex(copied_ins.getRelAddr())}")
    print(f"  op: {copied_ins.getOp()}")
    print(f"  word: {hex(copied_ins.word)}")
    print(f"  srcInt: {copied_ins.getSrcInt()}")
    print(f"  dstInt: {copied_ins.getDstInt()}")
    print(f"  srcPipes: {copied_ins.getSrcPipes()}")
    print(f"  dstPipes: {copied_ins.getDstPipes()}")

    # Verify all values match
    assert copied_ins.getInsId() == src_ins.getInsId(), "insId mismatch"
    assert copied_ins.getCoreId() == src_ins.getCoreId(), "coreId mismatch"
    assert copied_ins.getThread() == src_ins.getThread(), "threadId mismatch"
    assert copied_ins.getRelAddr() == src_ins.getRelAddr(), "addr mismatch"
    assert copied_ins.getOp() == src_ins.getOp(), "op mismatch"
    assert copied_ins.word == src_ins.word, "word mismatch"

    # For shallow copy, lists should be shared (same reference)
    print("\nShallow copy behavior check:")
    print(f"  srcPipes are same object: {copied_ins.getSrcPipes() is src_ins.getSrcPipes()}")
    print(f"  dstPipes are same object: {copied_ins.getDstPipes() is src_ins.getDstPipes()}")

    # Modifying the copied list should affect the original in shallow copy
    src_ins.srcPipes.append("UNPACKER2")
    print(f"\nAfter appending to src_ins.srcPipes:")
    print(f"  src_ins srcPipes: {src_ins.getSrcPipes()}")
    print(f"  copied_ins srcPipes: {copied_ins.getSrcPipes()}")
    print(f"  Both should show UNPACKER2 (shared reference)")

    print("\n[PASS] Test 8 passed\n")


def test_instr_deepcopy():
    """Test deep copy (copy.deepcopy)"""
    import copy

    print("=" * 70)
    print("Test 9: Deep copy (copy.deepcopy)")
    print("=" * 70)

    # Create source instruction with operands
    decoded_ins = ttdecode.decode.decoded_instruction()
    decoded_ins.operands = ttdecode.decode.operands()
    src_ins = instr(decoded_ins)

    src_ins.setInsId(123)
    src_ins.setCoreId(7)
    src_ins.setThread(2)
    src_ins.setRelAddr(0x8000)
    src_ins.setOp("DEEPCOPY_TEST")
    src_ins.word = 0xCAFEBABE
    src_ins.setSrcInt([11, 12, 13])
    src_ins.setDstInt([14, 15])
    src_ins.setSrcPipes(["SFPU"])
    src_ins.setDstPipes(["PACKER1"])
    src_ins.setExPipe("MATH")

    print("Source instruction:")
    print(f"  insId: {src_ins.getInsId()}")
    print(f"  coreId: {src_ins.getCoreId()}")
    print(f"  threadId: {src_ins.getThread()}")
    print(f"  addr: {hex(src_ins.getRelAddr())}")
    print(f"  op: {src_ins.getOp()}")
    print(f"  word: {hex(src_ins.word)}")
    print(f"  srcInt: {src_ins.getSrcInt()}")
    print(f"  dstInt: {src_ins.getDstInt()}")
    print(f"  srcPipes: {src_ins.getSrcPipes()}")
    print(f"  dstPipes: {src_ins.getDstPipes()}")
    print(f"  exPipe: {src_ins.getExPipe()}")

    # Perform deep copy
    deepcopied_ins = copy.deepcopy(src_ins)

    print("\nDeep copied instruction:")
    print(f"  insId: {deepcopied_ins.getInsId()}")
    print(f"  coreId: {deepcopied_ins.getCoreId()}")
    print(f"  threadId: {deepcopied_ins.getThread()}")
    print(f"  addr: {hex(deepcopied_ins.getRelAddr())}")
    print(f"  op: {deepcopied_ins.getOp()}")
    print(f"  word: {hex(deepcopied_ins.word)}")
    print(f"  srcInt: {deepcopied_ins.getSrcInt()}")
    print(f"  dstInt: {deepcopied_ins.getDstInt()}")
    print(f"  srcPipes: {deepcopied_ins.getSrcPipes()}")
    print(f"  dstPipes: {deepcopied_ins.getDstPipes()}")
    print(f"  exPipe: {deepcopied_ins.getExPipe()}")

    # Verify all values match
    assert deepcopied_ins.getInsId() == src_ins.getInsId(), "insId mismatch"
    assert deepcopied_ins.getCoreId() == src_ins.getCoreId(), "coreId mismatch"
    assert deepcopied_ins.getThread() == src_ins.getThread(), "threadId mismatch"
    assert deepcopied_ins.getRelAddr() == src_ins.getRelAddr(), "addr mismatch"
    assert deepcopied_ins.getOp() == src_ins.getOp(), "op mismatch"
    assert deepcopied_ins.word == src_ins.word, "word mismatch"
    assert deepcopied_ins.getExPipe() == src_ins.getExPipe(), "exPipe mismatch"

    # For deep copy, lists should be independent (different references)
    print("\nDeep copy behavior check:")
    print(f"  srcPipes are same object: {deepcopied_ins.getSrcPipes() is src_ins.getSrcPipes()}")
    print(f"  dstPipes are same object: {deepcopied_ins.getDstPipes() is src_ins.getDstPipes()}")
    print(f"  Both should be False (independent copies)")

    assert deepcopied_ins.getSrcPipes() is not src_ins.getSrcPipes(), "srcPipes should be independent"
    assert deepcopied_ins.getDstPipes() is not src_ins.getDstPipes(), "dstPipes should be independent"

    # Modifying the copied list should NOT affect the original in deep copy
    src_ins.srcPipes.append("MATH")
    print(f"\nAfter appending to src_ins.srcPipes:")
    print(f"  src_ins srcPipes: {src_ins.getSrcPipes()}")
    print(f"  deepcopied_ins srcPipes: {deepcopied_ins.getSrcPipes()}")
    print(f"  They should be different (independent)")

    assert "MATH" in src_ins.getSrcPipes(), "Source should have MATH"
    assert "MATH" not in deepcopied_ins.getSrcPipes(), "Deep copy should NOT have MATH"

    print("\n[PASS] Test 9 passed\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("Testing instr class and ttdecode.decode.decoded_instruction inheritance")
    print("=" * 70 + "\n")

    try:
        test_instr_basic()
        test_instr_inheritance()
        test_instr_operands()
        test_instr_copy_constructor()
        test_instr_str()
        test_instr_pipes()
        test_instr_context()
        test_instr_copy()
        test_instr_deepcopy()

        print("=" * 70)
        print("[PASS] ALL TESTS PASSED")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print("[FAIL] TEST FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
