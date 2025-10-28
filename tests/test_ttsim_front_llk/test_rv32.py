#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import typing

import pytest

import ttsim.front.llk.decoded_instruction as decoded_instruction
import ttsim.front.llk.rv32 as rv32

def test_instruction_kind():
    assert decoded_instruction.instruction_kind.rv32 == rv32.instruction_kind()

def test_get_default_instruction_set():
    ins_set = rv32.get_default_instruction_set()
    assert all(isinstance(key, str) for key in ins_set.keys())

def test_is_valid_instruction():
    assert False == rv32.is_valid_instruction(0)
    assert True  == rv32.is_valid_instruction(0b11)
    assert False == rv32.is_valid_instruction(0b11, is_swizzled = False)

    with pytest.raises(Exception):
        rv32.is_valid_instruction("") # type: ignore[arg-type]

    with pytest.raises(Exception):
        rv32.is_valid_instruction(0, is_swizzled = "True") # type: ignore[arg-type]

def test_get_operands():
    arguments: dict[str, int] = dict()
    arguments["rs0"]  = 0
    arguments["rs1"]  = 1
    arguments["frs1"] = 2
    arguments["frs2"] = 3
    arguments["imm"]  = -5
    arguments["rd"]   = 6
    arguments["frd"]  = 7
    arguments["abc"]  = 8

    operands = rv32.get_operands(arguments)
    assert hasattr(operands, 'all')
    assert hasattr(operands, 'sources')
    assert hasattr(operands, 'destinations')
    assert hasattr(operands, 'immediates')
    assert hasattr(operands, 'attributes')
    assert hasattr(operands.sources, 'integers')
    assert hasattr(operands.sources, 'floats')
    assert hasattr(operands.destinations, 'integers')
    assert hasattr(operands.destinations, 'floats')
    assert sorted(arguments.keys()) == sorted(operands.all.keys())
    assert all(operands.all[key] == arguments[key] for key in arguments.keys())
    expected: typing.Any = [arguments["rs0"],arguments["rs1"]]
    assert expected == operands.sources.integers
    expected = [arguments["frs1"],arguments["frs2"]]
    assert expected == operands.sources.floats
    expected = [arguments["rd"]]
    assert expected == operands.destinations.integers
    expected = [arguments["frd"]]
    assert expected == operands.destinations.floats
    expected = [arguments["imm"]]
    assert expected == operands.immediates
    expected = {"abc" : arguments["abc"]}
    assert expected == operands.attributes

def test_get_instruction_set():
    def_ins_set = rv32.get_default_instruction_set()
    assert rv32.get_instruction_set() == def_ins_set
    assert rv32.get_instruction_set(decoded_instruction.instruction_kind.rv32) == def_ins_set
    assert rv32.get_instruction_set(decoded_instruction.get_default_instruction_set_file_name(decoded_instruction.instruction_kind.rv32)) == def_ins_set
    assert rv32.get_instruction_set(def_ins_set) == def_ins_set
    assert rv32.get_instruction_set(dict()) == dict()
    assert rv32.get_instruction_set(dict({decoded_instruction.instruction_kind.rv32 : dict()})) == dict() # type: ignore[arg-type]

    with pytest.raises(Exception):
        rv32.get_instruction_set(dict({decoded_instruction.instruction_kind.ttwh : dict()})) # type: ignore[arg-type]

    with pytest.raises(Exception):
        rv32.get_instruction_set(decoded_instruction.instruction_kind.ttbh)

    with pytest.raises(Exception):
        rv32.get_instruction_set([0]) # type: ignore[arg-type]

@pytest.mark.slow
def test_decode_instruction():
    # TODO: Add all riscv 32 bit instructions
    # decode all RV32 instructions.
    # 32 bit instruction (input), its assembly str (expected)

    words_asms = {
        0b00000000100001111010011110000011 : "0x0087a783 lw x15, 8(x15)",
        0x1b78933 : "0x01b78933 add x18, x15, x27",
        0xbb218693 : "0xbb218693 addi x13, x3, -1102",
        0x64158d9b : "0x64158d9b addiw x27, x11, 1601",
        0x23083b : "0x0023083b addw x16, x6, x2",
        0x19bf333 : "0x019bf333 and x6, x23, x25",
        0xd207513 : "0x0d207513 andi x10, x0, 210",
        0x38a20497 : "0x38a20497 auipc x9, 231968",
        0xf6e03e3 : "0x0f6e03e3 beq x28, x22, 2278",
        0x2af3d163 : "0x2af3d163 bge x7, x15, 674",
        0x6ea9fbe3 : "0x6ea9fbe3 bgeu x19, x10, 3830",
        0xe00c4663 : "0xe00c4663 blt x24, x0, -2548",
        0xac4e6f63 : "0xac4e6f63 bltu x28, x4, -3362",
        0x18f49763 : "0x18f49763 bne x9, x15, 398",
        0xf25cb5f3 : "0xf25cb5f3 csrrc x11, 0xf25, x25",
        0x9b277f3 : "0x09b277f3 csrrci x15, 0x9b, 4",
        0x155ca0f3 : "0x155ca0f3 csrrs x1, 0x155, x25",
        0xe9c3e773 : "0xe9c3e773 csrrsi x14, 0xe9c, 7",
        0x3ad1e73 : "0x03ad1e73 csrrw x28, 0x3a, x26",
        0x1e875ff3 : "0x1e875ff3 csrrwi x31, 0x1e8, 14",
        0x219c2b3 : "0x0219c2b3 div x5, x19, x1",
        0x3b4d433 : "0x03b4d433 divu x8, x9, x27",
        0x2db533b : "0x02db533b divuw x6, x22, x13",
        0x2e1c1bb : "0x02e1c1bb divw x3, x3, x14",
        0x100073 : "0x00100073 ebreak",
        0x73 : "0x00000073 ecall",
        # 0x269ce53 : "0x0269ce53 fadd.d f28, f19, f6", # rounding mode missing.
        # 0x601a3d3 : "0x0601a3d3 fadd.q f7, f3, f0", # rounding mode missing.
        # 0x1e1d453 : "0x01e1d453 fadd.s f8, f3, f30", # https://luplab.gitlab.io/rvcodecjs/#q=0x1e1d453 error.
        0xe20e1453 : "0xe20e1453 fclass.d x8, f28",
        0xe60b1ed3 : "0xe60b1ed3 fclass.q x29, f22",
        0xe0071253 : "0xe0071253 fclass.s x4, f14",
        # 0xd22a9ad3 : "0xd22a9ad3 fcvt.d.l f21, x21", # rounding mode missing
        # 0xd23a9cd3 : "0xd23a9cd3 fcvt.d.lu f25, x21", # rounding mode missing
        # 0x423d9e53 : "0x423d9e53 fcvt.d.q f28, f27", # rounding mode missing
        # 0x42083e53 : "0x42083e53 fcvt.d.s f28, f16", # rounding mode missing
        # 0xd203cdd3 : "0xd203cdd3 fcvt.d.w f27, x7", # rounding mode missing
        # 0xd21db5d3 : "0xd21db5d3 fcvt.d.wu f11, x27", # rounding mode missing
        # 0xc227c253 : "0xc227c253 fcvt.l.d x4, f15", # rounding mode missing
        # 0xc62db053 : "0xc62db053 fcvt.l.q x0, f27", # rounding mode missing
        # 0xc02a9953 : "0xc02a9953 fcvt.l.s x18, f21", # rounding mode missing
        # 0xc23be3d3 : "0xc23be3d3 fcvt.lu.d x7, f23", # https://luplab.gitlab.io/rvcodecjs/#q=0xc23be3d3 rounding mode error
        # 0xc6354ed3 : "0xc6354ed3 fcvt.lu.q x29, f10", # rounding mode missing
        # 0xc03c4053 : "0xc03c4053 fcvt.lu.s x0, f24", # rounding mode missing
        # 0x4610bcd3 : "0x4610bcd3 fcvt.q.d f25, f1", # rounding mode missing
        # 0xd623add3 : "0xd623add3 fcvt.q.l f27, x7", # rounding mode missing
        # 0xd63520d3 : "0xd63520d3 fcvt.q.lu f1, x10", # rounding mode missing
        # 0x46023b53 : "0x46023b53 fcvt.q.s f22, f4", # rounding mode missing
        # 0xd60eecd3 : "0xd60eecd3 fcvt.q.w f25, x29", # https://luplab.gitlab.io/rvcodecjs/#q=0xd60eecd3 invalid float rounding mode field
        # 0xd6101053 : "0xd6101053 fcvt.q.wu f0, x0", # rounding mode missing
        0x4012fa53 : "0x4012fa53 fcvt.s.d f20, f5",
        # 0xd025b3d3 : "0xd025b3d3 fcvt.s.l f7, x11", # rounding mode missing
        0xd036fa53 : "0xd036fa53 fcvt.s.lu f20, x13",
        # 0x40326fd3 : "0x40326fd3 fcvt.s.q f31, f4", # error invalid rouding mode field
        # 0xd000d353 : "0xd000d353 fcvt.s.w f6, x1", # error invalid rouding mode field
        # 0xd011c7d3 : "0xd011c7d3 fcvt.s.wu f15, x3", # rounding mode missing
        # 0xc20fae53 : "0xc20fae53 fcvt.w.d x28, f31", # rounding mode missing
        # 0xc6051d53 : "0xc6051d53 fcvt.w.q x26, f10", # rounding mode missing
        # 0xc00faad3 : "0xc00faad3 fcvt.w.s x21, f31", # rounding mode missing
        # 0xc21bd6d3 : "0xc21bd6d3 fcvt.wu.d x13, f23", # rounding mode missing
        # 0xc616e853 : "0xc616e853 fcvt.wu.q x16, f13", # rounding mode missing
        # 0xc01a0ed3 : "0xc01a0ed3 fcvt.wu.s x29, f20", # rounding mode missing
        # 0x1ab42ad3 : "0x1ab42ad3 fdiv.d f21, f8, f11", # rounding mode missing
        # 0x1f235f53 : "0x1f235f53 fdiv.q f30, f6, f18", # error
        # 0x193a84d3 : "0x193a84d3 fdiv.s f9, f21, f19", # rounding mode missing
        # 0xfc828c8f : "0xfc828c8f fence io, i", # error.
        # 0x34031c8f : "0x34031c8f fence.i", # error.
        0xa2512153 : "0xa2512153 feq.d x2, f2, f5",
        0xa7bb2b53 : "0xa7bb2b53 feq.q x22, f22, f27",
        0xa0b6aa53 : "0xa0b6aa53 feq.s x20, f13, f11",
        0xb9bb3487 : "0xb9bb3487 fld f9, -1125(x22)",
        0xa37e8153 : "0xa37e8153 fle.d x2, f29, f23",
        0xa7c68dd3 : "0xa7c68dd3 fle.q x27, f13, f28",
        0xa1e38053 : "0xa1e38053 fle.s x0, f7, f30",
        0x54e5ce87 : "0x54e5ce87 flq f29, 1358(x11)",
        0xa3351753 : "0xa3351753 flt.d x14, f10, f19",
        0xa7d819d3 : "0xa7d819d3 flt.q x19, f16, f29",
        0xa1369d53 : "0xa1369d53 flt.s x26, f13, f19",
        0xf73baa87 : "0xf73baa87 flw f21, -141(x23)",
        # 0x63e71bc3 : "0x63e71bc3 fmadd.d f23, f14, f30, f12", # rounding mode missing
        # 0x16eae9c3 : "0x16eae9c3 fmadd.q f19, f21, f14, f2", # error
        # 0x20a4bdc3 : "0x20a4bdc3 fmadd.s f27, f9, f10, f4", # rounding mode missing
        0x2b5716d3 : "0x2b5716d3 fmax.d f13, f14, f21",
        0x2e8699d3 : "0x2e8699d3 fmax.q f19, f13, f8",
        0x28aa9653 : "0x28aa9653 fmax.s f12, f21, f10",
        0x2a0c89d3 : "0x2a0c89d3 fmin.d f19, f25, f0",
        0x2e4183d3 : "0x2e4183d3 fmin.q f7, f3, f4",
        0x299086d3 : "0x299086d3 fmin.s f13, f1, f25",
        0xda5b7247 : "0xda5b7247 fmsub.d f4, f22, f5, f27",
        # 0x5640d6c7 : "0x5640d6c7 fmsub.q f13, f1, f4, f10", # error
        # 0xb1c42dc7 : "0xb1c42dc7 fmsub.s f27, f8, f28, f22", # rounding mode missing
        # 0x12c6c753 : "0x12c6c753 fmul.d f14, f13, f12", # rounding mode missing
        # 0x1638ee53 : "0x1638ee53 fmul.q f28, f17, f3", # error
        # 0x107098d3 : "0x107098d3 fmul.s f17, f1, f7", # rouding mode missing
        0xf20002d3 : "0xf20002d3 fmv.d.x f5, x0",
        0xf0080fd3 : "0xf0080fd3 fmv.w.x f31, x16",
        0xe2008453 : "0xe2008453 fmv.x.d x8, f1",
        0xe00a07d3 : "0xe00a07d3 fmv.x.w x15, f20",
        0xfa72f3cf : "0xfa72f3cf fnmadd.d f7, f5, f7, f31",
        # 0x1769224f : "0x1769224f fnmadd.q f4, f18, f22, f2", # rounding mode missing
        # 0x71a797cf : "0x71a797cf fnmadd.s f15, f15, f26, f14", # rounding mode missing
        # 0xbb2874b : "0x0bb2874b fnmsub.d f14, f5, f27, f1", # rounding mode missing
        # 0xbf0824cb : "0xbf0824cb fnmsub.q f9, f16, f16, f23", # rounding mode missing
        # 0xf9c3e04b : "0xf9c3e04b fnmsub.s f0, f7, f28, f31", # error
        0xf2f73a27 : "0xf2f73a27 fsd f15, -204(x14)",
        0x221a87d3 : "0x221a87d3 fsgnj.d f15, f21, f1",
        0x277e8a53 : "0x277e8a53 fsgnj.q f20, f29, f23",
        0x215209d3 : "0x215209d3 fsgnj.s f19, f4, f21",
        0x234d90d3 : "0x234d90d3 fsgnjn.d f1, f27, f20",
        0x27f010d3 : "0x27f010d3 fsgnjn.q f1, f0, f31",
        0x20a31853 : "0x20a31853 fsgnjn.s f16, f6, f10",
        0x22b22b53 : "0x22b22b53 fsgnjx.d f22, f4, f11",
        0x277e2dd3 : "0x277e2dd3 fsgnjx.q f27, f28, f23",
        0x204d2c53 : "0x204d2c53 fsgnjx.s f24, f26, f4",
        0xc10047a7 : "0xc10047a7 fsq f16, -1009(x0)",
        # 0x5a019c53 : "0x5a019c53 fsqrt.d f24, f3", # rounding mode missing
        # 0x5e0cab53 : "0x5e0cab53 fsqrt.q f22, f25", # rounding mode missing
        # 0x580e12d3 : "0x580e12d3 fsqrt.s f5, f28", # rounding mode missing
        # 0xb30bb53 : "0x0b30bb53 fsub.d f22, f1, f19", # rounding mode missing
        0xe0a7453 : "0x0e0a7453 fsub.q f8, f20, f0",
        # 0x9f10dd3 : "0x09f10dd3 fsub.s f27, f2, f31", # rounding mode missing
        0xacdaaea7 : "0xacdaaea7 fsw f13, -1315(x21)",
        0xc6f68e6f : "0xc6f68e6f jal x28, -619410",
        0xdc6388e7 : "0xdc6388e7 jalr x17, -570(x7)",
        0xb5580983 : "0xb5580983 lb x19, -1195(x16)",
        0xa1b94803 : "0xa1b94803 lbu x16, -1509(x18)",
        0x8acd3b03 : "0x8acd3b03 ld x22, -1876(x26)",
        0x7fa61383 : "0x7fa61383 lh x7, 2042(x12)",
        0xf320dc83 : "0xf320dc83 lhu x25, -206(x1)",
        0x78f34bb7 : "0x78f34bb7 lui x23, 495412",
        0x7fbea483 : "0x7fbea483 lw x9, 2043(x29)",
        0x25b6383 : "0x025b6383 lwu x7, 37(x22)",
        0x31783b3 : "0x031783b3 mul x7, x15, x17",
        0x2d29d33 : "0x02d29d33 mulh x26, x5, x13",
        0x2012033 : "0x02012033 mulhsu x0, x2, x0",
        0x2c33333 : "0x02c33333 mulhu x6, x6, x12",
        0x3838f3b : "0x03838f3b mulw x30, x7, x24",
        0x10edb3 : "0x0010edb3 or x27, x1, x1",
        0x69866b93 : "0x69866b93 ori x23, x12, 1688",
        0x3a565b3 : "0x03a565b3 rem x11, x10, x26",
        0x26cf133 : "0x026cf133 remu x2, x25, x6",
        0x25e79bb : "0x025e79bb remuw x19, x28, x5",
        0x35e6e3b : "0x035e6e3b remw x28, x28, x21",
        0x454f8ea3 : "0x454f8ea3 sb x20, 1117(x31)",
        0xaf393fa3 : "0xaf393fa3 sd x19, -1281(x18)",
        0x688b93a3 : "0x688b93a3 sh x8, 1671(x23)",
        0x1781c33 : "0x01781c33 sll x24, x16, x23",
        0x869593 : "0x00869593 slli x11, x13, 8",
        0x1c71f13 : "0x01c71f13 slli x30, x14, 28",
        0xb29f9b : "0x00b29f9b slliw x31, x5, 11",
        0x9299bb : "0x009299bb sllw x19, x5, x9",
        0x6fa233 : "0x006fa233 slt x4, x31, x6",
        0xe9202513 : "0xe9202513 slti x10, x0, -366",
        0x6ecc3f13 : "0x6ecc3f13 sltiu x30, x24, 1772",
        0x14832b3 : "0x014832b3 sltu x5, x16, x20",
        0x411fd233 : "0x411fd233 sra x4, x31, x17",
        0x403fd313 : "0x403fd313 srai x6, x31, 3",
        0x40165813 : "0x40165813 srai x16, x12, 1",
        0x4174509b : "0x4174509b sraiw x1, x8, 23",
        0x41ffdfbb : "0x41ffdfbb sraw x31, x31, x31",
        0x199d833 : "0x0199d833 srl x16, x19, x25",
        0x1415013 : "0x01415013 srli x0, x2, 20",
        0x10dd21b : "0x010dd21b srliw x4, x27, 16",
        0x16ad5bb : "0x016ad5bb srlw x11, x21, x22",
        0x41388eb3 : "0x41388eb3 sub x29, x17, x19",
        0x415e0b3b : "0x415e0b3b subw x22, x28, x21",
        0x15b12423 : "0x15b12423 sw x27, 328(x2)",
        0x1edcab3 : "0x01edcab3 xor x21, x27, x30",
        0x64d5c613 : "0x64d5c613 xori x12, x11, 1613"
    }

    instruction_set = rv32.get_default_instruction_set()
    kinds_instruction_sets = dict()
    kinds_instruction_sets[rv32.instruction_kind()] = instruction_set
    instruction_set_file_name = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "ttsim", "config", "llk", "instruction_sets", f"{rv32.instruction_kind()}", "assembly.yaml"))

    for word, expected_asm in words_asms.items():
        asms = set()
        asms.add(rv32.to_assembly(rv32.decode_instruction(word)))
        asms.add(rv32.to_assembly(rv32.decode_instruction(word, instruction_set = decoded_instruction.instruction_kind.rv32)))
        asms.add(rv32.to_assembly(rv32.decode_instruction(word, instruction_set = instruction_set_file_name)))
        asms.add(rv32.to_assembly(rv32.decode_instruction(word, instruction_set = instruction_set)))
        asms.add(rv32.to_assembly(rv32.decode_instruction(word, instruction_set = kinds_instruction_sets)))

        if 1 != len(asms):
            msg = f"- error: expected all instruction decodes to be identical, received either none or multiple (len(asms) = {len(asms)}).\n"
            for ele in asms:
                msg += "  " + ele + "\n"

            raise Exception(msg)

        assert (list(asms)[0] == expected_asm)


    # test exceptions:
    for word, expected_asm in words_asms.items():
        with pytest.raises(Exception) as exc_info:
            rv32.decode_instruction(word, instruction_set = decoded_instruction.instruction_kind.ttwh)
        assert f"- error: expected instruction kind to be {rv32.instruction_kind()}, received " in str(exc_info.value)

        with pytest.raises(Exception) as exc_info:
            rv32.decode_instruction(word, instruction_set = {decoded_instruction.instruction_kind.ttwh : None})
        assert f"error: could not find instruction set associated with instruction kind {rv32.instruction_kind()}" in str(exc_info.value)

        with pytest.raises(Exception) as exc_info:
            rv32.decode_instruction(word, instruction_set = typing.cast(typing.Any, list()))
        assert "- error: no method defined to read instruction set of data type" in str(exc_info.value)

        break

def test_to_assembly():
    word = 0x0087a783
    assert "0x0087a783 lw x15, 8(x15)" == rv32.to_assembly(rv32.decode_instruction(word))

    with pytest.raises(Exception) as exc_info:
        rv32.to_assembly(rv32.decode_instruction(word), t_base = 16)
    assert "- error: only base 10 is supported at the moment" in str(exc_info.value)

def test_instruction_to_str():
    word = 0x0087a783
    assert "0x0087a783 lw x15, 8(x15)" == rv32.instruction_to_str(rv32.decode_instruction(word))

    words_strs = {
        0x16ad5bb  : "0x016ad5bb srlw x11, x21, x22",
        0x41388eb3 : "0x41388eb3 sub x29, x17, x19",
        0x415e0b3b : "0x415e0b3b subw x22, x28, x21",
    }
    assert list(words_strs.values()) == rv32.instruction_to_str([rv32.decode_instruction(key) for key in words_strs.keys()])

    with pytest.raises(Exception) as exc_info:
        rv32.instruction_to_str(typing.cast(typing.Any, dict()))

def test_print_instruction(capsys):
    word = 0x0087a783
    di = rv32.decode_instruction(word)
    rv32.print_instruction(di)
    captured = capsys.readouterr()
    msg = "  0x0087a783 lw x15, 8(x15)\n"
    assert captured.out == msg

    pc = 0
    di.set_program_counter(pc)
    rv32.print_instruction(di)
    captured = capsys.readouterr()
    msg = "  0x00000000: 0x0087a783 lw x15, 8(x15)\n"
    assert captured.out == msg