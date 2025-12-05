#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import numpy as np
import struct

# sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__),  "..", ".." , "ttsim", "back", "tensix_neo")))
import ttsim.back.tensix_neo.isaFunctions as isaFunctions
import ttsim.back.tensix_neo.triscFunc as triscFunc
import ttsim.back.tensix_neo.tensixFunc as tensixFunc
import ttsim.back.tensix_neo.tneoSim as tneoSim
# import t3sim

import ttsim.front.llk.decoded_instruction as decoded_instruction


TEST_ARCH: str = 'ttqs'
TEST_LLK_VERSION_TAG: str = 'sep23'

def argsInit():
    args            = {}
    args['debug']   = 0
    args['arch']    = TEST_ARCH
    args['llkVersionTag'] = TEST_LLK_VERSION_TAG
    args['stack']   = {
        "0": ["0x8023FF","0x802000"],
        "1": ["0x801FFF","0x801C00"],
        "2": ["0x801BFF","0x801800"],
        "3": ["0x8017FF","0x801400"]
    }
    args['globalPointer']   = "0xffb007f0"
    if "ttqs" == args['arch']:
        args["maxNumThreadsperNeoCore"] = 4
        args.update(tneoSim.get_memory_map_from_file(os.path.join(os.path.dirname(__file__), "../../config/tensix_neo/ttqs_memory_map_mar18.json")))

    return args

def funcInit(args):
    tReg    = triscFunc.triscRegs(0, 0, args)
    ttReg   = tensixFunc.ttSplRegs(0, args)
    mData   = triscFunc.triscMemFunc(args)
    tFunc   = triscFunc.triscFunc(0, 0, mData, args, ttReg, tReg)

    return tReg, ttReg, mData, tFunc

def test_instr_2op():
    args            = argsInit()
    tReg, ttReg, mData, tFunc   = funcInit(args)

    ins = isaFunctions.instr()
    ins.setRelAddr(0)
    ins.setSrcInt([10,11])
    ins.setDstInt([8])
    ins.setImm([])
    ins.setAttr({})
    ins.setKind(decoded_instruction.instruction_kind.rv32)

    tReg.__writeReg__(ins.getSrcInt()[0], 4)
    tReg.__writeReg__(ins.getSrcInt()[1], 3)
    ins.setOp("ADD");   tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 7

    ins.setOp("SUB");   tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 1

    tReg.__writeReg__(ins.getSrcInt()[0], 3)
    tReg.__writeReg__(ins.getSrcInt()[1], 8)
    ins.setOp("SUB");   tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == -5

    tReg.__writeReg__(ins.getSrcInt()[0], 15)
    tReg.__writeReg__(ins.getSrcInt()[1], 3)
    ins.setOp("AND");   tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 3

    tReg.__writeReg__(ins.getSrcInt()[0], 12)
    tReg.__writeReg__(ins.getSrcInt()[1], 3)
    ins.setOp("OR");   tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 15

    tReg.__writeReg__(ins.getSrcInt()[0], 15)
    tReg.__writeReg__(ins.getSrcInt()[1], 3)
    ins.setOp("XOR");   tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 12

    tReg.__writeReg__(ins.getSrcInt()[0], 15)
    tReg.__writeReg__(ins.getSrcInt()[1], 2)
    ins.setOp("SLL");   tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 60

    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFFFFFFFF', 16))
    tReg.__writeReg__(ins.getSrcInt()[1], 2)
    ins.setOp("SRL");   tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0x3FFFFFFF', 16)

    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFFFFFFFF', 16))
    tReg.__writeReg__(ins.getSrcInt()[1], 2)
    ins.setOp("SRA");   tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xFFFFFFFF', 16)

    tReg.__writeReg__(ins.getSrcInt()[0], 15)
    tReg.__writeReg__(ins.getSrcInt()[1], 3)
    ins.setOp("SLT");   tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 0

    tReg.__writeReg__(ins.getSrcInt()[0], 3)
    tReg.__writeReg__(ins.getSrcInt()[1], 15)
    ins.setOp("SLTU");    tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 1

    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFFFFFFFF', 16))   # Signed: (-1)0xFFFFFFFF < 0xF
    tReg.__writeReg__(ins.getSrcInt()[1], 15)
    ins.setOp("SLT");    tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 1

    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFFFFFFFF', 16))  # Unsigned: 0xFFFFFFFF > 0xF
    tReg.__writeReg__(ins.getSrcInt()[1], 15)
    ins.setOp("SLTU");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 0

    #RV32M Multiply Extension
    tReg.__writeReg__(ins.getSrcInt()[0], 2)
    tReg.__writeReg__(ins.getSrcInt()[1], 15)
    ins.setOp("MUL");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 30

def test_instr_2opimm():
    args            = argsInit()
    tReg, ttReg, mData, tFunc   = funcInit(args)

    ins = isaFunctions.instr()
    ins.setRelAddr(0)
    ins.setSrcInt([10])
    ins.setDstInt([8])
    ins.setImm([15])
    ins.setAttr({})
    ins.setKind(decoded_instruction.instruction_kind.rv32)

    tReg.__writeReg__(ins.getSrcInt()[0], 4)
    ins.setOp("ADDI");  tFunc.__execaddi__(ins)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 19

    ins.setOp("SUBI");  tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == -11

    tReg.__writeReg__(ins.getSrcInt()[0], 15)
    ins.setOp("SUBI");  tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 0

    tReg.__writeReg__(ins.getSrcInt()[0], 3)
    ins.setOp("ANDI");  tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 3

    tReg.__writeReg__(ins.getSrcInt()[0], 12)
    ins.setOp("ORI");  tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 15

    tReg.__writeReg__(ins.getSrcInt()[0], 3)
    ins.setOp("XORI");  tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 12

    tReg.__writeReg__(ins.getSrcInt()[0], 15)
    ins.setImm([2])
    ins.setOp("SLLI");  tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 60

    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFFFFFFFF', 16))
    ins.setOp("SRLI");  tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0x3FFFFFFF', 16)

    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFFFFFFFF', 16))
    ins.setOp("SRAI");  tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xFFFFFFFF', 16)

    tReg.__writeReg__(ins.getSrcInt()[0], 15)
    ins.setImm([3])
    ins.setOp("SLTI");  tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 0

    tReg.__writeReg__(ins.getSrcInt()[0], 3)
    ins.setImm([15])
    ins.setOp("SLTIU");  tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 1

    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFFFFFFFF', 16))   # Signed: (-1)0xFFFFFFFF < 0xF
    ins.setOp("SLTI");  tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 1

    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFFFFFFFF', 16))  # Unsigned: 0xFFFFFFFF > 0xF
    ins.setOp("SLTIU");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 0

    ins = isaFunctions.instr()
    ins.setRelAddr(0)
    ins.setDstInt([8])
    ins.setImm([2])
    ins.setAttr({})
    ins.setKind(decoded_instruction.instruction_kind.rv32)

    ins.setOp("LUI");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 2 << 12

    ins.setRelAddr(1 << 2)
    ins.setOp("AUIPC");   tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == (1 << 2) + (2 << 12)

def test_instr_loadstore():
    args            = argsInit()
    tReg, ttReg, mData, tFunc   = funcInit(args)

    ins = isaFunctions.instr()
    ins.setOp("LW")
    ins.setRelAddr(0)
    ins.setSrcInt([10])
    ins.setDstInt([8])
    ins.setImm([0])
    ins.setAttr({})
    ins.setKind(decoded_instruction.instruction_kind.rv32)

    tReg.__writeReg__(ins.getSrcInt()[0], int('0x800000', 16))
    ins.setOp("LW");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == 255
    mData.__writeMem__(int('0x800000', 16), int('0x1234abcd', 16))

    ins.setOp("LW");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0x1234abcd', 16)
    ins.setOp("LH");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xffffabcd', 16)
    ins.setOp("LHU");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xabcd', 16)
    ins.setOp("LB");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xffffffcd',16)
    ins.setOp("LBU");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xcd',16)

    ins = isaFunctions.instr()
    ins.setOp("SW")
    ins.setRelAddr(0)
    ins.setSrcInt([10,12])
    ins.setImm([0])
    ins.setAttr({})
    ins.setKind(decoded_instruction.instruction_kind.rv32)

    tReg.__writeReg__(ins.getSrcInt()[0], int('0x80002F', 16))
    tReg.__writeReg__(ins.getSrcInt()[1], int('0x1234abcd', 16))
    ins.setImm([0])

    ins.setOp("SW");     tFunc.execRIns(ins, 0)
    assert mData.__readMem__(tReg.__readReg__(ins.getSrcInt()[0],'riscgpr')+0) == int('0x1234abcd', 16)
    mData.__writeMem__(int('0x80002F',16), int('0x0000abcd', 16))
    ins.setOp("SH");     tFunc.execRIns(ins, 0)
    assert mData.__readMem__(tReg.__readReg__(ins.getSrcInt()[0],'riscgpr')+0) == int('0xabcd', 16)
    mData.__writeMem__(int('0x80002F',16), int('0x0000cd', 16))
    ins.setOp("SB");     tFunc.execRIns(ins, 0)
    assert mData.__readMem__(tReg.__readReg__(ins.getSrcInt()[0],'riscgpr')+0) == int('0xcd', 16)

    ins.setOp("SW");     tFunc.execRIns(ins, 0)
    assert mData.__readMem__(tReg.__readReg__(ins.getSrcInt()[0],'riscgpr')+0) == int('0x1234abcd', 16)
    mData.__writeMem__(int('0x80002F',16), int('0x5678efcd', 16))
    ins.setOp("SH");     tFunc.execRIns(ins, 0)
    assert mData.__readMem__(tReg.__readReg__(ins.getSrcInt()[0],'riscgpr')+0) == int('0x5678abcd', 16)
    mData.__writeMem__(int('0x80002F',16), int('0x5678efcd', 16))
    ins.setOp("SB");     tFunc.execRIns(ins, 0)
    assert mData.__readMem__(tReg.__readReg__(ins.getSrcInt()[0],'riscgpr')+0) == int('0x5678efcd', 16)

def test_instr_loadstoreCfg():
    args            = argsInit()
    tReg, ttReg, mData, tFunc   = funcInit(args)
    BYTES_PER_CFG_REG = 4

    # CFG Configuration Register Read - MMR
    ins = isaFunctions.instr()
    ins.setOp("LW")
    ins.setRelAddr(0)
    ins.setSrcInt([10])
    ins.setDstInt([8])
    ins.setImm([0])
    ins.setAttr({})
    ins.setKind(decoded_instruction.instruction_kind.rv32)

    cfg_regs = args['trisc_map']['cfg_regs']
    cfg_offsets = cfg_regs['OFFSETS']

    cfg_start_addr = cfg_regs['START']
    cfg_reg_name = "DEST_TARGET_REG_CFG_MATH_SEC0_Offset"
    cfg_offset = None
    for offset, regs in cfg_offsets.items():
        if cfg_reg_name in regs.keys():
            cfg_offset_dest_cfg_math_sec0 = offset
            break

    mData.__writeMem__(cfg_start_addr+cfg_offset_dest_cfg_math_sec0*BYTES_PER_CFG_REG, int('0x1234abce', 16))
    mData.__writeMem__(cfg_start_addr+cfg_offset_dest_cfg_math_sec0*BYTES_PER_CFG_REG, int('0x1234abce', 16))
    ttReg.__writeReg__(ins.getDstInt()[0], int('0x1234abcd', 16),'cfg')
    tReg.__writeReg__(ins.getSrcInt()[0], cfg_start_addr, 'riscgpr')
    ins.setImm([cfg_offset_dest_cfg_math_sec0*BYTES_PER_CFG_REG])

    regType,regIndex = ttReg.__ismmr__(tReg.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
    if(tReg.__readReg__(ins.getSrcInt()[0],'riscgpr')  + ins.getImm()[0] != cfg_start_addr +cfg_offset_dest_cfg_math_sec0*BYTES_PER_CFG_REG):
        assert regType == '';   assert regIndex == -1
    else:
        assert regType == 'cfg'; assert regIndex == cfg_offset_dest_cfg_math_sec0
    ins.setOp("LW");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0x1234abcd', 16)
    ins.setOp("LH");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xffffabcd', 16)
    ins.setOp("LHU");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xabcd', 16)
    ins.setOp("LB");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xffffffcd',16)
    ins.setOp("LBU");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xcd',16)

    # CFG Configuration Register Write- MMR
    ins = isaFunctions.instr()
    ins.setOp("SW")
    ins.setRelAddr(0)
    ins.setSrcInt([10,12])
    ins.setImm([0])
    ins.setAttr({})
    ins.setKind(decoded_instruction.instruction_kind.rv32)

    tReg.__writeReg__(ins.getSrcInt()[0], cfg_start_addr ,'riscgpr')
    tReg.__writeReg__(ins.getSrcInt()[1], int('0x1234abcd', 16)     ,'riscgpr')
    ins.setImm([cfg_offset_dest_cfg_math_sec0*BYTES_PER_CFG_REG])

    ins.setOp("SW");     tFunc.execRIns(ins, 0)
    assert ttReg.__readReg__(cfg_offset_dest_cfg_math_sec0, 'cfg') == int('0x1234abcd', 16)
    ins.setOp("SH");     tFunc.execRIns(ins, 0)
    assert ttReg.__readReg__(cfg_offset_dest_cfg_math_sec0, 'cfg') == int('0xabcd', 16)
    ins.setOp("SB");     tFunc.execRIns(ins, 0)
    assert ttReg.__readReg__(cfg_offset_dest_cfg_math_sec0, 'cfg') == int('0xcd', 16)

    # MOP Configuration Register Read - MMR
    ins = isaFunctions.instr()
    ins.setOp("LW")
    ins.setRelAddr(0)
    ins.setSrcInt([10])
    ins.setDstInt([8])
    ins.setImm([0])
    ins.setAttr({})
    ins.setKind(decoded_instruction.instruction_kind.rv32)

    mop_start_addr = args['trisc_map']['mop_cfg']['START']

    mData.__writeMem__(mop_start_addr, int('0x1234abce', 16))
    tReg.__writeReg__(ins.getSrcInt()[0], mop_start_addr, 'riscgpr')
    ins.setImm(0)

    regType,regIndex = ttReg.__ismmr__(tReg.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
    if(tReg.__readReg__(ins.getSrcInt()[0],'riscgpr')  + ins.getImm()[0] != mop_start_addr):
        assert regType == '';   assert regIndex == -1
    else:
        assert regType == 'mop'; assert regIndex == 0
        ttReg.__writeReg__(regIndex, int('0x1234abcd', 16), regType)

    ins.setOp("LW");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0x1234abcd', 16)
    ins.setOp("LH");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xffffabcd', 16)
    ins.setOp("LHU");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xabcd', 16)
    ins.setOp("LB");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xffffffcd',16)
    ins.setOp("LBU");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xcd',16)

    ## MOP Configuration Register Write - MMR
    ins = isaFunctions.instr()
    ins.setOp("SW")
    ins.setRelAddr(0)
    ins.setSrcInt([10,12])
    ins.setImm([0])
    ins.setAttr({})
    ins.setKind(decoded_instruction.instruction_kind.rv32)

    tReg.__writeReg__(ins.getSrcInt()[0], mop_start_addr ,'riscgpr')
    tReg.__writeReg__(ins.getSrcInt()[1], int('0x1234abcd', 16)     ,'riscgpr')
    ins.setImm(0)

    ins.setOp("SW");     tFunc.execRIns(ins, 0)
    assert ttReg.__readReg__(0,'mop') == int('0x1234abcd', 16)
    ins.setOp("SH");     tFunc.execRIns(ins, 0)
    assert ttReg.__readReg__(0,'mop') == int('0xabcd', 16)
    ins.setOp("SB");     tFunc.execRIns(ins, 0)
    assert ttReg.__readReg__(0,'mop') == int('0xcd', 16)

    ## Instruction Buffer Read - MMR
    ins = isaFunctions.instr()
    ins.setRelAddr(0)
    ins.setSrcInt([10])
    ins.setDstInt([8])
    ins.setImm([0])
    ins.setAttr({})
    ins.setKind(decoded_instruction.instruction_kind.rv32)

    instr_buffer_start_addr = args['trisc_map']['ibuffer']['START']

    mData.__writeMem__(instr_buffer_start_addr, int('0x1234abce', 16))
    tReg.__writeReg__(ins.getSrcInt()[0], instr_buffer_start_addr, 'riscgpr')
    ins.setImm(0)

    regType,regIndex = ttReg.__ismmr__(tReg.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
    if(tReg.__readReg__(ins.getSrcInt()[0],'riscgpr')  + ins.getImm()[0] != instr_buffer_start_addr):
        assert regType == '';   assert regIndex == -1
    else:
        assert regType == 'instrBuffer'; assert regIndex == 0
        ttReg.__writeReg__(regIndex, int('0x1234abcd', 16), regType)

    ins.setOp("LW");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0x1234abcd', 16)
    ins.setOp("LH");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xffffabcd', 16)
    ins.setOp("LHU");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xabcd', 16)
    ins.setOp("LB");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xffffffcd',16)
    ins.setOp("LBU");     tFunc.execRIns(ins, 0)
    assert tReg.__readReg__(ins.getDstInt()[0], 'riscgpr') == int('0xcd',16)

    ## Instruction Buffer Write - MMR
    ins = isaFunctions.instr()
    ins.setRelAddr(0)
    ins.setSrcInt([10,12])
    ins.setImm([0])
    ins.setAttr({})
    ins.setKind(decoded_instruction.instruction_kind.rv32)

    tReg.__writeReg__(ins.getSrcInt()[0], instr_buffer_start_addr,'riscgpr')
    tReg.__writeReg__(ins.getSrcInt()[1], int('0x1234abcd', 16),'riscgpr')
    ins.setImm(0)

    ins.setOp("SW");     tFunc.execRIns(ins, 0)
    assert ttReg.__readReg__(0,'instrBuffer') == int('0x1234abcd', 16)  # Read does not destroy
    ins.setOp("SH");     tFunc.execRIns(ins, 0)
    assert ttReg.__readReg__(0,'instrBuffer') == int('0xabcd', 16)      # Read does not destroy
    ins.setOp("SB");     tFunc.execRIns(ins, 0)
    assert ttReg.__readReg__(0,'instrBuffer') == int('0xcd', 16)        # Read does not destroy

    tReg.__writeReg__(ins.getSrcInt()[1], int('0x604000E', 16)     ,'riscgpr')
    ins.setOp("SW");     tFunc.execRIns(ins, 0)
    assert tFunc.readInstructionBufMem() == int('0x604000E', 16)        # Read and Destroy
    assert tFunc.readInstructionBufMem() == None                        # Confirm destruction

def test_instr_branch():
    args            = argsInit()
    tReg, ttReg, mData, tFunc   = funcInit(args)

    ins = isaFunctions.instr()
    ins.setRelAddr(0)
    ins.setSrcInt([10, 11])
    ins.setImm([8])
    ins.setAttr({})
    ins.setKind(decoded_instruction.instruction_kind.rv32)

    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFF', 16))
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xFF', 16))
    ins.setOp("BEQ");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 8
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xF', 16))
    ins.setOp("BEQ");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 4

    ins.setRelAddr(0)
    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFF', 16))
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xF', 16))
    ins.setOp("BNE");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 8
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xFF', 16))
    ins.setOp("BNE");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 4

    ins.setRelAddr(0)
    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFF', 16))
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xFF', 16))
    ins.setOp("BLT");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 4
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xFFF', 16))
    ins.setOp("BLT");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 8
    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFFFFFFFF', 16))
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xFFF', 16))
    ins.setOp("BLT");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 8

    ins.setRelAddr(0)
    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFF', 16))
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xFFF', 16))
    ins.setOp("BGE");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 4
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xFF', 16))
    ins.setOp("BGE");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 8
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xF', 16))
    ins.setOp("BGE");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 8
    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFFFFFFFF', 16))
    ins.setOp("BGE");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 4

    ins.setRelAddr(0)
    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFF', 16))
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xF', 16))
    ins.setOp("BLTU");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 4
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xFFF', 16))
    ins.setOp("BLTU");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 8
    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFFFFFFFF', 16))
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xFFF', 16))
    ins.setOp("BLTU");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 4

    ins.setRelAddr(0)
    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFF', 16))
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xFFF', 16))
    ins.setOp("BGEU");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 4
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xFF', 16))
    ins.setOp("BGEU");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 8
    tReg.__writeReg__(ins.getSrcInt()[1], int('0xF', 16))
    ins.setOp("BGEU");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 8
    tReg.__writeReg__(ins.getSrcInt()[0], int('0xFFFFFFFF', 16))
    ins.setOp("BGEU");       nAddr = tFunc.execRIns(ins, 0)
    assert nAddr == 8

    ins = isaFunctions.instr()
    ins.setRelAddr(0)
    ins.setDstInt([8])
    ins.setImm([256])
    ins.setAttr({})
    ins.setKind(decoded_instruction.instruction_kind.rv32)

    ins.setOp("JAL");       nAddr = tFunc.execRIns(ins,0)
    assert nAddr == 256

    ins.setSrcInt([10])
    tReg.__writeReg__(ins.getSrcInt()[0], int('0x800000', 16))
    ins.setOp("JALR");      nAddr = tFunc.execRIns(ins,0)
    assert nAddr == int('0x800100', 16)

def test_misc():
    args            = argsInit()
    tReg, ttReg, mData, tFunc   = funcInit(args)

    result  = triscFunc.convert_to_format(int('0xFFFFFFFF',16),'i', 8)
    assert result == -1
    result  = triscFunc.convert_to_format(int('0xFFFFFFFF',16),'i', 16)
    assert result == -1
    result  = triscFunc.convert_to_format(int('0xFFFFFFFF',16),'i', 32)
    assert result == -1
    # result  = triscFunc.convert_to_format(int('0xFFFFFFFF',16),'i', 64)   #### TODO - 32 to 64 bit conversion #####
    # assert result == int('0xFFFFFFFF',16)

    result  = triscFunc.convert_to_format(int('0xFF',16),'I', 32)
    assert result == 255
    result  = triscFunc.convert_to_format(int('0xFFFFFFFF',16),'I', 32)
    assert result == int('0xFFFFFFFF',16)

    result = triscFunc.toggle(1,1)                              #### TODO toggle testing ####

    ins = isaFunctions.instr()
    ins.setRelAddr(0)
    ins.setDstInt([8])
    ins.setSrcInt([10])
    ins.setAttr({'csr': 0})
    ins.setKind(decoded_instruction.instruction_kind.rv32)

    ins.setOp("CSRRW");      nAddr = tFunc.execRIns(ins,0)      #### TODO - CSRRW testing ####
    ins.setOp("CSRRC");      nAddr = tFunc.execRIns(ins,0)      #### TODO - CSRRC testing ####
    ins.setOp("CSRRS");      nAddr = tFunc.execRIns(ins,0)      #### TODO - CSRRC testing ####

    ins.setAttr({'x': 0, 'y': 0})
    ins.setOp("FENCE");      nAddr = tFunc.execRIns(ins,0)      #### TODO - FENCE testing ####
