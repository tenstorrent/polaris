#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Functional Model
import sys
from typing import List

import copy
import enum
import struct

import ttsim.back.tensix_neo.isaFunctions as isaFunctions
import ttsim.front.llk.decoded_instruction as decoded_instruction

MAX_THREADS                   = 4 # 3
NUM_RISCGPR_REGISTERS         = 64
NUM_TTGPR_REGISTERS           = 128 #65536
NUM_MOPCFG_REGISTERS          = 64
NUM_CFG_REGISTERS             = 64

BANK_UPDATE_THRESHOLD = 512
DEST_REG_INDEX = 3

forceSkipJump                 = []
# forceSkipJump                 = ["0x4eac", "0x4f80", "0x5d80", "0x55c0", "0x8320"]

def convert_to_format(num, format, nbits):
    num_bits_in_byte = 8
    assert nbits%num_bits_in_byte == 0 , "Not byte aligned"
    match format:
        case 'i'| 'I':
            assert struct.calcsize(format) >= int(nbits/num_bits_in_byte) , f"Not the intended format {format} size Expected not greater than{struct.calcsize(format)} bytes Actual:{int(nbits/num_bits_in_byte)}"

            # Signbit extension
            signBit = (num & (1 << (nbits-1)) == (1 << (nbits-1)))
            if signBit:     hex_str = hex((num + (1 << nbits)) % (1 << nbits))
            else:           hex_str = hex(num)
            hex_str = hex_str.replace('0x','')

            # struct.unpack needs at least 32 bits
            uBufferLen = (int(nbits/32) + 1)*32 if nbits%32 != 0 else (int(nbits/32))*32
            remLen32 = int((uBufferLen - (len(hex_str) * 4))/4)
            if signBit:     hex_str  = 'f' * remLen32 + hex_str
            else:           hex_str  = '0' * remLen32 + hex_str
            byte_array = bytes.fromhex(hex_str)
            si  = struct.unpack('>'+format, byte_array)[0] # Native Endianness, Convert to 4-byte integer

            return si
        case _:     assert False, f"Unsupported format {format}"


def toggle(val, mask):
    # Handle 4-bits toggle of val based on (toggle) mask
    assert val <= 15 , "More than 4-bits not supported"
    maskBits = [];  valBits = []
    result = 0

    # Threads = 4. TODO: Need to convert to an enum
    for i in range(4):
        maskBits.append((mask & (0x1 << (i))) >> i)
        valBits.append((val & (0x1 << (i))) >> i)

    #TODO: Is there a better way than int(not(bit)) to achieve toggle bit)
    for i in range(len(maskBits)):
        if(maskBits[i]): valBits[i] = int(not(valBits[i]))
        result = result + (valBits[i] << (i))

    # print(f"MaskBits={maskBits}, ValBits={valBits}")
    return result

class triscMemFunc:

    def __init__(self, args):
        self.memList = {}
        # TODO: PC Buffer Register Mapping
        self.memList[int(0xFFE80034)]   = 0
        # self.memList[int(0x80F080)]     = int(0xFF)
        # double indirection
        addr2 = 111
        self.memList[int(0xFFB00010)]  = 111
        # self.memList[self.memList(int(0xFFB00010))] = 1
        self.memList[111]   = 1

        self.memList[int(0xFFB006CC)] = 100

        ## exp
        self.memList[int(0x80F084)]  = 0

        ## WH eltwise binary
        self.memList[int(0xFFB48010)] = 0xFFFFFFFF
        self.memList[int(0xFFB49010)] = 0xFFFFFFFF
        self.memList[int(0xFFB48028)] = 0xFFFFFFFF
        self.memList[int(0xFFB49028)] = 0xFFFFFFFF
        self.memList[int(0xFFB4A028)] = 0xFFFFFFFF


        self.memList[int(0x2A3)]      = 0x80
        self.memList[int(0xFFE80024)] = 0x0

        match args['arch']:
            case 'ttqs':
                # TODO: Map to TILE_COUNTERS from Address Map
                # TODO: Find faster method than loop.
                a = int(0x0080b000)             # TILE_COUNTERS_START_ADDR
                while a < int(0x0080c000):      # TILE_COUNTERS_END_ADDR
                    self.memList[a]     = 0     # Set TILE_COUNTERS Memory Space to zero to exit the loop condition below
                                                # while(tile_counters[BUF_DESC].f.acked != 0);
                    a += 4                      # 32-bit type. TODO: Force type explicitly for 32-bit architecture
            case _:
                assert False, "Unhandled scenario"
    # TODO: Support non 32-bit writes
    def __writeMem__(self, addr, val):
        self.memList[addr] = val
        # print(f"WRITEMEM: MEM[{hex(addr)}]={hex(val)}")

    # TODO: Support non 32-bit reads
    def __readMem__(self, hexAddr,):
        assert self.memList.get(int(hexAddr))  != None, "Memory not initialized"+ str(hexAddr)
        # print(f"READMEM: MEM[{hex(hexAddr)}]={hex(self.memList.get(int(hexAddr)))}")
        return(self.memList.get(int(hexAddr)))

    def isInitialized(self, addr):
        return addr in self.memList

    def __printMem__(self):
        for k,v in self.memList.items():
            if (self.memList[k] != 0):
                print(hex(k), hex(self.memList[k]), end=",")
        print()

class triscRegs:
    def __init__(self, coreId, threadId, args):
        self.coreId     = coreId
        self.threadId   = threadId
        self.debug      = args['debug']
        self.args       = args
        ### CONFIGURATION REGISTERS
        ### REGISTERS
        self.regTypeDict = {
            'riscgpr' : [
                64,                 #NUM_REGISTERS,
                int(0xee000000),    #MMR_START #TODO: To verify. Currently random
                64,                 #MMR_SIZE
                1,                  #NUM_BANKS
            ],
            'csr' : [
                4096,              #NUM_REGISTERS,
                int(0xef000000),    #MMR_START #TODO: To verify. Currently random
                4096,               #MMR_SIZE
                1,                  #NUM_BANKS
            ],
            'triscId' : [
                1,                  #NUM_REGISTERS,
                int(0xa71c),        #MMR_START #TODO: To verify. Currently random
                1,                  #MMR_SIZE
                1,                  #NUM_BANKS
            ],

        }
        self.regTypes = [ key for key, val in self.regTypeDict.items() ]
        self.regSizes = [ self.regTypeDict[key][0]*self.regTypeDict[key][3] for key,val in self.regTypeDict.items() ]
        self.regTempsriscgpr = [0, 2, 5, 6, 7, 28, 29, 30, 31]

        self.reg : dict[str,List] = {}

        for i in range(len(self.regTypes)):
            regList : List[int] = []
            for j in range(self.regSizes[i]):
                regList.append(-1)
            self.reg[self.regTypes[i]] = copy.deepcopy(regList)

        self.reg['riscgpr'][0] = 0                          #Zero Register - RISCGPR
        self.reg['riscgpr'][2] = int(self.args['stack'][str(self.threadId)][0],16)   #Stack Pointer - RISCGPR
        self.reg['riscgpr'][3] = int(self.args['globalPointer'],16)   #Global Pointer - RISCGPR

        # TRISC_ID
        self.reg['triscId'][0] = self.threadId


    # Tensix Pipes Register File
    def __writeReg__(self,r, val, t ='riscgpr'):
        assert t in self.regTypes, f"RegType:{t} not supported:"
        assert val != None , "Only legal values supported" + str(val)
        # print(f"\t{self.threadId}: {t}[{r}] = {hex(val)}")
        self.reg[t][r] = val

    def __readReg__(self, r, t ='riscgpr'):
        assert t in self.regTypes, f"RegType:{t} not supported:"
        if (t == 'riscgpr' and (r in self.regTempsriscgpr)): # Temporary Registers - RISCV
            assert self.reg[t][r] != -1, "Illegal Read " + t + "[" + str(r) + "]"
            assert self.reg[t][r] != None, "Illegal Read " + t + "[" + str(r) + "]"
        elif(t == 'riscgpr' and self.reg[t][r] == -1):     # For non temporary variables, reset initial value to zero
            self.reg[t][r] = 0

        return self.reg[t][r]

    def __ismmr__(self,addr ):
        regTypeSel = ''
        offset = -1
        # 1. Check MMR or not
        for key, val in self.regTypeDict.items():
            if (addr in range(self.regTypeDict[key][1], (self.regTypeDict[key][1] + self.regTypeDict[key][2]))):
                if(self.debug & 0x10):       print(f"Is MMR, addr={hex(addr)},type={key},AddrStart={self.regTypeDict[key][1]},AddrEnd={self.regTypeDict[key][1] + self.regTypeDict[key][2]}")
                regTypeSel = key
                break
            # if(self.debug & 0x10):       print("Address:", hex(addr), "not in", key, "Address Range:", hex(self.regTypeDict[key][1]), hex(self.regTypeDict[key][1] + self.regTypeDict[key][2]))
        if regTypeSel != '':
            # 2. Find reg mapping
            # 2a. Test range
            match regTypeSel:
                case 'mop':
                    offset = int((addr - self.regTypeDict[regTypeSel][1])/4)
                case _:
                    offset = int((addr - self.regTypeDict[regTypeSel][1])/4)
                    # print(f"addr={hex(addr)}, base={hex(self.regTypeDict[regTypeSel][1])}, offsetinbytes={hex(addr - self.regTypeDict[regTypeSel][1])}, offset={offset} ")

        return regTypeSel, offset

    def __printReg__(self, type='riscgpr'):
        assert type in self.regTypes, "RegType not supported:" + type + "Supported RegTypes" +  self.regTypes
        regCnt = 0
        print("Reg. Thread[" , self.threadId, "]:", end='', sep='')
        while (regCnt < len(self.reg[type])):
            if(self.reg[type][regCnt] != -1):
                print (type, "[", regCnt ,"]=", hex(self.reg[type][regCnt]),",", sep='', end='' )
            regCnt += 1
        print()

class triscFunc:
    def __init__(self, coreId, threadId, mem, args,ttSplRegs, triscRegs):
        self.coreId     = coreId
        self.threadId   = threadId
        self.debug      = args['debug']
        self.args       = args
        self.ttSplRegs  = ttSplRegs
        self.triscRegs  = triscRegs

        #Memory Space.
        self.memData    = mem

    #TODO: Move outside triscFunc
    def _is_dest_target_reg_range(self, reg_index, min, max ):
        # Check if register is in address map range.
        return (reg_index >= min and reg_index <= max)

    #S-Type Store
    def __execsw__(self, ins):
        assert ins.getOp() == "SW", "Expected opcode SW Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two  Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        regType,regIndex = self.triscRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        cfgRegType,cfgRegIndex = self.ttSplRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        src = [] ; dst = [] ; imm = [] ; vldUpd = {} ; bankUpd = {}
        condChkVld = {}; condWriVld = {}

        #4 registers / 4 threads
        condChkVld[0] = [-2, -2, -2, -2] #-2 implies don't care
        condChkVld[1] = [-2, -2, -2, -2] #-2 implies don't care
        condChkVld[2] = [-2, -2, -2, -2] #-2 implies don't care
        condChkVld[3] = [-2, -2, -2, -2] #-2 implies don't care

        condWriVld[0] = [-2, -2, -2, -2] #-2 implies don't care
        condWriVld[1] = [-2, -2, -2, -2] #-2 implies don't care
        condWriVld[2] = [-2, -2, -2, -2] #-2 implies don't care
        condWriVld[3] = [-2, -2, -2, -2] #-2 implies don't care

        if(regType == '') and (cfgRegType == ''):  # Not MMR
            assert regIndex == -1 and cfgRegIndex == -1, "Unrecognized combination"
            if self.debug & 0x8: print(f"__execsw:MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")
            self.memData.__writeMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0], self.triscRegs.__readReg__(ins.getSrcInt()[1]))
        elif(cfgRegType == 'cfg'):              # Write to MMR (Tensix)
            assert cfgRegIndex != -1, "Config Reg Index cannot be uninitialized"
            if self.debug & 0x8:
                print(f"__execsw:{cfgRegType}[{hex(cfgRegIndex)}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")

            status = self.ttSplRegs.writeCfgReg(cfgRegIndex, self.triscRegs.__readReg__(ins.getSrcInt()[1]), ins)

            src = status.getSrcInt()
            dst = status.getDstInt()
            imm = status.getImm()
            vldUpd = status.vldUpdMask
            bankUpd = status.bankUpdMask
            condChkVld = status.condChkVldUpdVal
            condWriVld = status.condWriVldUpdVal
        elif(cfgRegType in ['mop', 'mopSync','idleSync']):              # Write to MMR (TRISC)
            if self.debug & 0x8: print(f"__execsw:{cfgRegType}[{hex(cfgRegIndex)}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            match cfgRegType:
                case 'mop': 
                    assert cfgRegIndex >=0 and cfgRegIndex <= 8, f"MOP Reg Index out of range. cfgRegIndex={cfgRegIndex}"
                    if self.debug & 0x8:     print(f"Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} Writing to (TENSIX) {hex(cfgRegIndex)} mop[{hex(cfgRegIndex)}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")
                    self.ttSplRegs.__writeReg__(self.threadId*64 + cfgRegIndex, self.triscRegs.__readReg__(ins.getSrcInt()[1]), cfgRegType)
                case 'mopSync': 
                    assert cfgRegIndex == 0, "Only one mopSync register supported. Unexpected Index=" + str(cfgRegIndex)
                    if self.debug & 0x8:     print(f"Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} Writing to (TENSIX) {hex(cfgRegIndex)} mopSync[{hex(cfgRegIndex)}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")
                    #TODO: Evaluate Disable Write to PCBuffer[2] for mopSync
                    self.ttSplRegs.__writeReg__(self.threadId + cfgRegIndex, self.triscRegs.__readReg__(ins.getSrcInt()[1]), cfgRegType)
                case 'idleSync': 
                    assert cfgRegIndex == 0, "Only one idleSync register supported. Unexpected Index=" + str(cfgRegIndex)
                    if self.debug & 0x8:     print(f"Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} Writing to (TENSIX) {hex(cfgRegIndex)} idleSync[{hex(cfgRegIndex)}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")
                    self.ttSplRegs.__writeReg__(self.threadId + cfgRegIndex, self.triscRegs.__readReg__(ins.getSrcInt()[1]), cfgRegType)
                case _:     
                    assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
        elif(cfgRegType in ['instrBuffer']):              # Write to MMR (TRISC)
            if self.debug & 0x8: print(f"__execsw:{cfgRegType}[{hex(cfgRegIndex)}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            match cfgRegType:
                case 'instrBuffer': assert cfgRegIndex == 0, "InstrBuffer Reg Index out of range"
                case _:             assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
            if self.debug & 0x8:     print(f"Writing to (TENSIX) Thread[{self.threadId}]:{hex(cfgRegIndex)} {cfgRegType}[{hex(cfgRegIndex)}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")
            self.ttSplRegs.__writeReg__(cfgRegIndex, self.triscRegs.__readReg__(ins.getSrcInt()[1]), cfgRegType)
        else:                                   # Write to MMR (TRISC)
            assert regIndex != -1, "Reg Index cannot be uninitialized"
            if self.debug & 0x8: print(f"__execsw:{regType}[{hex(regIndex)}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            print(f"WARNING: Unhandled special register(TRISC) {regType}[{hex(regIndex)}], Memlocation={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}" )
            assert not self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) , "MMR initialized in memory [" + str(ins.getSrcInt()[0] + ins.getImm()[0]) +"]"

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd) #Setting Bank Value instead of bank Update
        ins.setCondChkVldUpd(condChkVld)
        ins.setCondWriVldUpd(condWriVld)

        if(self.debug & 0x10):
            print(f"{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsh__(self, ins):
        assert ins.getOp() == "SH", "Expected opcode SH Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two  Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        regType,regIndex = self.triscRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        cfgRegType,cfgRegIndex = self.ttSplRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        if(regType == '') and (cfgRegType == ''):  # Not MMR
            assert regIndex == -1, "Unrecognized combination"
            assert cfgRegIndex == -1, "Unrecognized combination"
            if(not self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])):
                print("TODO: Initializing memory which should have been available. Addr=",hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]), " Value = 0xFF")
                self.memData.__writeMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0], 0xFF)
            self.memData.__writeMem__(
                self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0],
                (self.memData.__readMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) & 0xFFFF0000) + (self.triscRegs.__readReg__(ins.getSrcInt()[1]) & 0xFFFF)
                )
        elif(cfgRegType == 'cfg'):              # Write to MMR (Tensix)
            assert cfgRegIndex != -1, "Config Reg Index cannot be uninitialized"
            if self.debug & 0x8:
                print(f"__execsw:{cfgRegType} , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")

            status = self.ttSplRegs.writeCfgReg(cfgRegIndex, self.triscRegs.__readReg__(ins.getSrcInt()[1]) & 0xFFFF, ins)

            # if(cfgRegIndex >= self.args['trisc_map']['cfg_regs']['OFFSETS']['DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32'] and cfgRegIndex <= self.args['trisc_map']['cfg_regs']['OFFSETS']['DEST_TARGET_REG_CFG_MATH_SEC3_Offset_ADDR32']):  # DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32
            #     print(f"Writing to (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} DEST_TARGET_REG_CFG cfg[{hex(cfgRegIndex)}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")
            #     self.ttSplRegs.__writeReg__(cfgRegIndex,self.triscRegs.__readReg__(ins.getSrcInt()[1]) & 0xFFFF,cfgRegType)
            #     # if self.triscRegs.__readReg__(ins.getSrcInt()[1]) >=512:  bankUpd[3] = 1
            # else:
            #     print(f"WARNING: Unhandled special register (TENSIX) {cfgRegType}[{cfgRegIndex}], Memlocation={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}" )
        elif(cfgRegType == 'mop'):              # Write to MMR (TRISC)
            if self.debug & 0x8: print(f"__execsw:{cfgRegType} , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            if(cfgRegIndex >=0 and cfgRegIndex <= 8): # MOP
                if self.debug & 0x8:     print("Writing to (TENSIX) Thread[{0}]:{2} mop[{1}]={3}".format(self.threadId, hex(cfgRegIndex), hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]), hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))))
                self.ttSplRegs.__writeReg__(self.threadId*64 + cfgRegIndex, self.triscRegs.__readReg__(ins.getSrcInt()[1]) & 0xFFFF, cfgRegType)
            else:
                assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
        elif(cfgRegType == 'instrBuffer'):      # Write to MMR (TENSIX)
            if(cfgRegIndex ==0):  #InstrBuffer
                if self.debug & 0x8:     print("Writing to (TENSIX) Thread[{0}]:{2} instrbuf[{1}]={3}".format(self.threadId, hex(cfgRegIndex), hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]), hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))))
                self.ttSplRegs.__writeReg__(cfgRegIndex, self.triscRegs.__readReg__(ins.getSrcInt()[1]) & 0xFFFF, cfgRegType)
            else:
                assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
        else:               # Write to MMR
            self.triscRegs.__writeReg__(regIndex, self.triscRegs.__readReg__(ins.getSrcInt()[1]) & 0xFFFF, regType)
            assert not self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) , "MMR initialized in memory [" + str(ins.getSrcInt()[0] + ins.getImm()[0]) +"]"

        if(self.debug & 0x10):
            print(f"{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]) & 0xFFFF)}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsb__(self, ins):
        assert ins.getOp() == "SB", "Expected opcode SB Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two  Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        regType,regIndex = self.triscRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        cfgRegType,cfgRegIndex = self.ttSplRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        if(regType == '') and (cfgRegType == ''):  # Not MMR
            assert regIndex == -1, "Unrecognized combination"
            if(not self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])):
                print("TODO: Initializing memory which should have been available. Addr=",hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]), " Value = 0xFF")
                self.memData.__writeMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0], 0xFF)
            self.memData.__writeMem__(
                self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0],
                (self.memData.__readMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) & 0xFFFFFF00) + (self.triscRegs.__readReg__(ins.getSrcInt()[1]) & 0xFF)
                )
        elif(cfgRegType == 'cfg'):              # Write to MMR (Tensix)
            assert cfgRegIndex != -1, "Config Reg Index cannot be uninitialized"
            if self.debug & 0x8:
                print(f"__execsw:{cfgRegType} , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")

            status = self.ttSplRegs.writeCfgReg(cfgRegIndex, self.triscRegs.__readReg__(ins.getSrcInt()[1]) & 0xFF, ins)

            # if(cfgRegIndex >= self.args['trisc_map']['cfg_regs']['OFFSETS']['DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32'] and cfgRegIndex <= self.args['trisc_map']['cfg_regs']['OFFSETS']['DEST_TARGET_REG_CFG_MATH_SEC3_Offset_ADDR32']):  # DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32
            #     print(f"Writing to (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} DEST_TARGET_REG_CFG cfg[{hex(cfgRegIndex)}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")
            #     self.ttSplRegs.__writeReg__(cfgRegIndex,self.triscRegs.__readReg__(ins.getSrcInt()[1]) & 0xFF,cfgRegType)
            #     # if self.triscRegs.__readReg__(ins.getSrcInt()[1]) >=512:  bankUpd[3] = 1
            # else:
            #     print(f"WARNING: Unhandled special register (TENSIX) {cfgRegType}[{cfgRegIndex}], Memlocation={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}" )
        elif(cfgRegType == 'mop'):              # Write to MMR (TRISC)
            if self.debug & 0x8: print(f"__execsw:{cfgRegType} , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            if(cfgRegIndex >=0 and cfgRegIndex <= 8): # MOP
                if self.debug & 0x8:     print("Writing to (TENSIX) Thread[{0}]:{2} mop[{1}]={3}".format(self.threadId, hex(cfgRegIndex), hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]), hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))))
                self.ttSplRegs.__writeReg__(self.threadId*64 + cfgRegIndex, self.triscRegs.__readReg__(ins.getSrcInt()[1]) & 0xFF, cfgRegType)
            else:
                assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
        elif(cfgRegType == 'instrBuffer'):      # Write to MMR (TENSIX)
            if(cfgRegIndex ==0):  #InstrBuffer
                if self.debug & 0x8:     print("Writing to (TENSIX) Thread[{0}]:{2} instrbuf[{1}]={3}".format(self.threadId, hex(cfgRegIndex), hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]), hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))))
                self.ttSplRegs.__writeReg__(cfgRegIndex, self.triscRegs.__readReg__(ins.getSrcInt()[1]) & 0xFF, cfgRegType)
            else:
                assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
        else:               # Write to MMR
            self.triscRegs.__writeReg__(regIndex, self.triscRegs.__readReg__(ins.getSrcInt()[1]) & 0xFF, regType)
            assert not self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) , "MMR initialized in memory [" + str(ins.getSrcInt()[0] + ins.getImm()[0]) +"]"

        if(self.debug & 0x10):
            print(f"{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]) & 0xFF)}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr


    #R-type
    def __execadd__(self, ins):
        assert ins.getOp() == "ADD", "Expected opcode ADD Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) + self.triscRegs.__readReg__(ins.getSrcInt()[1]))

        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} + {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsub__(self, ins):
        assert ins.getOp() == "SUB", "Expected opcode SUB Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} - {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) - self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")
        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) - self.triscRegs.__readReg__(ins.getSrcInt()[1]))

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execand__(self,ins):
        assert ins.getOp() == "AND", "Expected opcode AND Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} and {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) & self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")

        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) & self.triscRegs.__readReg__(ins.getSrcInt()[1]))
        nextRelAddr = ins.getRelAddr() + 4

        return nextRelAddr

    def __execor__(self,ins):
        assert ins.getOp() == "OR", "Expected opcode OR Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} or {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) | self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")

        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) | self.triscRegs.__readReg__(ins.getSrcInt()[1]))
        nextRelAddr = ins.getRelAddr() + 4

        return nextRelAddr

    def __execxor__(self,ins):
        assert ins.getOp() == "XOR", "Expected opcode XOR Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} xor {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) ^ self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")

        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) ^ self.triscRegs.__readReg__(ins.getSrcInt()[1]))
        nextRelAddr = ins.getRelAddr() + 4

        return nextRelAddr

    def __execsll__(self,ins):
        assert ins.getOp() == "SLL", "Expected opcode SLL Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) <<  self.triscRegs.__readReg__(ins.getSrcInt()[1]))
        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} << {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) << self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsrl__(self,ins):
        assert ins.getOp() == "SRL", "Expected opcode SRL Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) >> self.triscRegs.__readReg__(ins.getSrcInt()[1]))
        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} >> {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) >> self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsra__(self,ins):
        assert ins.getOp() == "SRA", "Expected opcode SRA Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        signBit = self.triscRegs.__readReg__(ins.getSrcInt()[0]) & 0x80000000 == 0x80000000
        if(signBit):        self.triscRegs.__writeReg__(ins.getDstInt()[0], ((0xFFFFFFFF >> (32-ins.getSrcInt()[1])) << (32-ins.getSrcInt()[1])) + (self.triscRegs.__readReg__(ins.getSrcInt()[0]) >> ins.getSrcInt()[1]))
        else:               self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) >> self.triscRegs.__readReg__(ins.getSrcInt()[1]))

        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} >> {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) >> self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execslt__(self,ins):
        assert ins.getOp() == "SLT", "Expected opcode SLT Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        self.triscRegs.__writeReg__(ins.getDstInt()[0], 0)
        # 32-bit signed comparison
        if(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[0]),'i',32) < convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[1]),'i',32)):
            self.triscRegs.__writeReg__(ins.getDstInt()[0], 1)
        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} < {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))} = {hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsltu__(self,ins):
        assert ins.getOp() == "SLTU", "Expected opcode SLTU Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        self.triscRegs.__writeReg__(ins.getDstInt()[0], 0)
        if(self.triscRegs.__readReg__(ins.getSrcInt()[0]) < self.triscRegs.__readReg__(ins.getSrcInt()[1])):
            self.triscRegs.__writeReg__(ins.getDstInt()[0], 1)
        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} < {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))} = {hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    #I-type
    def __execaddi__(self, ins):
        assert ins.getOp() == "ADDI", "Expected opcode ADDI Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One Imm expected"

        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} + {hex(ins.getImm()[0])} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0],'riscgpr') + ins.getImm()[0])

        nextRelAddr = ins.getRelAddr() + 4

        return nextRelAddr

    def __execsubi__(self, ins):
        assert ins.getOp() == "SUBI", "Expected opcode SUBI Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One Imm expected"

        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) - ins.getImm()[0])

        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} - {hex(ins.getImm()[0])} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) - ins.getImm()[0])}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execandi__(self,ins):
        assert ins.getOp() == "ANDI", "Expected opcode ANDI Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One Imm expected"

        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} and {hex(ins.getImm()[0])} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) & ins.getImm()[0])}")

        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) & ins.getImm()[0])

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execori__(self,ins):
        assert ins.getOp() == "ORI", "Expected opcode ORI Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One Imm expected"

        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) | ins.getImm()[0])

        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} or {hex(ins.getImm()[0])} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) | ins.getImm()[0])}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execxori__(self,ins):
        assert ins.getOp() == "XORI", "Expected opcode XORI Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One Imm expected"

        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) ^ ins.getImm()[0])

        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} xor {hex(ins.getImm()[0])} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) ^ ins.getImm()[0])}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execslli__(self,ins):
        assert ins.getOp() == "SLLI", "Expected opcode SLLI Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One Imm expected"

        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) <<  ins.getImm()[0])
        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} << {hex(ins.getImm()[0])} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) << ins.getImm()[0])}")

        nextRelAddr = ins.getRelAddr() + 4

        return nextRelAddr

    def __execsrli__(self,ins):
        assert ins.getOp() == "SRLI", "Expected opcode SRLI Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One Imm expected"
        assert ins.getImm()[0] >=0, "Positive Shift value expected"+str(ins.getImm()[0])

        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) >> ins.getImm()[0])
        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} >> {hex(ins.getImm()[0])} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) >> ins.getImm()[0])}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsrai__(self,ins):
        assert ins.getOp() == "SRAI", "Expected opcode SRAI Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One Imm expected"

        signBit = self.triscRegs.__readReg__(ins.getSrcInt()[0]) & 0x80000000 == 0x80000000
        if(signBit):        self.triscRegs.__writeReg__(ins.getDstInt()[0], ((0xFFFFFFFF >> (32-ins.getImm()[0])) << (32-ins.getImm()[0])) + (self.triscRegs.__readReg__(ins.getSrcInt()[0]) >> ins.getImm()[0]))
        else:               self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) >> ins.getImm()[0])

        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} >> {hex(ins.getImm()[0])} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) >> ins.getImm()[0])}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execslti__(self,ins):
        assert ins.getOp() == "SLTI", "Expected opcode SLT Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One Imm expected"

        self.triscRegs.__writeReg__(ins.getDstInt()[0], 0)
        # 32-bit signed comparison
        if(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[0]),'i',32) < convert_to_format(ins.getImm()[0],'i',32)):
            self.triscRegs.__writeReg__(ins.getDstInt()[0], 1)
        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} < {hex(ins.getImm()[0])} = {hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsltiu__(self,ins):
        assert ins.getOp() == "SLTIU", "Expected opcode SLTIU Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One Imm expected"

        self.triscRegs.__writeReg__(ins.getDstInt()[0], 0)
        if(self.triscRegs.__readReg__(ins.getSrcInt()[0]) < ins.getImm()[0]):
            self.triscRegs.__writeReg__(ins.getDstInt()[0], 1)
        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} < {hex(ins.getImm()[0])} = {hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    #I-Type Load
    def __execlw__(self, ins):
        assert ins.getOp() == "LW", "Expected opcode LW Received " + str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "`One  Imm expected"
        # self.triscRegs.__printReg__('riscgpr')

        regType,regIndex = self.triscRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        cfgRegType,cfgRegIndex = self.ttSplRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        if(regType == '') and (cfgRegType == ''):  # Not MMR
            assert regIndex == -1, "Unrecognized combination"

            # assert self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) , "Uninitialized memory [" + str(ins.getSrcInt()[0] + ins.getImm()[0]) +"]"
            if(not self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])):
                print("TODO: Initializing memory which should have been available. Addr=",hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]), " Value = 0xFF")
                self.memData.__writeMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0], 0xFF)

            self.triscRegs.__writeReg__(ins.getDstInt()[0], self.memData.__readMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]))
            if self.debug & 0x8: print(f"__execlw:REG[{hex(ins.getDstInt()[0])}]=MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]")

            if(self.debug & 0x10):
                print(f"\t{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]={hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))}")

        elif(cfgRegType == 'cfg'):              # Write to MMR (Tensix)
            assert cfgRegIndex != -1, "Config Reg Index cannot be uninitialized"
            if self.debug & 0x8:
                print(f"__execlw(cfg):{cfgRegType} , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")

            if "DEST_TARGET_REG_CFG_MATH" == self.ttSplRegs.getCfgRegUpdateClass(cfgRegIndex): # DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32
                print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} DEST_TARGET_REG_CFG cfg[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")
                self.triscRegs.__writeReg__(ins.getDstInt()[0], self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))
                # if self.triscRegs.__readReg__(ins.getSrcInt()[1]) >=512:  bankUpd[3] = 1
            else:
                print(f"WARNING: Unhandled special register (TENSIX) {cfgRegType}[{cfgRegIndex}], Memlocation={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}" )
        elif(cfgRegType in ['mop','mopSync', 'idleSync']):              # Write to MMR (TRISC)
            if self.debug & 0x8: print(f"__execlw:{cfgRegType} , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            match cfgRegType:
                case 'mop':       
                    assert cfgRegIndex >=0 and cfgRegIndex <= 8, f"MOP Reg Index out of range. cfgRegIndex={cfgRegIndex}"
                    if self.debug & 0x8:    print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} mop[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")
                    self.triscRegs.__writeReg__(ins.getDstInt()[0], self.ttSplRegs.__readReg__(self.threadId*64 + cfgRegIndex, cfgRegType))
                case 'mopSync':   
                    assert cfgRegIndex == 0, f"Only one mopSync register supported. Unexpected Index={cfgRegIndex}"
                    if(self.ttSplRegs.__readReg__(self.threadId + cfgRegIndex, cfgRegType) != 0):
                        print(f"Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread[{self.threadId}] (TENSIX) MOPSync register read as non-zero value {hex(self.ttSplRegs.__readReg__(self.threadId + cfgRegIndex, cfgRegType))}. Stalling")
                        #TODO: Implement MOPSync stall behavior
                        # nextRelAddr = ins.getRelAddr() - 4
                        # return nextRelAddr
                        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.ttSplRegs.__readReg__(self.threadId + cfgRegIndex, cfgRegType))
                    else:
                        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.ttSplRegs.__readReg__(self.threadId + cfgRegIndex, cfgRegType))
                        print(f"Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread[{self.threadId}] (TENSIX) MOPSync register read as zero value {hex(self.ttSplRegs.__readReg__(self.threadId + cfgRegIndex, cfgRegType))}. Continuing")
                case 'idleSync':  
                    assert cfgRegIndex == 0, f"Only one idleSync register supported. Unexpected Index={cfgRegIndex}"
                    #TODO: Implement idleSync stall behavior
                    self.triscRegs.__writeReg__(ins.getDstInt()[0], self.ttSplRegs.__readReg__(self.threadId + cfgRegIndex, cfgRegType))
                case _:          
                    assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
        elif(cfgRegType in ['instrBuffer', 'ttsemaphores']):  # Tensix Special Regs
            if self.debug & 0x8: print(f"__execlw({cfgRegType}) , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            match cfgRegType:
                case 'instrBuffer':  assert cfgRegIndex == 0, f"InstrBuffer Reg Index out of range. cfgRegIndex={cfgRegIndex}"
                case 'ttsemaphores': assert cfgRegIndex >= 0 and cfgRegIndex <= 31, f"ttsemaphores Reg Index out of range. cfgRegIndex={cfgRegIndex}"
                case _:              assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
            if self.debug & 0x8:    print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} {cfgRegType}[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")
            self.triscRegs.__writeReg__(ins.getDstInt()[0], self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))
        else:   # Read from MMR
            self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(regIndex, regType))
            if(self.debug & 0x10):
                print(f"\t{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]MMR[{hex(self.triscRegs.__readReg__(regIndex,regType))}]={hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execlh__(self,ins):
        assert ins.getOp() == "LH", "Expected opcode LH Received " + str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        regType,regIndex = self.triscRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        cfgRegType,cfgRegIndex = self.ttSplRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        if(regType == '') and (cfgRegType == ''):  # Not MMR
            assert regIndex == -1, "Unrecognized combination"

            # assert self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) , "Uninitialized memory [" + str(ins.getSrcInt()[0] + ins.getImm()[0]) +"]"
            if(not self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])):
                print("TODO: Initializing memory which should have been available. Addr=",hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]), " Value = 0xFF")
                self.memData.__writeMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0], 0xFF)

            signBit = self.memData.__readMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) & 0x8000 == 0x8000
            if(signBit):    self.triscRegs.__writeReg__(ins.getDstInt()[0],self.memData.__readMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) | 0xFFFF0000)
            else:           self.triscRegs.__writeReg__(ins.getDstInt()[0],self.memData.__readMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) & 0xFFFF)

            if self.debug & 0x8: print(f"__execlh:{regType} , {regIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")

            if(self.debug & 0x10):
                if signBit: print(f"\t{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]={hex(self.triscRegs.__readReg__(ins.getDstInt()[0]) | 0xFFFF0000)}")
                else:       print(f"\t{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]={hex(self.triscRegs.__readReg__(ins.getDstInt()[0]) & 0xFFFF)}")

        elif(cfgRegType == 'cfg'):              # Write to MMR (Tensix)
            assert cfgRegIndex != -1, "Config Reg Index cannot be uninitialized"
            if self.debug & 0x8:
                print(f"__execlh:{cfgRegType} , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            if "DEST_TARGET_REG_CFG_MATH" == self.ttSplRegs.getCfgRegUpdateClass(cfgRegIndex): # DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32
                print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} DEST_TARGET_REG_CFG cfg[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")

                signBit = self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) & 0x8000 == 0x8000
                if(signBit):    self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) | 0xFFFF0000)
                else:           self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) & 0xFFFF)
                # if self.triscRegs.__readReg__(ins.getSrcInt()[1]) >=512:  bankUpd[3] = 1
            else:
                assert cfgRegIndex != -1 , f"Register Type {cfgRegType} can't be uninitialized"
                print(f"WARNING: Unhandled special register (TENSIX) {cfgRegType}[{cfgRegIndex}], Memlocation={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}" )
        elif(cfgRegType == 'mop'):              # Write to MMR (TRISC)
            if self.debug & 0x8: print(f"__execlw:{cfgRegType} , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            if(cfgRegIndex >=0 and cfgRegIndex <= 8): # MOP
                if self.debug & 0x8:    print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} mop[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")
                signBit = self.ttSplRegs.__readReg__(self.threadId*64 + cfgRegIndex, cfgRegType) & 0x8000 == 0x8000
                if(signBit):    self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(self.threadId*64 + cfgRegIndex, cfgRegType) | 0xFFFF0000)
                else:           self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(self.threadId*64 + cfgRegIndex, cfgRegType) & 0xFFFF)
                # self.triscRegs.__writeReg__(ins.getDstInt()[0], self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))
            else:
                assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
        elif(cfgRegType == 'instrBuffer'):      # Write to MMR (TENSIX)
            if(cfgRegIndex ==0):  #InstrBuffer
                if self.debug & 0x8:    print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} instrbuf[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")
                signBit = self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) & 0x8000 == 0x8000
                if(signBit):    self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) | 0xFFFF0000)
                else:           self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) & 0xFFFF)
                # self.triscRegs.__writeReg__(ins.getDstInt()[0], self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))
            else:
                assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
        else:   # Read from MMR
            self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(regIndex, regType) & 0xFFFF)
            if(self.debug & 0x10):
                print(f"\t{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]MMR[{hex(self.triscRegs.__readReg__(regIndex,regType))}]={hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execlb__(self,ins):
        assert ins.getOp() == "LB", "Expected opcode LB Received " + str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        regType,regIndex = self.triscRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        cfgRegType,cfgRegIndex = self.ttSplRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        if(regType == '') and (cfgRegType == ''):  # Not MMR
            assert regIndex == -1, "Unrecognized combination"

            # assert self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) , "Uninitialized memory [" + str(ins.getSrcInt()[0] + ins.getImm()[0]) +"]"
            if(not self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])):
                print("TODO: Initializing memory which should have been available. Addr=",hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]), " Value = 0xFF")
                self.memData.__writeMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0], 0xFF)

            signBit = self.memData.__readMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) & 0x80 == 0x80
            if(signBit):    self.triscRegs.__writeReg__(ins.getDstInt()[0],self.memData.__readMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) | 0xFFFFFF00)
            else:           self.triscRegs.__writeReg__(ins.getDstInt()[0],self.memData.__readMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) & 0xFF)

            if(self.debug & 0x10):
                if signBit: print(f"\t{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]={hex(self.triscRegs.__readReg__(ins.getDstInt()[0]) | 0xFF000000)}")
                else:       print(f"\t{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]={hex(self.triscRegs.__readReg__(ins.getDstInt()[0]) & 0xFF)}")

        elif(cfgRegType == 'cfg'):              # Write to MMR (Tensix)
            assert cfgRegIndex != -1, "Config Reg Index cannot be uninitialized"
            if self.debug & 0x8:
                print(f"__execlb:{cfgRegType} , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            if "DEST_TARGET_REG_CFG_MATH" == self.ttSplRegs.getCfgRegUpdateClass(cfgRegIndex): # DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32
                print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} DEST_TARGET_REG_CFG cfg[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")

                signBit = self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) & 0x80 == 0x80
                if(signBit):    self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) | 0xFFFFFF00)
                else:           self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) & 0xFF)

                # if self.triscRegs.__readReg__(ins.getSrcInt()[1]) >=512:  bankUpd[3] = 1
            else:
                assert cfgRegIndex != -1 , f"Register Type {cfgRegType} can't be uninitialized"
                print(f"WARNING: Unhandled special register (TENSIX) {cfgRegType}[{cfgRegIndex}], Memlocation={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}" )
        elif(cfgRegType == 'mop'):              # Write to MMR (TRISC)
            if self.debug & 0x8: print(f"__execlw:{cfgRegType} , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            if(cfgRegIndex >=0 and cfgRegIndex <= 8): # MOP
                if self.debug & 0x8:    print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} mop[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")
                signBit = self.ttSplRegs.__readReg__(self.threadId*64 + cfgRegIndex, cfgRegType) & 0x80 == 0x80
                if(signBit):    self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(self.threadId*64 + cfgRegIndex, cfgRegType) | 0xFFFFFF00)
                else:           self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(self.threadId*64 + cfgRegIndex, cfgRegType) & 0xFF)
                # self.triscRegs.__writeReg__(ins.getDstInt()[0], self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))
            else:
                assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
        elif(cfgRegType == 'instrBuffer'):      # Write to MMR (TENSIX)
            if(cfgRegIndex ==0):  #InstrBuffer
                if self.debug & 0x8:    print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} instrbuf[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")
                signBit = self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) & 0x80 == 0x80
                if(signBit):    self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) | 0xFFFFFF00)
                else:           self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) & 0xFF)
                # self.triscRegs.__writeReg__(ins.getDstInt()[0], self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))
            else:
                assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
        else:   # Read from MMR
            self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(regIndex, regType) & 0xFF)
            if(self.debug & 0x10):
                print(f"\t{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]MMR[{hex(self.triscRegs.__readReg__(regIndex,regType))}]={hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execlhu__(self,ins):
        assert ins.getOp() == "LHU", "Expected opcode LHU.  Received " + str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        regType,regIndex = self.triscRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        cfgRegType,cfgRegIndex = self.ttSplRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        if(regType == '') and (cfgRegType == ''):  # Not MMR
            assert regIndex == -1, "Unrecognized combination"

            # assert self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) , "Uninitialized memory [" + str(ins.getSrcInt()[0] + ins.getImm()[0]) +"]"
            if(not self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])):
                print("TODO: Initializing memory which should have been available. Addr=",hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]), " Value = 0xFF")
                self.memData.__writeMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0], 0xFF)

            self.triscRegs.__writeReg__(ins.getDstInt()[0], (self.memData.__readMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])) & 0xFFFF)

            if(self.debug & 0x10):
                print(f"\t{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]={hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))}")

        elif(cfgRegType == 'cfg'):              # Write to MMR (Tensix)
            assert cfgRegIndex != -1, "Config Reg Index cannot be uninitialized"
            if self.debug & 0x8:
                print(f"__execlb:{cfgRegType} , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            if "DEST_TARGET_REG_CFG_MATH" == self.ttSplRegs.getCfgRegUpdateClass(cfgRegIndex): # DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32
                print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} DEST_TARGET_REG_CFG cfg[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")
                self.triscRegs.__writeReg__(ins.getDstInt()[0], self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) & 0xFFFF)
                # if self.triscRegs.__readReg__(ins.getSrcInt()[1]) >=512:  bankUpd[3] = 1
            else:
                assert cfgRegIndex != -1 , f"Register Type {cfgRegType} can't be uninitialized"
                print(f"WARNING: Unhandled special register (TENSIX) {cfgRegType}[{cfgRegIndex}], Memlocation={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}" )
        elif(cfgRegType == 'mop'):              # Write to MMR (TRISC)
            if self.debug & 0x8: print(f"__execlw:{cfgRegType} , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            if(cfgRegIndex >=0 and cfgRegIndex <= 8): # MOP
                if self.debug & 0x8:    print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} mop[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")
                self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(self.threadId*64 + cfgRegIndex, cfgRegType) & 0xFFFF)
            else:
                assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
        elif(cfgRegType == 'instrBuffer'):      # Write to MMR (TENSIX)
            if(cfgRegIndex ==0):  #InstrBuffer
                if self.debug & 0x8:    print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} instrbuf[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")
                self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) & 0xFFFF)
            else:
                assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
        else:   # Read from MMR
            self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(regIndex, regType) & 0xFFFF)
            if(self.debug & 0x10):
                print(f"\t{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]MMR[{hex(self.triscRegs.__readReg__(regIndex,regType))}]={hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr


    def __execlbu__(self,ins):
        assert ins.getOp() == "LBU", "Expected opcode LBU Received " + str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        regType,regIndex = self.triscRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        cfgRegType,cfgRegIndex = self.ttSplRegs.__ismmr__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])
        if(regType == '') and (cfgRegType == ''):  # Not MMR
            assert regIndex == -1, "Unrecognized combination"

            # assert self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]) , "Uninitialized memory [" + str(ins.getSrcInt()[0] + ins.getImm()[0]) +"]"
            if(not self.memData.isInitialized(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])):
                print("TODO: Initializing memory which should have been available. Addr=",hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0]), " Value = 0xFF")
                self.memData.__writeMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0], 0xFF)

            self.triscRegs.__writeReg__(ins.getDstInt()[0], (self.memData.__readMem__(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])) & 0xFF)

            if(self.debug & 0x10):
                print(f"\t{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]={hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))}")

        elif(cfgRegType == 'cfg'):              # Write to MMR (Tensix)
            assert cfgRegIndex != -1, "Config Reg Index cannot be uninitialized"
            if self.debug & 0x8:
                print(f"__execlb:{cfgRegType} , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            if "DEST_TARGET_REG_CFG_MATH" == self.ttSplRegs.getCfgRegUpdateClass(cfgRegIndex): # DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32
                print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} DEST_TARGET_REG_CFG cfg[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")
                self.triscRegs.__writeReg__(ins.getDstInt()[0], self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) & 0xFF)
                # if self.triscRegs.__readReg__(ins.getSrcInt()[1]) >=512:  bankUpd[3] = 1
            else:
                assert cfgRegIndex != -1 , f"Register Type {cfgRegType} can't be uninitialized"
                print(f"WARNING: Unhandled special register (TENSIX) {cfgRegType}[{cfgRegIndex}], Memlocation={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}" )
        elif(cfgRegType == 'mop'):              # Write to MMR (TRISC)
            if self.debug & 0x8: print(f"__execlw:{cfgRegType} , {cfgRegIndex}, {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}")
            if(cfgRegIndex >=0 and cfgRegIndex <= 8): # MOP
                if self.debug & 0x8:    print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} mop[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")
                self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(self.threadId*64 + cfgRegIndex, cfgRegType) & 0xFF)
            else:
                assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
        elif(cfgRegType == 'instrBuffer'):      # Write to MMR (TENSIX)
            if(cfgRegIndex ==0):  #InstrBuffer
                if self.debug & 0x8:    print(f"Reading from (TENSIX) Thread[{self.threadId}]:{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])} instrbuf[{hex(cfgRegIndex)}]={hex(self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType))}")
                self.triscRegs.__writeReg__(ins.getDstInt()[0],self.ttSplRegs.__readReg__(cfgRegIndex, cfgRegType) & 0xFF)
            else:
                assert False, "Unknown reg Index=" + str(cfgRegIndex) + ",Type=" + str(cfgRegType)
        else:   # Read from MMR
            self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(regIndex, regType) & 0xFF)
            if(self.debug & 0x10):
                print(f"\t{ins.getOp()}: MEM[{hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) + ins.getImm()[0])}]MMR[{hex(self.triscRegs.__readReg__(regIndex,regType))}]={hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    #U-type
    def __execlui__(self, ins):
        assert ins.getOp() == "LUI", "Expected opcode LUI Received " + str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getImm()) == 1, "One Imm expected"

        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(ins.getImm()[0])} << 12 = {hex(ins.getImm()[0] << 12)}")
        self.triscRegs.__writeReg__(ins.getDstInt()[0], (ins.getImm()[0]) << 12)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execauipc__(self, ins):
        assert ins.getOp() == "AUIPC", "Expected opcode AUIPC Received " + str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getImm()) == 1, "One Imm expected"

        self.triscRegs.__writeReg__(ins.getDstInt()[0], ins.getRelAddr() + (ins.getImm()[0] << 12))
        if(self.debug & 0x10):
            print(f'\t{ins.getOp()}: Reg[{hex(ins.getDstInt()[0])}] {hex(ins.getRelAddr())} + {hex(ins.getImm()[0])} << 12 = {hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))}')

        nextRelAddr = ins.getRelAddr() + 4

        return nextRelAddr

    def __execbeq__(self,ins):
        assert ins.getOp() == "BEQ", "Expected opcode BEQ Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        if(ins.getRelAddr() + ins.getImm()[0] in forceSkipJump):
            nextRelAddr = ins.getRelAddr() + 4
            if(self.debug & 0x10):
                print("\t{4}: {0}: FORCE SKIP JUMP NextAddr={1} if {2} < {3}".format(ins.getOp(), hex(nextRelAddr), hex(self.triscRegs.__readReg__(ins.getSrcInt()[0])), hex(ins.getSrcInt()[1]), ins.getThread()) )
            return nextRelAddr
        if(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[0], 'riscgpr'), 'i', 32) == convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[1], 'riscgpr'), 'i', 32)):
            nextRelAddr = ins.getRelAddr() + ins.getImm()[0]
            if(self.debug & 0x10):
                print(f"\tAddr:{ins.getRelAddr()} {ins.getOp()}: {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[0]),'i',32))} = {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[1]),'i',32))}. JMP {hex(nextRelAddr)}")
        else:
            nextRelAddr = ins.getRelAddr() + 4
            if(self.debug & 0x10):
                print(f"\tAddr:{ins.getRelAddr()} {ins.getOp()}: {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[0]),'i',32))} != {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[1]),'i',32))}. JMP[PCINCR] {hex(nextRelAddr)}")

        return nextRelAddr

    def __execbne__(self,ins):
        assert ins.getOp() == "BNE", "Expected opcode BNE Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        if(ins.getRelAddr() + ins.getImm()[0] in forceSkipJump):
            nextRelAddr = ins.getRelAddr() + 4
            if(self.debug & 0x10):
                print("\t{4}: {0}: FORCE SKIP JUMP NextAddr={1} if {2} >= {3}".format(ins.getOp(), hex(nextRelAddr), hex(self.triscRegs.__readReg__(ins.getSrcInt()[0])), hex(self.triscRegs.__readReg__(ins.getSrcInt()[1])), ins.getThread()) )
            return nextRelAddr
        if(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[0]), 'i', 32) != convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[1]), 'i', 32)):
            nextRelAddr = ins.getRelAddr() + ins.getImm()[0]
            if(self.debug & 0x10):
                print(f"\tAddr:{ins.getRelAddr()} {ins.getOp()}: {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[0]),'i',32))} != {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[1]),'i',32))}. JMP {hex(nextRelAddr)}")
        else:
            nextRelAddr = ins.getRelAddr() + 4
            if(self.debug & 0x10):
                print(f"\tAddr:{ins.getRelAddr()} {ins.getOp()}: {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[0]),'i',32))} = {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[1]),'i',32))}. JMP[PCINCR] {hex(nextRelAddr)}")

        return nextRelAddr

    def __execblt__(self,ins):
        assert ins.getOp() == "BLT", "Expected opcode BLT Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        if(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[0], 'riscgpr'), 'i', 32) < convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[1], 'riscgpr'), 'i', 32)):
            nextRelAddr = ins.getRelAddr() + ins.getImm()[0]
            if(self.debug & 0x10):
                print(f"\tAddr:{ins.getRelAddr()} {ins.getOp()}: {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[0]),'i',32))} < {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[1]),'i',32))}. JMP {hex(nextRelAddr)}")
        else:
            nextRelAddr = ins.getRelAddr() + 4
            if(self.debug & 0x10):
                print(f"\tAddr:{ins.getRelAddr()} {ins.getOp()}: {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[0]),'i',32))} !< {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[1]),'i',32))}. JMP[PCINCR] {hex(nextRelAddr)}")

        return nextRelAddr

    def __execbge__(self,ins):
        assert ins.getOp() == "BGE", "Expected opcode BGE Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        if(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[0], 'riscgpr'), 'i', 32) >= convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[1], 'riscgpr'), 'i', 32)):
            nextRelAddr = ins.getRelAddr() + ins.getImm()[0]
            if(self.debug & 0x10):
                print(f"\tAddr:{ins.getRelAddr()} {ins.getOp()}: {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[0]),'i',32))} >= {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[1]),'i',32))}. JMP {hex(nextRelAddr)}")
        else:
            nextRelAddr = ins.getRelAddr() + 4
            if(self.debug & 0x10):
                print(f"\tAddr:{ins.getRelAddr()} {ins.getOp()}: {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[0]),'i',32))} !>= {hex(convert_to_format(self.triscRegs.__readReg__(ins.getSrcInt()[1]),'i',32))}. JMP[PCINCR] {hex(nextRelAddr)}")

        return nextRelAddr


    def __execbltu__(self,ins):
        assert ins.getOp() == "BLTU", "Expected opcode BLTU Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        if(self.triscRegs.__readReg__(ins.getSrcInt()[0], 'riscgpr') < self.triscRegs.__readReg__(ins.getSrcInt()[1], 'riscgpr')):
            nextRelAddr = ins.getRelAddr() + ins.getImm()[0]
            if(self.debug & 0x10):
                print(f"\tAddr:{ins.getRelAddr()} {ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} < {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))}. JMP {hex(nextRelAddr)}")
        else:
            nextRelAddr = ins.getRelAddr() + 4
            if(self.debug & 0x10):
                print(f"\tAddr:{ins.getRelAddr()} {ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} !< {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))}. JMP[PCINCR] {hex(nextRelAddr)}")

        return nextRelAddr


    def __execbgeu__(self,ins):
        assert ins.getOp() == "BGEU", "Expected opcode BGEU Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        if(ins.getRelAddr() + ins.getImm()[0] in forceSkipJump):
            nextRelAddr = ins.getRelAddr() + 4
            if(self.debug & 0x10):
                print("\t{4}: {0}: FORCE SKIP JUMP NextAddr={1} if {2} >= {3}".format(ins.getOp(), hex(nextRelAddr), hex(self.triscRegs.__readReg__(ins.getSrcInt()[0])), hex(self.triscRegs.__readReg__(ins.getSrcInt()[1])), ins.getThread()) )
            return nextRelAddr
        if(self.triscRegs.__readReg__(ins.getSrcInt()[0], 'riscgpr') >= self.triscRegs.__readReg__(ins.getSrcInt()[1],'riscgpr')):
            nextRelAddr = ins.getRelAddr() + ins.getImm()[0]
            if(self.debug & 0x10):
                print(f"\tAddr:{hex(ins.getRelAddr())} {ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} >= {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))}. JMP {hex(nextRelAddr)}")
        else:
            nextRelAddr = ins.getRelAddr() + 4
            if(self.debug & 0x10):
                print(f"\tAddr:{hex(ins.getRelAddr())} {ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} !>= {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))}. JMP[PCINCR] {hex(nextRelAddr)}")

        return nextRelAddr

    def __execjal__(self,ins):
        assert ins.getOp() == "JAL", "Expected opcode JAL Received " + str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        if(ins.getRelAddr() + ins.getImm()[0] in forceSkipJump):
            nextRelAddr = ins.getRelAddr() + 4
            if(self.debug & 0x10):
                print("\t{3}: {0}: FORCE SKIP JUMP NextAddr={1} if {2} !=0".format(ins.getOp(), hex(ins.getRelAddr() + ins.getImm()[0]) , hex(self.triscRegs.__readReg__(ins.getDstInt()[0])), ins.getThread()))
            return nextRelAddr

        #TODO: Destination register of jal is sometimes x0, the zero register which should be read-only
        if(ins.getDstInt()[0] != 0):
            self.triscRegs.__writeReg__(ins.getDstInt()[0], ins.getRelAddr() + 4)
            nextRelAddr = ins.getRelAddr() + ins.getImm()[0]
            if(self.debug & 0x10):
                print(f"\tAddr:{hex(ins.getRelAddr())}: {ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))} !=0 JMP {hex(nextRelAddr)}")
        else:
            if (ins.getImm()[0] == 0):
                print(f"WARNING: JAL with x0 as destination and 0 offset at addr {hex(ins.getRelAddr())}. Treating this as End of Kernel.")
                nextRelAddr = 0x0
            else:
                nextRelAddr = ins.getRelAddr() + ins.getImm()[0]
            if(self.debug & 0x10):
                print(f"\tAddr:{hex(ins.getRelAddr())}: {ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))} ==0 JMP[PCINCR] {hex(nextRelAddr)}")

        return nextRelAddr

    def __execjalr__(self,ins):
        assert ins.getOp() == "JALR", "Expected opcode JALR Received " + str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert len(ins.getImm()) == 1, "One  Imm expected"

        if(self.debug & 0x10):
            print(f"Addr:{hex(ins.getRelAddr())}: {ins.getOp()}: Jump Target Address calculated from SrcReg[{hex(ins.getSrcInt()[0])}]={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0], 'riscgpr'))}, Imm={hex(ins.getImm()[0])}, Target={hex(self.triscRegs.__readReg__(ins.getSrcInt()[0], 'riscgpr') + ins.getImm()[0])}")

        #TODO: Destination register of jalr is sometimes x0, the zero register which should be read-only
        # Step 1: Calculate next address = srcReg + imm
        nextRelAddr = self.triscRegs.__readReg__(ins.getSrcInt()[0], 'riscgpr') + ins.getImm()[0]
        # Step 2: Write return address to dstReg 
        if(ins.getDstInt()[0] != 0):
            self.triscRegs.__writeReg__(ins.getDstInt()[0], ins.getRelAddr() + 4)

        if(self.debug & 0x10):
            print(f"\tAddr:{ins.getRelAddr()}: {ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getDstInt()[0]))} !=0 RET {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0], 'riscgpr'))} + {hex(ins.getImm()[0])} = {hex(nextRelAddr)}")
            print(f"\tAddr:{hex(ins.getRelAddr())}: {ins.getOp()}: Next Address at DstReg[{hex(ins.getDstInt()[0])}]={hex(self.triscRegs.__readReg__(ins.getDstInt()[0], 'riscgpr'))}")

        return nextRelAddr

    def __execmul__(self, ins):
        assert ins.getOp() == "MUL", "Expected opcode MUL Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 2, "Two Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getSrcInt()[0]) * self.triscRegs.__readReg__(ins.getSrcInt()[1]))
        if(self.debug & 0x10):
            print(f"\t{ins.getOp()}: {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]))} * {hex(self.triscRegs.__readReg__(ins.getSrcInt()[1]))} = {hex(self.triscRegs.__readReg__(ins.getSrcInt()[0]) * self.triscRegs.__readReg__(ins.getSrcInt()[1]))}")

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execfence__(self, ins):
        assert ins.getOp() == "FENCE", "Expected opcode FENCE Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        assert len(ins.getAttr()) == 2, "Two attribs expected. Received " + str(len(ins.getAttr()))

        #TODO: Order memory writes and/or memory reads. Needed?
        nextRelAddr = ins.getRelAddr() + 4

        return nextRelAddr

    def __execcsrrs__(self, ins):
        assert ins.getOp() == "CSRRS" or ins.getOp() == "CSRRC" or ins.getOp() == "CSRRW", "Expected opcode CSRRS/C/W Recieved "+ str(ins.getOp())
        assert len(ins.getDstInt()) == 1, "One Dst expected"
        assert len(ins.getSrcInt()) == 1, "One Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        assert len(ins.getAttr()) == 1, "One attrib expected. Received " + str(len(ins.getAttr()))

        # self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getAttr()['csr'], 'csr'), 'riscgpr')
        #TODO: Needs review
        if(ins.getDstInt()[0] != 0):
            self.triscRegs.__writeReg__(ins.getDstInt()[0], self.triscRegs.__readReg__(ins.getAttr()['csr'], 'csr'), 'riscgpr')

        match ins.getOp():
            case "CSRRW":       self.triscRegs.__writeReg__(ins.getAttr()['csr'], self.triscRegs.__readReg__(ins.getSrcInt()[0],'riscgpr'), 'csr')                                                       # Overwrite CSR
            case "CSRRS":       self.triscRegs.__writeReg__(ins.getAttr()['csr'], self.triscRegs.__readReg__(ins.getAttr()['csr'],'csr') & ins.getSrcInt()[0], 'csr')         # Set CSR based on mask. Mask = 0xFFFFFFFF implies no change to CSR
            case "CSRRC":       self.triscRegs.__writeReg__(ins.getAttr()['csr'], self.triscRegs.__readReg__(ins.getAttr()['csr'],'csr') & (not(ins.getSrcInt()[0])), 'csr')  # Clr CSR based on mask. Mask = 0 implies no change to CSR

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def execRIns(self,ins, cycle):
        nextAddr = -1

        match ins.getOp():
            case "ADDI":                 nextAddr = self.__execaddi__(ins)
            case "SUBI":                 nextAddr = self.__execsubi__(ins)
            case "XORI":                 nextAddr = self.__execxori__(ins)
            case "ORI":                  nextAddr = self.__execori__(ins)
            case "ANDI":                 nextAddr = self.__execandi__(ins)
            case "SLLI":                 nextAddr = self.__execslli__(ins)
            case "SRLI":                 nextAddr = self.__execsrli__(ins)
            case "SRAI":                 nextAddr = self.__execsrai__(ins)
            case "SLTI":                 nextAddr = self.__execslti__(ins)
            case "SLTIU":                nextAddr = self.__execsltiu__(ins)

            case "ADD":                 nextAddr = self.__execadd__(ins)
            case "SUB":                 nextAddr = self.__execsub__(ins)
            case "XOR":                 nextAddr = self.__execxor__(ins)
            case "OR":                  nextAddr = self.__execor__(ins)
            case "AND":                 nextAddr = self.__execand__(ins)
            case "SLL":                 nextAddr = self.__execsll__(ins)
            case "SRL":                 nextAddr = self.__execsrl__(ins)
            case "SRA":                 nextAddr = self.__execsra__(ins)
            case "SLT":                 nextAddr = self.__execslt__(ins)
            case "SLTU":                nextAddr = self.__execsltu__(ins)

            case "LB":                   nextAddr = self.__execlb__(ins)
            case "LH":                   nextAddr = self.__execlh__(ins)
            case "LBU":                  nextAddr = self.__execlbu__(ins)
            case "LHU":                  nextAddr = self.__execlhu__(ins)
            case "LW":                   nextAddr = self.__execlw__(ins)

            case "SB":                   nextAddr = self.__execsb__(ins)
            case "SH":                   nextAddr = self.__execsh__(ins)
            case "SW":                   nextAddr = self.__execsw__(ins)

            case "BEQ":                  nextAddr = self.__execbeq__(ins)
            case "BNE":                  nextAddr = self.__execbne__(ins)
            case "BLT":                  nextAddr = self.__execblt__(ins)
            case "BGE":                  nextAddr = self.__execbge__(ins)
            case "BLTU":                 nextAddr = self.__execbltu__(ins)
            case "BGEU":                 nextAddr = self.__execbgeu__(ins)

            case "JAL":                  nextAddr = self.__execjal__(ins)
            case "JALR":                 nextAddr = self.__execjalr__(ins)

            case "LUI":                  nextAddr = self.__execlui__(ins)
            case "AUIPC":                nextAddr = self.__execauipc__(ins)

            case "MUL":                  nextAddr = self.__execmul__(ins)
            case "FENCE":                nextAddr = self.__execfence__(ins)

            case "CSRRW":                nextAddr = self.__execcsrrs__(ins)
            case "CSRRC":                nextAddr = self.__execcsrrs__(ins)
            case "CSRRS":                nextAddr = self.__execcsrrs__(ins)

            case _:
                print("WARNING: Unsupported RISC Instruction ", ins.getOp())
                nextAddr = ins.getRelAddr() + 4
                ins.printInstr(self.threadId)

        assert nextAddr != -1 , "Error in execution"
        if (self.debug & 0x1):
            print(f"RFunctional Cycle:{cycle} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} ", end='')
            ins.printInstr(self.threadId)
        return nextAddr

    def readInstructionBufMem(self):
        # Assumes 1:1 mapping. If this assumption changes the control flow management needs to be updated accordingly
        # If 1:n mapping in future, convert the return to a list
        insBuf = self.ttSplRegs.__readReg__(0,'instrBuffer')
        if (insBuf == -1):         return None
        else:
            self.ttSplRegs.__writeReg__(0,-1, 'instrBuffer')
            return insBuf

    def decodeInstructionBufMem(self):
        insBuf = self.readInstructionBufMem()
        if (insBuf == None):        return None
        else:                       return isaFunctions.decodeInstr(insBuf, decoded_instruction.instruction_kind.ttqs, True, ttISA = self.args['ttISA'])# insBuf
