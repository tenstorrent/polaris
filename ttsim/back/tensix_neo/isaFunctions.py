#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import copy
import typing

from enum import IntFlag, auto, IntEnum

# sys.path.append('./binutils-playground/py')
# from instructions import decode_instruction
import ttsim.front.llk.instructions as instructions
import ttsim.front.llk.tensix as tensix
import ttsim.front.llk.rv32 as rv32
import ttsim.front.llk.read_elf as read_elf
import ttsim.front.llk.decoded_instruction as decoded_instruction

# TODO: Change this mapping to come from test or test specification if possible
class THREADMAP(IntEnum):
    UNPACKER_THREAD = 0
    MATH_THREAD     = auto()
    SFPU_THREAD     = auto()
    PACKER_THREAD   = auto()
    NUM_CONTEXTS    = auto()

class CLEARDVALID_dest_pulse_last_MASKS(IntFlag):
    ZERO     = 0x0    # 0x0
    UNPACKER = auto() # 0x1
    MATH     = auto() # 0x2
    SFPU     = auto() # 0x4
    PACKER   = auto() # 0x8

class regIndex(IntEnum):
    srcA = 0
    srcB = auto() # 1
    srcS = auto() # 2
    dst  = auto() # 3

class valueStatus(IntEnum):
    IGNORE = -2
    UNSET = -1

class instr(decoded_instruction.decoded_instruction, decoded_instruction.operands):
    def __init__(self, source= None):
        self.addr                           = -1
        self.coreId                         = -1
        self.threadId                       = -1
        self.insId                          = -1
        self.pipeDelay                      = -1
        self.exPipe                         = None
        self.operands                       = decoded_instruction.operands()
        self.vldUpdMask                     = {}
        self.bankUpdMask                    = {}
        self.condChkVldUpdVal               = {}
        self.condWriVldUpdVal               = {}
        self.pipeBankCtrl                   = {}
        self.srcPipes                       = []
        self.dstPipes                       = []
        self.mnemonic                       = ""

        self.srcFormat                      = None
        self.dstFormat                      = None
        self.numDatums                      = 0

        self.memInfo                        = {}
        self.eventInfo                      = {}
        if source is not None:
            self.__dict__.update(source.__dict__)

    def __str__(self):
        msg  = f"Instruction ID: {self.getInsId() if hasattr(self, 'insId') else -1}, "
        msg += f"thread ID: {self.getThread()}, "
        msg += f"addr: {hex(self.getRelAddr())}, "
        msg += f"{self.getOp()} "
        msg += f"src regs: {self.getSrcInt()}, "
        msg += f"dst regs: {self.getDstInt()}, "
        msg += f"attr: {self.operands.attributes if hasattr(self.operands, 'attributes') else "no attributes"}, "
        msg += f"src pipes: {self.getSrcPipes()}, "
        msg += f"dst pipes: {self.getDstPipes()}, "
        msg += f"thread id of pipes: {self.getPipesThreadId() if hasattr(self, 'pipesThreadId') else "not defined"}, "
        msg += f"exe pipe: {self.getExPipe()}"

        return msg

    def setInsId(self,i):
        self.insId = i

    def getInsId(self):
        return self.insId

    def setCoreId(self,i):
        self.coreId = i

    def getCoreId(self):
        return(self.coreId)

    def setThread(self,i):
        self.threadId = i

    def getThread(self):
        return(self.threadId)

    def getContext(self):
        context = -1
        # Handle op level controls for context
        if self.mnemonic  == "SETDVALID":
            context = self.getThread() # TODO: This is a temporary fix, need to check if this is correct

        elif self.mnemonic  == "CLEARDVALID":
            if   (self.getAttr()['dest_pulse_last'] & CLEARDVALID_dest_pulse_last_MASKS.UNPACKER): context = THREADMAP.UNPACKER_THREAD.value
            elif (self.getAttr()['dest_pulse_last'] & CLEARDVALID_dest_pulse_last_MASKS.MATH):     context = THREADMAP.MATH_THREAD.value
            elif (self.getAttr()['dest_pulse_last'] & CLEARDVALID_dest_pulse_last_MASKS.SFPU):     context = THREADMAP.SFPU_THREAD.value
            elif (self.getAttr()['dest_pulse_last'] & CLEARDVALID_dest_pulse_last_MASKS.PACKER):   context = THREADMAP.PACKER_THREAD.value
            else:                                                                            context = self.getThread()

        elif self.exPipe in ["UNPACKER0", "UNPACKER1", "UNPACKER2"]: context = THREADMAP.UNPACKER_THREAD.value
        elif self.exPipe in ["MATH", "INSTRISSUE"]:                  context = THREADMAP.MATH_THREAD.value
        elif self.exPipe in ["SFPU"]:                                context = THREADMAP.SFPU_THREAD.value
        elif self.exPipe in ["PACKER0", "PACKER1"]:                  context = THREADMAP.PACKER_THREAD.value
        else:                                                        context = self.getThread()

        assert context >= 0, f"Invalid context {context} for instruction {self.getOp()} with mnemonic {self.mnemonic} and thread {self.getThread()}"
        assert context < THREADMAP.NUM_CONTEXTS, f"Context {context} is out of bounds for instruction {self.getOp()} with mnemonic {self.mnemonic} and thread {self.getThread()}"

        return context

    def setExPipe(self,i):
        if i is not None:
            self.exPipe = i

    def getExPipe(self):
        return(self.exPipe)

    def getRelAddr(self):
        return self.addr

    def setRelAddr(self, v):
        self.addr       = v

    def setOp(self, opCodeStr):
        self.mnemonic = opCodeStr

    def getOp(self):
        return self.mnemonic

    def getAttr(self):
        if "attributes" in dir(self.getOperands()):
            return self.operands.attributes
        return {}

    def getOperands(self):
        return self.operands

    def getImm(self):
        if "immediates" in dir(self.getOperands()):
            return self.operands.immediates
        return []

    def getSrcInt(self):
        if "sources" in dir(self.getOperands()):
            return self.operands.sources.integers
        return []

    def getDstInt(self):
        if "destinations" in dir(self.getOperands()):
            return self.operands.destinations.integers
        return []

    def setSrcInt(self, srcList):
        if(srcList != []):
            self.operands.set_integer_sources(copy.deepcopy(srcList))

    def setDstInt(self,dstList):
        if(dstList != []):
            self.operands.set_integer_destinations(copy.deepcopy(dstList))

    def setImm(self,immList):
        if(immList != []):
            self.operands.set_immediates(copy.deepcopy(immList))

    def setAttr(self,attribList):
        if(attribList != {}):
            self.operands.set_attributes(copy.deepcopy(attribList))

    def isTT(self):
        return ( (self.kind == decoded_instruction.instruction_kind.ttwh) or (self.kind == decoded_instruction.instruction_kind.ttbh) or (self.kind == decoded_instruction.instruction_kind.ttqs))

    def isMop(self):
        return self.mnemonic == "MOP"

    def isReplay(self):
        return self.mnemonic == "REPLAY"

    def setInstr(self, relAddr, sizeBytes, ins_class, mnemonic, srcList, dstList, immList, attribList):
        self.setRelAddr(relAddr)
        # self.sizeBytes      = sizeBytes
        self.mnemonic       = mnemonic
        self.setSrcInt(copy.deepcopy(srcList))
        self.setDstInt(copy.deepcopy(dstList))
        self.setImm(copy.deepcopy(immList))
        self.setAttr(copy.deepcopy(attribList))
        self.setKind(ins_class)
        # self.printInstr(0)

    def setKind(self, kind):
        self.set_kind(kind)

    def resetOperands(self):
        self.setDstInt([])
        self.setSrcInt([])

    def printInstr(self, threadId, repeat=0):
        # print("Call to printInstr")
        # print("Addr=", hex(self.getRelAddr()), "[", repeat, "] Thread[", threadId, "] ", self.mnemonic, "\t", sep='', end='')
        print(f"Addr:{hex(self.getRelAddr())} Ins (hex): {hex(self.word)}, Ins:{self.mnemonic}    ", sep='', end='')

        if (len(self.getDstInt()) != 0):
            x = [i for i in self.getDstInt()]
            print("rd", end = '')
            print(x, end = ' ')
        else:
            print("rd[]", end=' ')

        if (len(self.getSrcInt()) != 0):
            print("rs", end = '')
            x = [i for i in self.getSrcInt()]
            print(x, end = ' ')
        else:
            print("rs[]", end=' ')

        if (len(self.getImm()) != 0):
            print("ri", end = '')
            x = [hex(i) for i in self.getImm()]
            print(x, end = ' ')
        else:
            print("ri[]", end=' ')

        if (len(self.getAttr()) != 0):
            print("attr", end = '')
            x = {key: hex(val) if isinstance(val, int) else val for key,val in self.getAttr().items()}
            print(x, end = ' ')
        else:
            print("attr[]", end=' ')

        print(end='\n')

    ## Synchronization
    def hasVldUpdMask(self, r):
        # print("VldUpdMask=", self.vldUpdMask)
        return r in list(self.vldUpdMask.keys())

    def getVldUpdMask(self, r):
        # print("VldUpdMask=", self.vldUpdMask)
        return self.vldUpdMask[r]

    def hasBankUpdMask(self, r):
        # print("BankUpdMask=", self.vldUpdMask)
        return r in list(self.bankUpdMask.keys())

    def getBankUpdMask(self, r):
        # print("BankUpdMask = ", self.bankUpdMask)
        return self.bankUpdMask[r]

    def hasCondChkVldUpd(self, r, t):
        if(len(self.condChkVldUpdVal) == 0):    return False
        elif(self.condChkVldUpdVal[r][t] != -2):return True
        else:                                   return False

    def getCondChkVldUpd(self, r, t):
        return self.condChkVldUpdVal[r][t]

    def hasCondWriVldUpd(self, r, t):
        if(len(self.condWriVldUpdVal) == 0):    return False
        elif(self.condWriVldUpdVal[r][t] != -2):return True
        else:                                   return False

    def getCondWriVldUpd(self, r, t):
        return self.condWriVldUpdVal[r][t]

    def getPipeBankCtrl(self, r):
        return self.pipeBankCtrl[r]

    def getSrcPipes(self):
        return self.srcPipes

    def getDstPipes(self):
        return self.dstPipes

    def setVldUpdMask(self, m):
        for k, v in m.items():
            # print("VldUpdMask: [", str(k), "]=", str(v))
            self.vldUpdMask[k]     = v

    def setBankUpdMask(self, m):
        for k, v in m.items():
            # print("BankUpdMask: [", str(k), "]=", str(v))
            self.bankUpdMask[k]     = v

    def setCondChkVldUpd(self, v):
        self.condChkVldUpdVal = v

    def setCondWriVldUpd(self, v):
        self.condWriVldUpdVal = v

    def setPipeBankCtrl(self,m):
        for k, v in m.items():
            self.pipeBankCtrl[k]     = v

    def setSrcPipes(self,pipeList):
        if pipeList:
            self.srcPipes = []
            for i in range(len(pipeList)):
                self.srcPipes.append(pipeList[i])

    def setDstPipes(self,pipeList):
        if pipeList:
            self.dstPipes = []
            for i in range(len(pipeList)):
                self.dstPipes.append(pipeList[i])

    def setPipesThreadId(self, thread_id):
        if thread_id is not None:
            self.pipesThreadId = thread_id

    def getPipesThreadId(self):
        if hasattr(self, 'pipesThreadId'):
            return self.pipesThreadId
        else:
            print(f"- WARNING: pipeThreadId not defined for instruction {self}, returning getThread()")
            return self.getThread()

    def setPipeDelay(self, delay):
        assert delay >=0 , f"Invalid pipeDelay={delay} being set for ins={self.mnemonic}"
        self.pipeDelay = delay

    def getPipeDelay(self):
        assert self.pipeDelay >=0 , f"Invalid pipeDelay={self.pipeDelay} set for ins={self.mnemonic}"
        return self.pipeDelay

    def setState(self, state):
        self.setBankUpdMask(state.bankUpdMask)
        self.setCondChkVldUpd(state.condChkVldUpdVal)
        self.setCondWriVldUpd(state.condWriVldUpdVal)
        self.setDstInt(state.getDstInt())
        self.setDstPipes(state.getDstPipes())
        self.setExPipe(state.getExPipe())
        self.setImm(state.getImm())
        self.setSrcInt(state.getSrcInt())
        self.setSrcPipes(state.getSrcPipes())
        self.setVldUpdMask(state.vldUpdMask)

        if hasattr(state, 'pipesThreadId'):
            self.setPipesThreadId(state.getPipesThreadId())

    def getSrcSize(self):
        bytesPerDatum = getNumBytesFromDataFormat(self.srcFormat)
        return bytesPerDatum * self.numDatums

    def getSrcFormat(self):
        return self.srcFormat

    def setSrcFormat(self, format):
        self.srcFormat = format

    def getDstSize(self):
        bytesPerDatum = getNumBytesFromDataFormat(self.dstFormat)
        return bytesPerDatum * self.numDatums

    def getDstFormat(self):
        return self.dstFormat

    def setDstFormat(self, format):
        self.dstFormat = format

    def getNumDatums(self):
        return self.numDatums

    def setNumDatums(self, numDatums):
        self.numDatums = numDatums

    def incrMemInfo(self, key, value):
        if not hasattr(self, 'memInfo'):        assert False, "memInfo not defined"
        if key not in self.memInfo:             self.memInfo[key] = 0; print(f"WARNING: {key} not in memInfo, initializing to 0")
        self.memInfo[key] += value

    def setMemInfo(self, key, value):
        if not hasattr(self, 'memInfo'):        assert False, "memInfo not defined"
        self.memInfo[key] = value

    def getMemInfo(self, key):
        if hasattr(self, 'memInfo') and key in self.memInfo:
            return self.memInfo[key]
        else:       assert False, f"key {key} not in memInfo"

    def getAllMemInfo(self):
        return self.memInfo

    def setEventInfo(self, key, value):
        if not hasattr(self, 'eventInfo'):
            raise AttributeError("eventInfo not defined")
        self.eventInfo[key] = value

    def getEventInfo(self, key):
        if hasattr(self, 'eventInfo') and key in self.eventInfo:
            return self.eventInfo[key]
        else:
            raise KeyError(f"key {key} not in eventInfo")

# decodeInstr - return instr object
def decodeInstr(insBinary, insKind, swz, ttISA = None):
    # print(hex(insBinary))
    ins     = instr()
    iAddr   = -1 #TODO: Need to fix
    iBytes  = 4
    # iClass   = insClass.INS_CLASS_T

    if(insKind == decoded_instruction.instruction_kind.rv32):
        ins = instructions.decode_instruction(instruction = insBinary, kind = insKind, instruction_set = ttISA)
    elif(swz):
        ins = instructions.decode_instruction(instruction = tensix.swizzle_instruction(insBinary), kind = insKind, instruction_set = ttISA)
    else:
        ins = instructions.decode_instruction(instruction = insBinary, kind = insKind, instruction_set = ttISA)

    return ins

# decodeElf - return list of instr objects
def decodeElf(elfPath, elfName, ttISA: dict[decoded_instruction.instruction_kind, dict[str, typing.Any]]):
    elfFileName = os.path.join(elfPath, elfName)
    print(elfFileName)
    insList = read_elf.decode_all_functions(elfFileName, sets = ttISA)
    return insList

def decodeFn(elfPath, elfName, kernelName, ttISA: dict[decoded_instruction.instruction_kind, dict[str, typing.Any]]):
    elfFileName = os.path.join(elfPath, elfName)
    print(elfPath, elfName, kernelName)
    print(elfFileName)
    insList = read_elf.decode_function(kernelName, elfFileName, sets = ttISA)
    return insList


def get_all_function_ranges(elfPath, elfName):
    elfFileName = os.path.join(elfPath, elfName)
    print(elfFileName)
    fnList = read_elf.get_all_function_ranges(elfFileName )
    return fnList

# decodeInstr - return list of instr objects
def decodeLLK(path, elfTxtName, kernel, arch, startAddr, stopAddr):
    insDict     = {}
    # print(f"decodeLLK Kernel: {kernel}, Thread ID:  {0} , Kernel: {kernel} , StartAddress =  {startAddr} ,EndAddress= {stopAddr}")
    start   = int(startAddr/4)
    stop    = int(stopAddr/4)
    insDict = synTest(path+elfTxtName, kernel, arch, start, stop)

    # print(insDict)
    return insDict

def synTest(elfFile,kernel, arch, start, stop):
    insList = []
    ins     = instr()

    iAddr   = 1000
    iBytes  = 4
    with open(elfFile) as f:
        lines = f.read().splitlines()

    # print(f"synTest Kernel: {kernel}, Thread ID:  {0} , Kernel: {kernel} , StartAddress =  {start} ,EndAddress= {stop}")
    for insCnt in range(len(lines)):
        if(insCnt < start or insCnt >= stop):
            continue
        ins     = lines[insCnt].split('#')[0]
        if((int(ins , 16) & 0x3) == 0x3):
            iClass =    decoded_instruction.instruction_kind.rv32
        else:
            match arch:
                case "ttwh":
                    iClass =    decoded_instruction.instruction_kind.ttwh
                case "ttbh":
                    iClass =    decoded_instruction.instruction_kind.ttbh
                case "ttqs":
                    iClass =    decoded_instruction.instruction_kind.ttqs

        insDec      = decodeInstr(int(ins, 16), iClass, swz = False)
        ins         = instr(insDec)
        insList.append(copy.deepcopy(ins))
        iAddr = iAddr + iBytes

    insDict = {kernel : insList}
    # return insList
    return insDict

def getNumBytesFromDataFormat(format):
        match format:
            case 0:  # Float32 - E8M23
                return 4
            case 4:  # Tf32 - E8M10 - Stored in L1 in 32b container
                return 2
            case 1:  # Float16 - E5M10
                return 2
            case 5:  # Float16_b - E8M7
                return 2
            case 10: # Fp8R - E5M2
                return 1
            case 16: # Fp8P - E4M3
                return 1
            case 18: # MxFp8R - E5M2 with block exp
                return 1
            case 20: # MxFp8P - E4M3 with block exp
                return 1
            case 19: # MxFp6R - E3M2 with block exp
                return 1
            case 21: # MxFp6P - E2M3 with block exp
                return 1
            case 22: # MxFp4 - E2M1 with block exp
                return 1
            case 2:  # MxInt8 - E0M7 with block exp
                return 1
            case 3:  # MxInt4 - E0M3 with block exp
                return 1
            case 11: # MxInt2 - E0M1 with block exp
                return 1
            case 8:  # Int32
                return 4
            case 14: # Int8
                return 1
            case 9:  # Int16
                return 2
            case 17: # Uint8 - Unsigned INT with 8-bit magnitude
                return 1
            case 130: # Uint16 - Unsigned INT with 16-bit magnitude
                return 2
            case 27: # MxFp4_2x_A - store MXFP4 in Src Regs as 2x-packed format with 5-bit exp
                return 1
            case 24: # MxFp4_2x_B - store MXFP4 in Src Regs as 2x-packed format with 8-bit exp
                return 1
            case 23: # Int4
                return 1
            case 25: # Uint4
                return 1
            case 26: # Int8_2x - int 2x-packed Src Reg Storage
                return 1
            case 255: # TODO: Unknown Format used in UNPACR0_STRIDE. For now default to Float16
                print("WARNING: Unknown format 255 used in UNPACR0_STRIDE. Defaulting to Float16.")
                return 2
            case _:  # Default case
                assert False, f"Unhandled format: {format}"
        return None

def getT3SimPipesFromStallRes(stallRes, pipeGrps, pipes):
    if not isinstance(stallRes, list):
        raise Exception(f"- error: expected stall_res to a list, received object of type {type(stallRes)}. Please check the decoding by binutils.")

    pipeList = []
    for res in stallRes:
        engineGrp = "TDMA" if res == "compute/tdma" else res.upper()
        pipeList.extend(pipeGrps[engineGrp])

    stallResPipes = sorted(list(set([pipes.index(p) for p in pipeList])))

    # stallResPipes = []
    # for p in pipeList:
    #     if(pipes.index(p) not in stallResPipes):
    #         stallResPipes.append(pipes.index(p))

    return stallResPipes

    #setc16 reg, value
    #b2070808  - setc16 c7, 0x808
    # b608:   [TT]c81c2022 b2070808   setc16  fa7,fa6,fa6
    # b60c:   [TT]c85c0022 b2170008   setc16  fa7,ft0,fa6
    # b610:   [TT]c8c00002 b2300000   setc16  ft1,ft0,ft0
    # b614:   [TT]c8240002 b2090000   setc16  ft1,ft0,fa6
    # b618:   [TT]c8600002 b2180000   setc16  ft1,ft0,ft0
    # b61c:   [TT]c8c40002 b2310000   setc16  ft1,ft0,fa6
    # b620:   [TT]c82e0202 b20b8080   setc16  ft1,ft1,fs8
    # b624:   [TT]c8649002 b2192400   setc16  ft1,fs0,fs2
    # b628:   [TT]c8c80002            setc16  ft1,ft0,ft0
    # b62c:   [TT]c8360202            setc16  ft1,ft1,fs8
    # b630:   [TT]c86a4022            setc16  fa7,ft0,fs1
    # b634:   [TT]c8cc0002            setc16  ft1,ft0,fa6
