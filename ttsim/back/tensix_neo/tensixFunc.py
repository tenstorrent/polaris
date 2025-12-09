#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Functional Model
import sys

import math
import copy
import enum

import ttsim.back.tensix_neo.isaFunctions as isaFunctions
import ttsim.front.llk.decoded_instruction as decoded_instruction
from ttsim.back.tensix_neo.isaFunctions import valueStatus

MEMORY_MAP_KEY_TRISC_MAP = 'trisc_map'
MEMORY_MAP_KEY_TRISC_MAP_CFG = 'cfg_regs'
MEMORY_MAP_KEY_TRISC_MAP_OFFSETS = 'OFFSETS'
MEMORY_MAP_KEY_END = 'END'
MEMORY_MAP_KEY_NUM_BYTES_PER_REG = 'NUM_BYTES_PER_REGISTER'
MEMORY_MAP_KEY_START = 'START'
MEMORY_MAP_KEY_TRISC_MAP_INSTR_BUFFER = 'ibuffer'
MEMORY_MAP_KEY_TRISC_MAP_MOP = 'mop_cfg'

# PCBUFFER
MEMORY_MAP_KEY_TRISC_MAP_PCBUFFER = 'pcbuffer'
MEMORY_MAP_VALUE_TRISC_MAP_SEMAPHORES   = 0x80
MEMORY_MAP_VALUE_TRISC_MAP_MOPSYNC      = 0x08
MEMORY_MAP_VALUE_TRISC_MAP_IDLESYNC     = 0x04


BANK_UPDATE_THRESHOLD = 512

class tt_semaphore_idx(enum.IntEnum):
    # start enum from 0. https://stackoverflow.com/a/61438054/27310047
    def _generate_next_value_(name, start, count, last_values):
        """generate consecutive automatic numbers starting from zero"""
        return count

    id             = enum.auto(),
    bank           = enum.auto(),
    init_value     = enum.auto(),
    max_value      = enum.auto(),
    current_value  = enum.auto(),
    wait_sem_cond  = enum.auto(),
    pipes_to_stall = enum.auto(),
    thread_id_of_pipes = enum.auto() # both srcPipes and dstPipes are assumed to belong to same thread id.

MAX_THREADS                   = 4
NUM_VARIABLES_PER_TTSEMAPHORE = len(tt_semaphore_idx)
NUM_TTSEMAPHORES_PER_BANK     = 8

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

class ttSplRegs:
    TREG_NUMREGS    = 0
    TREG_STARTADDR  = 1
    TREG_ADDRSIZE   = 2
    TREG_NUMBANKS   = 3

    def __init__(self,coreId, args):
        self.debug      = args['debug']
        self.args       = args
        self.coreId     = coreId
        ### REGISTERS
        self.regTypeDict = {
            'cfg' : [
                (args[MEMORY_MAP_KEY_TRISC_MAP][MEMORY_MAP_KEY_TRISC_MAP_CFG][MEMORY_MAP_KEY_END] - \
                 args[MEMORY_MAP_KEY_TRISC_MAP][MEMORY_MAP_KEY_TRISC_MAP_CFG][MEMORY_MAP_KEY_START])/ \
                 args[MEMORY_MAP_KEY_TRISC_MAP][MEMORY_MAP_KEY_TRISC_MAP_CFG][MEMORY_MAP_KEY_NUM_BYTES_PER_REG], #NUM_REGISTERS
                args[MEMORY_MAP_KEY_TRISC_MAP][MEMORY_MAP_KEY_TRISC_MAP_CFG][MEMORY_MAP_KEY_START], #MMR_START
                (args[MEMORY_MAP_KEY_TRISC_MAP][MEMORY_MAP_KEY_TRISC_MAP_CFG][MEMORY_MAP_KEY_END] - \
                 args[MEMORY_MAP_KEY_TRISC_MAP][MEMORY_MAP_KEY_TRISC_MAP_CFG][MEMORY_MAP_KEY_START]), #MMR_SIZE
                1,                  #NUM_BANKS
            ],
            'instrBuffer' : [
                1,                 #NUM_REGISTERS,
                args[MEMORY_MAP_KEY_TRISC_MAP][MEMORY_MAP_KEY_TRISC_MAP_INSTR_BUFFER][MEMORY_MAP_KEY_START],    #MMR_START
                1,                 #MMR_SIZE
                1,                  #NUM_BANKS
            ],
            'mop' : [
                64*MAX_THREADS,     #NUM_REGISTERS *,
                args[MEMORY_MAP_KEY_TRISC_MAP][MEMORY_MAP_KEY_TRISC_MAP_MOP][MEMORY_MAP_KEY_START], #MMR_START
                64,                 #MMR_SIZE
                1,                  #NUM_BANKS
            ],
            'mopSync' : [
                MAX_THREADS,       #NUM_REGISTERS ,
                args[MEMORY_MAP_KEY_TRISC_MAP][MEMORY_MAP_KEY_TRISC_MAP_PCBUFFER][MEMORY_MAP_KEY_START] + MEMORY_MAP_VALUE_TRISC_MAP_MOPSYNC,    #MMR_START = PCBUFFER + 0x08. TODO: Get this from cfg itself
                4,                 #MMR_SIZE
                1,                  #NUM_BANKS
            ],
            'idleSync' : [
                MAX_THREADS,       #NUM_REGISTERS ,
                args[MEMORY_MAP_KEY_TRISC_MAP][MEMORY_MAP_KEY_TRISC_MAP_PCBUFFER][MEMORY_MAP_KEY_START] + MEMORY_MAP_VALUE_TRISC_MAP_IDLESYNC,    #MMR_START = PCBUFFER + 0x04. TODO: Get this from cfg itself
                4,                 #MMR_SIZE
                1,                  #NUM_BANKS
            ],
            'ttsemaphores' : [
                NUM_VARIABLES_PER_TTSEMAPHORE * NUM_TTSEMAPHORES_PER_BANK,
                args[MEMORY_MAP_KEY_TRISC_MAP][MEMORY_MAP_KEY_TRISC_MAP_PCBUFFER][MEMORY_MAP_KEY_START] + MEMORY_MAP_VALUE_TRISC_MAP_SEMAPHORES,    #MMR_START = PCBUFFER + 0x80. TODO: Get this from cfg itself
                64,                 #MMR_SIZE
                32,                 #NUM_BANKS. There are most likely only 2 banks, but bank ID field width is 5 bits, so we set NUM_BANKS as 32.
            ]
        }
        self.regTypes = [ key for key, val in self.regTypeDict.items() ]
        self.regSizes = [ int(self.regTypeDict[key][0])*self.regTypeDict[key][3] for key,val in self.regTypeDict.items() ]

        self.reg = {}

        for i in range(len(self.regTypes)):
            regList = []
            for j in range(self.regSizes[i]):
                regList.append(-1)
            self.reg[self.regTypes[i]] = copy.deepcopy(regList)

    # Tensix Pipes Register File
    def __writeReg__(self,r, val, type='riscgpr'):
        assert type in self.regTypes, "RegType not supported:" + type
        assert val != None , "Only legal values supported" + str(val)
        if(self.debug & 0x10):      print(f"{type}[{hex(r)}] = {hex(val)}")
        self.reg[type][r] = val

    def __readReg__(self, r, type ='riscgpr'):
        assert type in self.regTypes, "RegType not supported:" + type + "Supported RegTypes" +  self.regTypes
        if (type == 'riscgpr' and (r in self.regTempsriscgpr)): # Temporary Registers - RISCV
            assert self.reg[type][r] != -1, "Illegal Read " + type + "[" + str(r) + "]"
            assert self.reg[type][r] != None, "Illegal Read " + type + "[" + str(r) + "]"
        elif(type == 'riscgpr' and self.reg[type][r] == -1):     # For non temporary variables, reset initial value to zero
            self.reg[type][r] = 0

        return self.reg[type][r]

    def __ismmr__(self,addr ):
        regTypeSel = ''
        offset = -1
        # 1. Check MMR or not
        for key, val in self.regTypeDict.items():
            if (addr in range(self.regTypeDict[key][1], (self.regTypeDict[key][1] + self.regTypeDict[key][2]))):
                if(self.debug & 0x10):       print(f"Is MMR, addr={hex(addr)},type={key},AddrStart={hex(self.regTypeDict[key][1])},AddrEnd={hex(self.regTypeDict[key][1] + self.regTypeDict[key][2])}")
                regTypeSel = key
                break
            if(self.debug & 0x10):       print("Address:", hex(addr), "not in", key, "Address Range:", hex(self.regTypeDict[key][1]), hex(self.regTypeDict[key][1] + self.regTypeDict[key][2]))
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

    def __printReg__(self, type='mop'):
        assert type in self.regTypes, "RegType not supported:" + type + "Supported RegTypes" +  self.regTypes
        regCnt = 0
        print("Reg. :", end='')
        while (regCnt < len(self.reg[type])):
            if(self.reg[type][regCnt] != -1):
                print (type, "[", regCnt ,"]=", hex(self.reg[type][regCnt]),",", sep='', end='' )
            regCnt += 1
        print()

    def __getParamsNumRegs__(self, regType: str):
        assert regType in self.regTypes, f"RegType not supported:{regType} Supported RegTypes" +  self.regTypes
        return self.regTypeDict[regType][self.TREG_NUMREGS]

    def __getParamsStartAddr__(self, regType: str):
        assert regType in self.regTypes, f"RegType not supported:{regType} Supported RegTypes" +  self.regTypes
        return self.regTypeDict[regType][self.TREG_STARTADDR]

    def __getParamsAddrSize__(self, regType: str):
        assert regType in self.regTypes, f"RegType not supported:{regType} Supported RegTypes" +  self.regTypes
        return self.regTypeDict[regType][self.TREG_ADDRSIZE]

    def __getParamsNumBanks__(self, regType: str):
        assert regType in self.regTypes, f"RegType not supported:{regType} Supported RegTypes" +  self.regTypes
        return self.regTypeDict[regType][self.TREG_NUMBANKS]

    ### Special Functions to handle regType = 'cfg'
    def getCfgRegOffsets(self):
        return self.args[MEMORY_MAP_KEY_TRISC_MAP][MEMORY_MAP_KEY_TRISC_MAP_CFG][MEMORY_MAP_KEY_TRISC_MAP_OFFSETS]

    def getCfgRegOffset(self, name: str):
        cfg_offsets = self.getCfgRegOffsets()
        for offset, regs in cfg_offsets.items():
            for reg_name in regs.keys():
                if reg_name == name:
                    return offset

        raise Exception(f"Could not find cfg register with name: {name}")

    def getCfgRegsInfowithOffset(self, offset: int):
        cfg_offsets = self.getCfgRegOffsets()
        assert offset in cfg_offsets, "Invalid offset"
        regs = cfg_offsets[offset]
        for reg_info in regs.values():
            reg_info['OFFSET'] = offset
        return regs

    def getCfgRegInfo(self, name: str):
        offset = self.getCfgRegOffset(name)
        info = self.getCfgRegOffsets()[offset][name]
        info['OFFSET'] = offset
        return info

    def readCfgRegwithOffsetandName(self, offset: int, name: str):
        assert offset in self.getCfgRegOffsets(), "Invalid offset"
        regs = self.getCfgRegOffsets()[offset]
        if name not in regs:
            raise Exception(f"Register {name} not found at offset {offset}")
        reg_info = regs[name]
        value = self.__readReg__(offset, type='cfg')
        if -1 == value:
            if self.debug & 0x08:
                print(f"WARNING: Register {name} at offset {offset} is uninitialized.")
                value = 0 # For debug purposes, forcing this to zero. Note that this is dangerous and should be used for debug purposes ONLY
            else:
                raise Exception(f"Register {name} at offset {offset} is uninitialized.")

        if ('SHAMT' in reg_info) or ('MASK' in reg_info):
            assert 'SHAMT' in reg_info, "SHAMT must be present in register info"
            assert 'MASK' in reg_info, "MASK must be present in register info"
            lsb = reg_info['SHAMT']
            mask = reg_info['MASK']
            return (value & mask) >> lsb # (value >> lsb) & (mask >> lsb)
        else:
            return value

    def readCfgReg(self, name):
        reg_info = self.getCfgRegInfo(name)
        return self.readCfgRegwithOffsetandName(reg_info['OFFSET'], name)

    def readCfgRegswithOffset(self, offset):
        assert offset in self.getCfgRegOffsets(), "Invalid offset"
        regs = self.getCfgRegOffsets()[offset]
        regs_values = dict()
        for reg_name in regs.keys():
            regs_values[reg_name] = self.readCfgRegwithOffsetandName(offset, reg_name)

        return regs_values

    def isDstRegProgrammed(self):
        # In Quasar synchronisation between different clients/pipes
        # /resources (e.g. unpacker, math, sfpu, pack) can be done
        # with either semaphores or with dvalid (data valid) scheme.
        # The dvalid scheme for Dest (also referred to as Dst in present
        # model) register is programmed with the help of the following
        # config registers. If they are set to 0, the Dest register
        # read/writes are not programmed. The performance model sets
        # the initial value of all registers to -1.
        # So we check for two values [0, -1].

        reg_values: list[int] = [
            self.__readReg__(self.getCfgRegOffset('UNPACK_TO_DEST_DVALID_CTRL_disable_auto_bank_id_toggle'), 'cfg'),
            self.__readReg__(self.getCfgRegOffset('MATH_DEST_DVALID_CTRL_disable_auto_bank_id_toggle'), 'cfg'),
            self.__readReg__(self.getCfgRegOffset('SFPU_DEST_DVALID_CTRL_disable_auto_bank_id_toggle'), 'cfg'),
            self.__readReg__(self.getCfgRegOffset('PACK_DEST_DVALID_CTRL_disable_auto_bank_id_toggle'), 'cfg')]

        return any(((v != -1) and (v != 0)) for v in reg_values)

    def getCfgRegUpdateClass(self, offset: int):
        regs_with_offsets = self.getCfgRegsInfowithOffset(offset)
        if any(reg_name in regs_with_offsets.keys() for reg_name in [
            'DEST_TARGET_REG_CFG_MATH_SEC0_Offset',
            'DEST_TARGET_REG_CFG_MATH_SEC1_Offset',
            'DEST_TARGET_REG_CFG_MATH_SEC2_Offset',
            'DEST_TARGET_REG_CFG_MATH_SEC3_Offset']):
            return "DEST_TARGET_REG_CFG_MATH"

        elif any(reg_name.endswith("_DEST_DVALID_CTRL_disable_auto_bank_id_toggle") for reg_name in regs_with_offsets.keys()):
            return "DEST_DVALID_CTRL"

        else:
            return "UNKNOWN"

    def updateDstRegBankId(self, offset: int):
        assert "DEST_TARGET_REG_CFG_MATH" == self.getCfgRegUpdateClass(offset)

        if self.__readReg__(offset,"cfg") >= BANK_UPDATE_THRESHOLD:
            return True
        return False

    def getDstRegCondValids(self, offset: int):
        assert "DEST_DVALID_CTRL" == self.getCfgRegUpdateClass(offset)

        valids = {"ContextID" : None, "WRITE" : None, "READ" : None}
        regs_with_offsets = self.getCfgRegsInfowithOffset(offset)
        suffix = "_DEST_DVALID_CTRL_disable_auto_bank_id_toggle"
        prefix = ""
        for reg_name in regs_with_offsets.keys():
            if reg_name.endswith(suffix):
                prefix = reg_name.split(suffix)[0]
                break # we assume there's only 1 register that matches this suffix.
        if not prefix:
            raise Exception(f"- error: could not find register name which ends with {suffix}. registers associated with offset {addr} are {regs_with_offsets.keys()}")

        cfg_offset_dvalid_mask = offset - self.getCfgRegOffset('UNPACK_TO_DEST_DVALID_CTRL_disable_auto_bank_id_toggle')
        wait_mask     = self.readCfgRegwithOffsetandName(offset = offset, name = f"{prefix}_DEST_DVALID_CTRL_wait_mask")
        wait_polarity = self.readCfgRegwithOffsetandName(offset = offset, name = f"{prefix}_DEST_DVALID_CTRL_wait_polarity")
        toggle_mask   = self.readCfgRegwithOffsetandName(offset = offset, name = f"{prefix}_DEST_DVALID_CTRL_toggle_mask")

        if self.debug & 0x8:
            print(f"DEST_DVALID_CTRL toggle_mask: {toggle_mask}, wait_polarity: {wait_polarity}, wait_mask: {wait_mask}")

        valids["ContextID"] = cfg_offset_dvalid_mask
        valids["READ"]      = wait_polarity & wait_mask
        valids["WRITE"]     = toggle(wait_polarity, toggle_mask)

        return valids

    def writeCfgReg(self, addr, value, ins):
        assert addr in self.getCfgRegOffsets(), "Invalid offset"
        self.__writeReg__(addr, value, "cfg")

        src        = []
        dst        = []
        imm        = []
        vldUpd     = {}
        bankUpd    = {}
        condChkVld = {}
        condWriVld = {}

        #4 registers / 4 threads
        max_num_threads = self.args['maxNumThreadsperNeoCore']
        condChkVld[isaFunctions.regIndex.srcA] = [valueStatus.IGNORE for _ in range(max_num_threads)] #-2 implies don't care
        condChkVld[isaFunctions.regIndex.srcB] = [valueStatus.IGNORE for _ in range(max_num_threads)] #-2 implies don't care
        condChkVld[isaFunctions.regIndex.srcS] = [valueStatus.IGNORE for _ in range(max_num_threads)] #-2 implies don't care
        condChkVld[isaFunctions.regIndex.dst]  = [valueStatus.IGNORE for _ in range(max_num_threads)] #-2 implies don't care

        condWriVld[isaFunctions.regIndex.srcA] = [valueStatus.IGNORE for _ in range(max_num_threads)] #-2 implies don't care
        condWriVld[isaFunctions.regIndex.srcB] = [valueStatus.IGNORE for _ in range(max_num_threads)] #-2 implies don't care
        condWriVld[isaFunctions.regIndex.srcS] = [valueStatus.IGNORE for _ in range(max_num_threads)] #-2 implies don't care
        condWriVld[isaFunctions.regIndex.dst]  = [valueStatus.IGNORE for _ in range(max_num_threads)] #-2 implies don't care

        update_class = self.getCfgRegUpdateClass(addr)

        print(f"Writing to (TENSIX) instruction addr: {hex(ins.getRelAddr())} {ins.getOp()}, "
              f"thread[{ins.getThread()}], update class: {update_class}, cfg[{addr}] = {value}")
        self.__writeReg__(addr, value, "cfg")

        match update_class:
            case "DEST_TARGET_REG_CFG_MATH":
                if self.updateDstRegBankId(addr):
                    bankUpd[isaFunctions.regIndex.dst] = 1 # Update Bank 3
                    print(f"DEST_TARGET_REG_CFG_MATH BankUpdate[{isaFunctions.regIndex.dst}] = {bankUpd[isaFunctions.regIndex.dst]} "
                          f"for instruction: {ins}")

            case "DEST_DVALID_CTRL":
                condValids = self.getDstRegCondValids(addr)
                context_id = condValids["ContextID"]
                condChkVld[isaFunctions.regIndex.dst][context_id] = condValids["READ"]
                condWriVld[isaFunctions.regIndex.dst][context_id] = condValids["WRITE"]

                print(f"DEST_DVALID_CTRL Masks: "
                    f"condChkVld[{isaFunctions.regIndex.dst}][{context_id}] = {condChkVld[isaFunctions.regIndex.dst][context_id]} "
                    f"condWriVld[{isaFunctions.regIndex.dst}][{context_id}] = {condWriVld[isaFunctions.regIndex.dst][context_id]} "
                    f"for instruction: {ins}")

            case _:
                print(f"WARNING: Unhandled special register (TENSIX) cfg[{addr}]. Current values: {self.readCfgRegswithOffset(addr)}. Instruction: {ins}")

        status = isaFunctions.instr()
        status.setSrcInt(src)
        status.setDstInt(dst)
        status.setImm(imm)
        status.setVldUpdMask(vldUpd)
        status.setBankUpdMask(bankUpd) # Setting Bank Value instead of bank Update
        status.setCondChkVldUpd(condChkVld)
        status.setCondWriVldUpd(condWriVld)

        return status

    def getCfgRegLeastSignificantBitIndex(self, name: str) -> int:
        return self.getCfgRegInfo(name)["SHAMT"]

    def getCfgRegMask(self, name: str) -> int:
        mask = self.getCfgRegInfo(name)["MASK"]
        if isinstance(mask, str):
            mask = int(mask.strip(), 0)
        return mask

    def getCfgRegMostSignificantBitIndex(self, name: str) -> int:
        mask = self.getCfgRegMask(name)
        if 0 == mask:
            raise ValueError(f"Mask for register {name} is zero, cannot determine most significant bit index.")

        lsb_index = self.getCfgRegLeastSignificantBitIndex(name)

        assert (mask & -mask).bit_length() - 1 >= lsb_index # first non-zero bit should be at least lsb

        mask >>= lsb_index
        # does not check if mask is continous.
        return lsb_index + mask.bit_length() - 1

    def getCfgRegSizeinNumBits(self, name: str) -> int:
        mask = self.getCfgRegMask(name)
        if 0 == mask:
            return mask

        lsb_index = self.getCfgRegLeastSignificantBitIndex(name)
        shifted_mask = mask >> lsb_index

        if (shifted_mask & (shifted_mask + 1)): # disjoint bitfields mask (e.g. 0b110110001)
            msg  = f"- WARNING: mask for register {name} does not represent continuous bitfield, returning bit_length of mask value.\n"
            msg += f"  register: {name}\n"
            msg += f"  mask:     {mask:#x}\n"
            print(msg.rstrip())

        return shifted_mask.bit_length()

    def getCfgRegMaxPossibleValue(self, name: str) -> int:
        return (2 ** self.getCfgRegSizeinNumBits(name)) - 1

    def getMaxNumUnpacker2Packer1AutoloopIterations(self) -> int:
        max_instrn_count = max(self.getCfgRegMaxPossibleValue(name) for name in ["THCON_UNPACKER2_REG0_INSTRN_COUNT", "THCON_PACKER1_REG0_INSTRN_COUNT"])
        max_instrn_loop_count = max(self.getCfgRegMaxPossibleValue(name) for name in ["THCON_UNPACKER2_REG0_INSTRN_LOOP_COUNT", "THCON_PACKER1_REG0_INSTRN_LOOP_COUNT"])

        # +1 because autoloop executes (instrn_count + 1) instructions for (loop_count + 1) iterations
        return (max_instrn_count + 1) * (max_instrn_loop_count + 1)

class tensixFunc:
    def __init__(self, coreId, mem, args, pipeGrps, pipes, tensixSplRegs, triscRegs):
        self.debug      = args['debug']
        self.args       = args
        self.pipeGrps   = pipeGrps
        self.pipes      = pipes
        self.coreId     = coreId
        #Memory Space.
        self.memData    = mem
        self.tensixSplRegs = tensixSplRegs
        self.triscRegs  = triscRegs
        print(args)

    def _compute_num_datums_from_row_cnt_enc(self, ins):
        """Helper method to compute ``numDatums`` based on the Row_Cnt_Enc attribute.

        For ``llkVersionTag.group`` values 0 and 1, this method derives the number
        of datums from ``ins.getAttr()["Row_Cnt_Enc"]`` using the encoded mapping
        (0 → 4 * 16, 1 → 2 * 16, 2 → 1 * 16, 3 → 16 // 2, 4 → 16 // 4, 5 → 16 // 8).
        For any other group (for example, group 2), the Row_Cnt_Enc value is ignored
        and a fixed default of ``4 * 16`` datums is returned.

        Args:
            ins: The instruction containing the Row_Cnt_Enc attribute.

        Returns:
            int: Number of datums computed from Row_Cnt_Enc for groups 0 and 1,
            or the default value ``4 * 16`` for all other groups.

        Raises:
            AssertionError: If ``Row_Cnt_Enc`` is outside the valid range [0, 5] or
                if an unhandled encoded value is encountered for the given
                instruction.
        """
        if (self.args['llkVersionTag'].group in [0,1]):
            row_cnt_enc = ins.getAttr()["Row_Cnt_Enc"]
            assert row_cnt_enc >= 0 and row_cnt_enc <= 5, f"Unknown Row_Cnt_Enc value={row_cnt_enc} for instruction {ins.getOp()}"
            match row_cnt_enc:
                case 0: return 4 * 16
                case 1: return 2 * 16
                case 2: return 1 * 16
                case 3: return 16//2
                case 4: return 16//4
                case 5: return 16//8
                case _:
                    assert False, f"Unhandled Row_Cnt_Enc value={row_cnt_enc} for instruction {ins.getOp()}"
        else:
            return 4 * 16

    def __execunpacrti__(self,ins):
        opList0 = ["UNPACR0_TILE_INC", "UNPACR1_TILE_INC", "UNPACR2_TILE_INC", "UNPACR_DEST_TILE_INC"]
        opList1 = ["UNPACR0_FACE_INC", "UNPACR1_FACE_INC", "UNPACR2_FACE_INC", "UNPACR_DEST_FACE_INC"]
        opList2 = ["UNPACR0_ROW_INC", "UNPACR1_ROW_INC", "UNPACR2_ROW_INC", "UNPACR_DEST_ROW_INC"]

        assert ins.getOp() in opList0 or ins.getOp() in opList1 or ins.getOp() in opList2, "Expected opcode UNPACR0/1/2/DEST_TILE/FACE/ROW_INC. Received " + str(ins.getOp())
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                if(ins.getOp() in opList0):
                    assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))
                elif(ins.getOp() in opList1):
                    assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))
                elif(ins.getOp() in opList2):
                    assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                if(ins.getOp() in opList0):
                    assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))
                elif(ins.getOp() in opList1):
                    assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))
                elif(ins.getOp() in opList2):
                    assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))

        src = [] ; dst = [] ; imm = [] ; vldUpd = {} ; bankUpd = {};

        if("UNPACR0" in ins.mnemonic):
            reg_id = isaFunctions.regIndex.srcA
            dst.append(reg_id);
            if(ins.getAttr()["SetDatValid"]):
                vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcA
            else:
                vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcA
            exPipe = "UNPACKER0"
        elif("UNPACR1" in ins.mnemonic):
            reg_id = isaFunctions.regIndex.srcB
            dst.append(reg_id);
            if(ins.getAttr()["SetDatValid"]):
                vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcB
            else:
                vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcB
            exPipe = "UNPACKER1"
        elif("UNPACR2" in ins.mnemonic):
            reg_id = isaFunctions.regIndex.srcS
            dst.append(reg_id);
            if(ins.getAttr()["SetDatValid"]):
                vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcS
            else:
                vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcS
            exPipe = "UNPACKER2"
        elif("UNPACR_DEST" in ins.mnemonic):
            reg_id = isaFunctions.regIndex.dst
            dst.append(reg_id);
            if(ins.getAttr()["SetDatValid"]):
                vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #dst
            else:
                vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #dst
            exPipe = "UNPACKER0"

        # Finding mapped formats and numDatums
        assert exPipe in self.pipes , str(exPipe) + " not found in configured engines"
        bufferFormatReg = f'BUFFER_DESCRIPTOR_TABLE_REG{ins.getAttr()["Buffer_Descriptor_Table_Sel"]}_TILE_FORMAT'
        outFormatReg = f'THCON_{exPipe}_REG0_OUT_DATA_FORMAT'
        srcFormat = self.tensixSplRegs.readCfgReg(bufferFormatReg)
        dstFormat = self.tensixSplRegs.readCfgReg(outFormatReg)
        if self.debug & 0x8:            print(f"Reading cfgReg {bufferFormatReg} and {outFormatReg}. Setting srcFormat={srcFormat}, dstFormat={dstFormat}")

        assert isaFunctions.getNumBytesFromDataFormat(srcFormat) != None, f"Unsupported format {srcFormat} in {bufferFormatReg}"
        assert isaFunctions.getNumBytesFromDataFormat(dstFormat) != None, f"Unsupported format {dstFormat} in {outFormatReg}"
        if("TILE" in ins.mnemonic):     numDatums = 32 * 32
        elif("FACE" in ins.mnemonic):   numDatums = 16 * 16
        elif("ROW" in ins.mnemonic):    numDatums = 1 * 16
        else:
            print(f"WARNING: Don't know how to calculate numDatums. Setting to 1")
            numDatums = 1
        if self.debug & 0x8:            print(f"Computed srcSize={isaFunctions.getNumBytesFromDataFormat(srcFormat)}, dstSize={isaFunctions.getNumBytesFromDataFormat(dstFormat)} numDatums={numDatums}")

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)
        ins.setExPipe(exPipe)
        ins.setNumDatums(numDatums)
        ins.setSrcFormat(srcFormat)
        ins.setDstFormat(dstFormat)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execunpacrt__(self,ins):
        opList0 = ["UNPACR0_TILE", "UNPACR1_TILE", "UNPACR2_TILE", "UNPACR_DEST_TILE"]
        opList1 = ["UNPACR0_FACE", "UNPACR1_FACE", "UNPACR2_FACE", "UNPACR_DEST_FACE"]
        opList2 = ["UNPACR0_ROW", "UNPACR1_ROW", "UNPACR2_ROW", "UNPACR_DEST_ROW"]

        assert ins.getOp() in opList0 or ins.getOp() in opList1 or ins.getOp() in opList2, "Expected opcode UNPACR0/1/2/_DEST_TILE/FACE/ROW. Received " + str(ins.getOp())
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                if(ins.getOp() in opList0):
                    assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))
                elif(ins.getOp() in opList1):
                    assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))
                elif(ins.getOp() in opList2):
                    assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                if(ins.getOp() in opList0):
                    assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))
                elif(ins.getOp() in opList1):
                    assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))
                elif(ins.getOp() in opList2):
                    assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))

        src = [] ; dst = [] ; imm = [] ; vldUpd = {} ; bankUpd = {};

        if("UNPACR0" in ins.mnemonic):
            reg_id = isaFunctions.regIndex.srcA
            dst.append(reg_id);
            if(ins.getAttr()["SetDatValid"]):
                vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcA
            else:
                vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcA
            exPipe = "UNPACKER0"
        elif("UNPACR1" in ins.mnemonic):
            reg_id = isaFunctions.regIndex.srcB
            dst.append(reg_id);
            if(ins.getAttr()["SetDatValid"]):
                vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcB
            else:
                vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcB
            exPipe = "UNPACKER1"
        elif("UNPACR2" in ins.mnemonic):
            reg_id = isaFunctions.regIndex.srcS
            dst.append(reg_id);
            if(ins.getAttr()["SetDatValid"]):
                vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcS
            else:
                vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcS
            exPipe = "UNPACKER2"
        elif("UNPACR_DEST" in ins.mnemonic):
            reg_id = isaFunctions.regIndex.dst
            dst.append(reg_id);
            if(ins.getAttr()["SetDatValid"]):
                vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #dst
            else:
                vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #dst
            exPipe = "UNPACKER0"

        # Finding mapped formats and numDatums
        assert exPipe in self.pipes , str(exPipe) + " not found in configured engines"
        bufferFormatReg = f'BUFFER_DESCRIPTOR_TABLE_REG{ins.getAttr()["Buffer_Descriptor_Table_Sel"]}_TILE_FORMAT'
        outFormatReg = f'THCON_{exPipe}_REG0_OUT_DATA_FORMAT'
        srcFormat = self.tensixSplRegs.readCfgReg(bufferFormatReg)
        dstFormat = self.tensixSplRegs.readCfgReg(outFormatReg)
        if self.debug & 0x8:            print(f"Reading cfgReg {bufferFormatReg} and {outFormatReg}. Setting srcFormat={srcFormat}, dstFormat={dstFormat}")

        assert isaFunctions.getNumBytesFromDataFormat(srcFormat) != None, f"Unsupported format {srcFormat} in {bufferFormatReg}"
        assert isaFunctions.getNumBytesFromDataFormat(dstFormat) != None, f"Unsupported format {dstFormat} in {outFormatReg}"
        if("TILE" in ins.mnemonic):     numDatums = 32 * 32
        elif("FACE" in ins.mnemonic):   numDatums = 16 * 16
        elif("ROW" in ins.mnemonic):    numDatums = 1 * 16
        else:
            print(f"WARNING: Don't know how to calculate numDatums. Setting to 1")
            numDatums = 1
        if self.debug & 0x8:            print(f"Computed srcSize={isaFunctions.getNumBytesFromDataFormat(srcFormat)}, dstSize={isaFunctions.getNumBytesFromDataFormat(dstFormat)} numDatums={numDatums}")

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)
        ins.setExPipe(exPipe)
        ins.setNumDatums(numDatums)
        ins.setSrcFormat(srcFormat)
        ins.setDstFormat(dstFormat)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execunpacrs__(self,ins):
        opList0 = ["UNPACR0_STRIDE", "UNPACR1_STRIDE", "UNPACR2_STRIDE", "UNPACR_DEST_STRIDE"]
        assert ins.getOp() in opList0 , "Expected opcode UNPACR0/1/2/_DEST_STRIDE. Received " + str(ins.getOp())
        # TODO: Reset destination/source
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                if(ins.getOp() in opList0):
                    assert len(ins.getAttr()) == 7, "Seven attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                if(ins.getOp() in opList0):
                    assert len(ins.getAttr()) == 7, "Seven attribs expected. Received " + str(len(ins.getAttr()))

        src = [] ; dst = [] ; imm = [] ; vldUpd = {} ; bankUpd = {};

        if("UNPACR0" in ins.mnemonic):
            reg_id = isaFunctions.regIndex.srcA
            dst.append(reg_id);
            if(ins.getAttr()["SetDatValid"]):
                vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcA
            else:
                vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcA
            exPipe = "UNPACKER0"
        elif("UNPACR1" in ins.mnemonic):
            reg_id = isaFunctions.regIndex.srcB
            dst.append(reg_id);
            if(ins.getAttr()["SetDatValid"]):
                vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcB
            else:
                vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcB
            exPipe = "UNPACKER1"
        elif("UNPACR2" in ins.mnemonic):
            reg_id = isaFunctions.regIndex.srcS
            dst.append(reg_id);
            if(ins.getAttr()["SetDatValid"]):
                vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcS
            else:
                vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcS
            exPipe = "UNPACKER2"
        elif("UNPACR_DEST" in ins.mnemonic):
            reg_id = isaFunctions.regIndex.dst
            dst.append(reg_id);
            if(ins.getAttr()["SetDatValid"]):
                vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #dst
            else:
                vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #dst
            exPipe = "UNPACKER0"

        # Finding mapped formats and numDatums
        assert exPipe in self.pipes , str(exPipe) + " not found in configured engines"
        bufferFormatReg = f'BUFFER_DESCRIPTOR_TABLE_REG{ins.getAttr()["Buffer_Descriptor_Table_Sel"]}_TILE_FORMAT'
        outFormatReg = f'THCON_{exPipe}_REG0_OUT_DATA_FORMAT'
        srcFormat = self.tensixSplRegs.readCfgReg(bufferFormatReg)
        dstFormat = self.tensixSplRegs.readCfgReg(outFormatReg)
        if self.debug & 0x8:            print(f"Reading cfgReg {bufferFormatReg} and {outFormatReg}. Setting srcFormat={srcFormat}, dstFormat={dstFormat}")

        assert isaFunctions.getNumBytesFromDataFormat(srcFormat) != None, f"Unsupported format {srcFormat} in {bufferFormatReg}"
        assert isaFunctions.getNumBytesFromDataFormat(dstFormat) != None, f"Unsupported format {dstFormat} in {outFormatReg}"
        numDatums = 8 * 16  # TODO: Upto 8 Rows. Check for THCON_UNPACKER<0/1/2>_REG1_UNPACK_STRIDE_NO_WRITE
                            # We can write at least 128 datums per cycle, so ignore because of no BW savings ?
                            # Also check if THCON_UNPACKER0/1/2_REG1_UNPACK_STRIDE_ROW_MASK and THCON_UNPACKER0/1/2_REG1_UNPACK_STRIDE_ROW_MASK are needed
        if self.debug & 0x8:            print(f"Computed srcSize={isaFunctions.getNumBytesFromDataFormat(srcFormat)}, dstSize={isaFunctions.getNumBytesFromDataFormat(dstFormat)} numDatums={numDatums}")

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)
        ins.setExPipe(exPipe)
        ins.setNumDatums(numDatums)
        ins.setSrcFormat(srcFormat)
        ins.setDstFormat(dstFormat)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execunpacrtm__(self,ins):
        assert ins.getOp() == "UNPACR_TILE_MISC" , "Expected opcode UNPACR_TILE_MISC. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 7, "Seven attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 7, "Seven attribs expected. Received " + str(len(ins.getAttr()))

        src = [] ; dst = [] ; imm = [] ; vldUpd = {} ; bankUpd = {};
        exPipe  = None

        match (ins.getAttr()['Unpack_Type']):
            case 0:
                exPipe = "UNPACKER0"
                assert exPipe in self.pipes , str(exPipe) + " not found in configured engines"
                assert False, "Unhandled: Write to Metadata registers"
            case 1|3|6:
                reg_id = isaFunctions.regIndex.srcA
                dst.append(reg_id);
                if(ins.getAttr()["SetDatValid"]):
                    vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcA
                else:
                    vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcA
                exPipe = "UNPACKER0"
            case 2|4:
                reg_id = isaFunctions.regIndex.srcB
                dst.append(reg_id);
                if(ins.getAttr()["SetDatValid"]):
                    vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcB
                else:
                    vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcB
                exPipe = "UNPACKER1"
            case 5:
                reg_id = isaFunctions.regIndex.srcS
                dst.append(reg_id);
                if(ins.getAttr()["SetDatValid"]):
                    vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcS
                else:
                    vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcS
                exPipe = "UNPACKER2"
            case _:
                assert False, "Unhandled Unpack_Type value=" + str(ins.getAttr()['Unpack_Type']) + " for instruction " + ins.getOp()

        # Finding mapped formats and numDatums
        assert exPipe in self.pipes , str(exPipe) + " not found in configured engines"
        bufferFormatReg = f'BUFFER_DESCRIPTOR_TABLE_REG{ins.getAttr()["Buffer_Descriptor_Table_Sel"]}_TILE_FORMAT'
        outFormatReg = f'THCON_{exPipe}_REG0_OUT_DATA_FORMAT'
        srcFormat = self.tensixSplRegs.readCfgReg(bufferFormatReg)
        dstFormat = self.tensixSplRegs.readCfgReg(outFormatReg)
        if self.debug & 0x8:            print(f"Reading cfgReg {bufferFormatReg} and {outFormatReg}. Setting srcFormat={srcFormat}, dstFormat={dstFormat}")

        assert isaFunctions.getNumBytesFromDataFormat(srcFormat) != None, f"Unsupported format {srcFormat} in {bufferFormatReg}"
        assert isaFunctions.getNumBytesFromDataFormat(dstFormat) != None, f"Unsupported format {dstFormat} in {outFormatReg}"
        numDatums = 32 * 32     #Tile
        if self.debug & 0x8:            print(f"Computed srcSize={isaFunctions.getNumBytesFromDataFormat(srcFormat)}, dstSize={isaFunctions.getNumBytesFromDataFormat(dstFormat)} numDatums={numDatums}")

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)
        ins.setExPipe(exPipe)
        ins.setNumDatums(numDatums)
        ins.setSrcFormat(srcFormat)
        ins.setDstFormat(dstFormat)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execunpacrtz__(self,ins):
        assert ins.getOp() == "UNPACR_TILIZE" , "Expected opcode UNPACR_TILIZE. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 7, "Seven attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 7, "Seven attribs expected. Received " + str(len(ins.getAttr()))

        src = [] ; dst = [] ; imm = [] ; vldUpd = {} ; bankUpd = {};
        exPipe  = None

        match (ins.getAttr()['Unpack_Sel']):
            case 0:
                reg_id = isaFunctions.regIndex.srcA
                dst.append(reg_id);
                if(ins.getAttr()["SetDatValid"]):
                    vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcA
                else:
                    vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcA
                exPipe = "UNPACKER0"
            case 1:
                reg_id = isaFunctions.regIndex.srcB
                dst.append(reg_id);
                if(ins.getAttr()["SetDatValid"]):
                    vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcB
                else:
                    vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcB
                exPipe = "UNPACKER1"
            case 3:
                reg_id = isaFunctions.regIndex.dst
                dst.append(reg_id);
                if(ins.getAttr()["SetDatValid"]):
                    vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #dst
                else:
                    vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #dst
                exPipe = "UNPACKER0"
            case _:
                assert False, "Unhandled Unpack_Type value=" + str(ins.getAttr()['Unpack_Type']) + " for instruction " + ins.getOp()

        # Finding mapped formats and numDatums
        assert exPipe in self.pipes , str(exPipe) + " not found in configured engines"
        bufferFormatReg = f'BUFFER_DESCRIPTOR_TABLE_REG{ins.getAttr()["Buffer_Descriptor_Table_Sel"]}_TILE_FORMAT'
        outFormatReg = f'THCON_{exPipe}_REG0_OUT_DATA_FORMAT'
        srcFormat = self.tensixSplRegs.readCfgReg(bufferFormatReg)
        dstFormat = self.tensixSplRegs.readCfgReg(outFormatReg)
        if self.debug & 0x8:            print(f"Reading cfgReg {bufferFormatReg} and {outFormatReg}. Setting srcFormat={srcFormat}, dstFormat={dstFormat}")

        assert isaFunctions.getNumBytesFromDataFormat(srcFormat) != None, f"Unsupported format {srcFormat} in {bufferFormatReg}"
        assert isaFunctions.getNumBytesFromDataFormat(dstFormat) != None, f"Unsupported format {dstFormat} in {outFormatReg}"

        numDatums = self._compute_num_datums_from_row_cnt_enc(ins)

        if self.debug & 0x8:
            print(f"Computed srcSize={isaFunctions.getNumBytesFromDataFormat(srcFormat)}, dstSize={isaFunctions.getNumBytesFromDataFormat(dstFormat)} numDatums={numDatums}")

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)
        ins.setExPipe(exPipe)
        ins.setNumDatums(numDatums)
        ins.setSrcFormat(srcFormat)
        ins.setDstFormat(dstFormat)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execunpacrnop__(self,ins):
        assert ins.getOp() == "UNPACR_NOP" , "Expected opcode UNPACR_NOP. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))
            case decoded_instruction.instruction_kind.ttwh:
                assert len(ins.getAttr()) == 2, "Two attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 2, "Two attribs expected. Received " + str(len(ins.getAttr()))

        src = [] ; dst = [] ; imm = [] ; vldUpd = {} ; bankUpd = {};
        exPipe  = None

        if ins.kind == decoded_instruction.instruction_kind.ttqs:
            match (ins.getAttr()['Unpacker_Select']):
                case 0:
                    reg_id = isaFunctions.regIndex.srcA
                    dst.append(reg_id);
                    if(ins.getAttr()["Set_Dvalid"]):
                        vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcA
                    else:
                        vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcA
                    exPipe = "UNPACKER0"
                case 1:
                    reg_id = isaFunctions.regIndex.srcB
                    dst.append(reg_id);
                    if(ins.getAttr()["Set_Dvalid"]):
                        vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcB
                    else:
                        vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcB
                    exPipe = "UNPACKER1"
                case _:
                    assert False, "Unhandled Unpacker_Select value=" + str(ins.getAttr()['Unpack_Type']) + " for instruction " + ins.getOp()
        else:
            match (ins.getAttr()['Unpack_block_selection']):
                case 0:
                    reg_id = isaFunctions.regIndex.srcA
                    dst.append(reg_id);
                    vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcA
                    exPipe = "UNPACKER0"
                case 1:
                    reg_id = isaFunctions.regIndex.srcB
                    dst.append(reg_id);
                    vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcB
                    exPipe = "UNPACKER1"
                case _:
                    assert False, "Unhandled Unpacker_Select value=" + str(ins.getAttr()['Unpack_Type']) + " for instruction " + ins.getOp()

        assert exPipe in self.pipes , str(exPipe) + " not found in configured engines"

        # Finding mapped formats and numDatums
        # Nothing to do - Status Change only Op

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)
        ins.setExPipe(exPipe)
        ins.setNumDatums(0)
        ins.setSrcFormat(0)
        ins.setDstFormat(0)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execunpacr__(self,ins):
        assert ins.getOp() == "UNPACR", "Expected opcode UNPACR Received " + str(ins.getOp())
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert False, ins.getOp() + " not supported in QSR"
            case _:
                assert len(ins.getAttr()) == 13, "Thirteen attribs expected. Received " + str(len(ins.getAttr()))
                # assert False, "Unsupported currently"

        src = [] ; dst = [] ; imm = [] ; vldUpd = {} ; bankUpd = {};

        # assert ins.getAttr()['OvrdThreadId'] != 1, "UNPACR: Override of ThreadId not supported"
        # assert ins.getAttr()['Last'] != 1, "UNPACR: Last not supported" # Does not affect performance
        if(ins.getAttr()['OvrdThreadId'] == 1):            print("WARNING: UNPACR: Override of ThreadId not supported")
        match (ins.getAttr()['Unpack_block_selection']):
            case 0:
                reg_id = isaFunctions.regIndex.srcA
                dst.append(reg_id);
                if(ins.getAttr()["SetDatValid"]):
                    vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcA
                else:
                    vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcA
                exPipe = "UNPACKER0"
            case 1:
                reg_id = isaFunctions.regIndex.srcB
                dst.append(reg_id);
                if(ins.getAttr()["SetDatValid"]):
                    vldUpd[reg_id] = 1; bankUpd[reg_id] = 1; #srcB
                else:
                    vldUpd[reg_id] = 0; bankUpd[reg_id] = 0; #srcB
                exPipe = "UNPACKER1"
            case _:
                assert False, "Unhandled Unpacker_Select value=" + str(ins.getAttr()['Unpack_Type']) + " for instruction " + ins.getOp()

        # Finding mapped formats and numDatums
        assert exPipe in self.pipes , str(exPipe) + " not found in configured engines"
        bufferFormatReg = f'BUFFER_DESCRIPTOR_TABLE_REG{ins.getAttr()["Buffer_Descriptor_Table_Sel"]}_TILE_FORMAT'
        outFormatReg = f'THCON_{exPipe}_REG0_OUT_DATA_FORMAT'
        srcFormat = self.tensixSplRegs.readCfgReg(bufferFormatReg)
        dstFormat = self.tensixSplRegs.readCfgReg(outFormatReg)
        if self.debug & 0x8:            print(f"Reading cfgReg {bufferFormatReg} and {outFormatReg}. Setting srcFormat={srcFormat}, dstFormat={dstFormat}")

        assert isaFunctions.getNumBytesFromDataFormat(srcFormat) != None, f"Unsupported format {srcFormat} in {bufferFormatReg}"
        assert isaFunctions.getNumBytesFromDataFormat(dstFormat) != None, f"Unsupported format {dstFormat} in {outFormatReg}"
        numDatums = 16*16 #Tile
        if self.debug & 0x8:            print(f"Computed srcSize={isaFunctions.getNumBytesFromDataFormat(srcFormat)}, dstSize={isaFunctions.getNumBytesFromDataFormat(dstFormat)} numDatums={numDatums}")

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)
        ins.setExPipe(exPipe)
        ins.setNumDatums(numDatums)
        ins.setSrcFormat(srcFormat)
        ins.setDstFormat(dstFormat)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execpacrti__(self,ins):
        opList0 = ["PACR0_TILE_INC", "PACR1_TILE_INC", "PACR2_TILE_INC", "PACR_DEST_TILE_INC"]
        opList1 = ["PACR0_FACE_INC", "PACR1_FACE_INC", "PACR2_FACE_INC", "PACR_DEST_FACE_INC"]
        opList2 = ["PACR0_ROW_INC", "PACR1_ROW_INC", "PACR2_ROW_INC", "PACR_DEST_ROW_INC"]
        opList3 = ["PACR0_TILE", "PACR1_TILE", "PACR2_TILE", "PACR_DEST_TILE"]
        opList4 = ["PACR0_FACE", "PACR1_FACE", "PACR2_FACE", "PACR_DEST_FACE"]
        opList5 = ["PACR0_ROW", "PACR1_ROW", "PACR2_ROW", "PACR_DEST_ROW"]
        assert ins.getOp() in opList0 or ins.getOp() in opList1 or ins.getOp() in opList2 or ins.getOp() in opList3 or ins.getOp() in opList4 or ins.getOp() in opList5 , "Expected opcode PACR0/1/2/_DEST_TILE/FACE/ROW. Received " + str(ins.getOp())

        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        if("TILE" in ins.mnemonic):
            assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))
        elif("FACE" in ins.mnemonic):
            assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))
        elif("ROW" in ins.mnemonic):
            assert len(ins.getAttr()) == 8, "Eight attribs expected. Received " + str(len(ins.getAttr()))

        src = [] ; dst = [] ; imm = [] ; vldUpd = {} ; bankUpd = {};
        reg_id = None

        if("PACR0" in ins.mnemonic):
            reg_id = isaFunctions.regIndex.dst
            exPipe = "PACKER0"
        elif("PACR1" in ins.mnemonic):
            reg_id = isaFunctions.regIndex.srcS
            exPipe = "PACKER1"
        else:
            assert False, f"Unhandled Packer Op {ins.getOp()}"

        src.append(reg_id)
        if(ins.getAttr()["ClrDatValid"]):
            vldUpd[reg_id] = 1
            bankUpd[reg_id] = 1
        else:
            vldUpd[reg_id] = 0
            bankUpd[reg_id] = 0

        # Finding mapped formats and numDatums
        assert exPipe in self.pipes , f"{exPipe} not found in configured engines"
        bufferFormatReg = f'BUFFER_DESCRIPTOR_TABLE_REG{ins.getAttr()["Buffer_Descriptor_Table_Sel"]}_TILE_FORMAT'
        outFormatReg = f'THCON_{exPipe}_REG0_IN_DATA_FORMAT'
        srcFormat = self.tensixSplRegs.readCfgReg(outFormatReg)
        dstFormat = self.tensixSplRegs.readCfgReg(bufferFormatReg)
        if self.debug & 0x8:            print(f"Reading cfgReg {bufferFormatReg} and {outFormatReg}. Setting srcFormat={srcFormat}, dstFormat={dstFormat}")

        assert isaFunctions.getNumBytesFromDataFormat(srcFormat) != None, f"Unsupported format {srcFormat} in {bufferFormatReg}"
        assert isaFunctions.getNumBytesFromDataFormat(dstFormat) != None, f"Unsupported format {dstFormat} in {outFormatReg}"
        if("TILE" in ins.mnemonic):     numDatums = 32 * 32
        elif("FACE" in ins.mnemonic):   numDatums = 16 * 16
        elif("ROW" in ins.mnemonic):    numDatums = 1 * 16
        else:
            print(f"WARNING: Don't know how to calculate numDatums. Setting to 1")
            numDatums = 1
        if self.debug & 0x8:            print(f"Computed srcSize={isaFunctions.getNumBytesFromDataFormat(srcFormat)}, dstSize={isaFunctions.getNumBytesFromDataFormat(dstFormat)} numDatums={numDatums}")

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)
        ins.setExPipe(exPipe)
        ins.setNumDatums(numDatums)
        ins.setSrcFormat(srcFormat)
        ins.setDstFormat(dstFormat)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execpacr_stride__(self, ins):
        assert ins.getOp() == "PACR_STRIDE", "Expected opcode PACR_STRIDE Received " + str(ins.getOp())
        assert len(ins.getAttr()) == 8, "Eight attribs expected. Received "  + str(len(ins.getAttr()))
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        src = []
        dst = []
        imm = []
        vldUpd = {}
        bankUpd = {}
        exPipe = None
        reg_id = None

        packerSel = ins.getAttr()['PackerSel']
        match packerSel:
            case 0:
                reg_id = isaFunctions.regIndex.dst
                exPipe = "PACKER0"
            case 1:
                reg_id = isaFunctions.regIndex.srcS
                exPipe = "PACKER1"
            case _:
                assert False, f"Unhandled PackerSel value = {packerSel} for instruction {ins.getOp()}"

        src.append(reg_id)
        if(ins.getAttr()["ClrDatValid"]):
            vldUpd[reg_id] = 1
            bankUpd[reg_id] = 1
        else:
            vldUpd[reg_id] = 0
            bankUpd[reg_id] = 0

        # Finding mapped formats and numDatums
        assert exPipe in self.pipes , f"{exPipe} not found in configured engines"
        bufferFormatReg = f'BUFFER_DESCRIPTOR_TABLE_REG{ins.getAttr()["Buffer_Descriptor_Table_Sel"]}_TILE_FORMAT'
        outFormatReg = f'THCON_{exPipe}_REG0_IN_DATA_FORMAT'
        srcFormat = self.tensixSplRegs.readCfgReg(outFormatReg)
        dstFormat = self.tensixSplRegs.readCfgReg(bufferFormatReg)
        if self.debug & 0x8:            print(f"Reading cfgReg {bufferFormatReg} and {outFormatReg}. Setting srcFormat={srcFormat}, dstFormat={dstFormat}")

        assert isaFunctions.getNumBytesFromDataFormat(srcFormat) != None, f"Unsupported format {srcFormat} in {bufferFormatReg}"
        assert isaFunctions.getNumBytesFromDataFormat(dstFormat) != None, f"Unsupported format {dstFormat} in {outFormatReg}"
        numDatums = 4 * 16  # TODO: Upto 4 Rows. Check for THCON_PACKER<0/1>_REG3_PACK_STRIDE_NO_WRITE. A contiguous 64B with NoWrite Set could save a write.
                            # We can write at least 64 datums per cycle, so ignore because of no BW savings ?
        if self.debug & 0x8:            print(f"Computed srcSize={isaFunctions.getNumBytesFromDataFormat(srcFormat)}, dstSize={isaFunctions.getNumBytesFromDataFormat(dstFormat)} numDatums={numDatums}")

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)
        ins.setExPipe(exPipe)
        ins.setNumDatums(numDatums)
        ins.setSrcFormat(srcFormat)
        ins.setDstFormat(dstFormat)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execpacr_untilize__(self, ins):
        assert ins.kind == decoded_instruction.instruction_kind.ttqs, "PACR_UNTILIZE not defined for " + str(ins.kind)
        assert ins.getOp() == "PACR_UNTILIZE", "Expected opcode PACR_UNTILIZE Received " + str(ins.getOp())
        assert len(ins.getAttr()) == 7, "Seven attribs expected for instruction PACR_UNTILIZE. Received "  + str(len(ins.getAttr()))
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        src = []
        dst = []
        imm = []
        vldUpd = {}
        bankUpd = {}
        exPipe = None
        reg_id = None

        packerSel = ins.getAttr()['Packer_Sel']
        match packerSel:
            case 0:
                reg_id = isaFunctions.regIndex.dst
                exPipe = "PACKER0"
            case 1:
                assert False, f"PackerSel value of {packerSel} for instruction PACR_UNTILIZE is reserved. @instruction: {ins}"
            case _:
                assert False, f"Unhandled Packer_Sel value = {packerSel} for instruction {ins.getOp()}"

        src.append(reg_id)
        if(ins.getAttr()["ClrDatValid"]):
            vldUpd[reg_id] = 1
            bankUpd[reg_id] = 1
        else:
            vldUpd[reg_id] = 0
            bankUpd[reg_id] = 0

        # Finding mapped formats and numDatums
        assert exPipe in self.pipes , f"{exPipe} not found in configured engines"
        bufferFormatReg = f'BUFFER_DESCRIPTOR_TABLE_REG{ins.getAttr()["Buffer_Descriptor_Table_Sel"]}_TILE_FORMAT'
        outFormatReg = f'THCON_{exPipe}_REG0_IN_DATA_FORMAT'
        srcFormat = self.tensixSplRegs.readCfgReg(outFormatReg)
        dstFormat = self.tensixSplRegs.readCfgReg(bufferFormatReg)
        if self.debug & 0x8:            print(f"Reading cfgReg {bufferFormatReg} and {outFormatReg}. Setting srcFormat={srcFormat}, dstFormat={dstFormat}")

        assert isaFunctions.getNumBytesFromDataFormat(srcFormat) != None, f"Unsupported format {srcFormat} in {bufferFormatReg}"
        assert isaFunctions.getNumBytesFromDataFormat(dstFormat) != None, f"Unsupported format {dstFormat} in {outFormatReg}"

        numDatums = self._compute_num_datums_from_row_cnt_enc(ins)

        if self.debug & 0x8:
            print(f"Computed srcSize={isaFunctions.getNumBytesFromDataFormat(srcFormat)}, dstSize={isaFunctions.getNumBytesFromDataFormat(dstFormat)} numDatums={numDatums}")

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)
        ins.setExPipe(exPipe)
        ins.setNumDatums(numDatums)
        ins.setSrcFormat(srcFormat)
        ins.setDstFormat(dstFormat)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execgpool__(self,ins):
        assert ins.getOp() == "GAPOOL" or ins.getOp() == "GMPOOL", "Expected opcode G[A/M]POOL. Received " + str(ins.getOp())
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        assert len(ins.getAttr()) == 5, "Five attribs expected. Received " + str(len(ins.getAttr()))

        src = [] ; dst = [] ; imm = []; vldUpd ={}; bankUpd = {};

        dst.append(3);  vldUpd[3] = 0; bankUpd[3] = 0; #dst0
        src.append(0);  vldUpd[0] = 0; bankUpd[0] = 0; #srcA
        src.append(1);  vldUpd[1] = 0; bankUpd[1] = 0; #srcB
        match ins.getAttr()["clear_dvalid"]:
            case 1: vldUpd[0] = 1; bankUpd[0] = 1; #srcA
            case 2: vldUpd[1] = 1; bankUpd[1] = 1; #srcB
            case 3:
                    vldUpd[0] = 1; bankUpd[0] = 1; #srcA
                    vldUpd[1] = 1; bankUpd[1] = 1; #srcB

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execelwadd__(self,ins):
        assert ins.getOp() == "ELWADD", "Expected opcode ELWADD. Received " + str(ins.getOp())
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 5, "Five attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))

        src = [] ; dst = [] ; imm = []; vldUpd ={}; bankUpd = {};

        dst.append(3);  vldUpd[3] = 0; bankUpd[3] = 0; #dst0
        src.append(0);  vldUpd[0] = 0; bankUpd[0] = 0; #srcA
        src.append(1);  vldUpd[1] = 0; bankUpd[1] = 0; #srcB

        match ins.getAttr()["clear_dvalid"]:
            case 1: vldUpd[0] = 1; bankUpd[0] = 1;     #srcA
            case 2: vldUpd[1] = 1; bankUpd[1] = 1;     #srcB
            case 3:
                    vldUpd[0] = 1; bankUpd[0] = 1;     #srcA
                    vldUpd[1] = 1; bankUpd[1] = 1;     #srcB

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execelwsub__(self,ins):
        assert ins.getOp() == "ELWSUB", "Expected opcode ELWSUB. Received " + str(ins.getOp())
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 5, "Five attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))

        src = [] ; dst = [] ; imm = []; vldUpd ={}; bankUpd = {};

        dst.append(3);  vldUpd[3] = 0; bankUpd[3] = 0; #dst0
        src.append(0);  vldUpd[0] = 0; bankUpd[0] = 0; #srcA
        src.append(1);  vldUpd[1] = 0; bankUpd[1] = 0; #srcB

        match ins.getAttr()["clear_dvalid"]:
            case 1: vldUpd[0] = 1; bankUpd[0] = 1;     #srcA
            case 2: vldUpd[1] = 1; bankUpd[1] = 1;     #srcB
            case 3:
                    vldUpd[0] = 1; bankUpd[0] = 1;     #srcA
                    vldUpd[1] = 1; bankUpd[1] = 1;     #srcB

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execelwmul__(self,ins):
        assert ins.getOp() == "ELWMUL", "Expected opcode ELWMUL. Received " + str(ins.getOp())
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 5, "Five attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))

        src = [] ; dst = [] ; imm = []; vldUpd ={}; bankUpd = {};

        dst.append(3);  vldUpd[3] = 0; bankUpd[3] = 0; #dst0
        src.append(0);  vldUpd[0] = 0; bankUpd[0] = 0; #srcA
        src.append(1);  vldUpd[1] = 0; bankUpd[1] = 0; #srcB

        match ins.getAttr()["clear_dvalid"]:
            case 1: vldUpd[0] = 1; bankUpd[0] = 1;     #srcA
            case 2: vldUpd[1] = 1; bankUpd[1] = 1;     #srcB
            case 3:
                    vldUpd[0] = 1; bankUpd[0] = 1;     #srcA
                    vldUpd[1] = 1; bankUpd[1] = 1;     #srcB

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execmvmul__(self,ins):
        assert ins.getOp() == "MVMUL", "Expected opcode MVMUL. Received " + str(ins.getOp())
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))

        src = [] ; dst = [] ; imm = []; vldUpd ={}; bankUpd = {};

        dst.append(3);  vldUpd[3] = 0; bankUpd[3] = 0; #dst0
        src.append(0);  vldUpd[0] = 0; bankUpd[0] = 0; #srcA
        src.append(1);  vldUpd[1] = 0; bankUpd[1] = 0; #srcB

        match ins.getAttr()["clear_dvalid"]:
            case 1: vldUpd[0] = 1; bankUpd[0] = 1;     #srcA
            case 2: vldUpd[1] = 1; bankUpd[1] = 1;     #srcB
            case 3:
                    vldUpd[0] = 1; bankUpd[0] = 1;     #srcA
                    vldUpd[1] = 1; bankUpd[1] = 1;     #srcB

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execmvmuldi__(self,ins):
        assert ins.getOp() == "MVMULDI", "Expected opcode MVMULDI. Received " + str(ins.getOp())
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                raise Exception(f"- instruction MVMULDI is not defined for {ins.kind}")

        src = [] ; dst = [] ; imm = []; vldUpd ={}; bankUpd = {};

        dst.append(3);  vldUpd[3] = 0; bankUpd[3] = 0; #dst0
        src.append(0);  vldUpd[0] = 0; bankUpd[0] = 0; #srcA
        src.append(1);  vldUpd[1] = 0; bankUpd[1] = 0; #srcB

        match ins.getAttr()["clear_dvalid"]:
            case 1: vldUpd[0] = 1; bankUpd[0] = 1;     #srcA
            case 2: vldUpd[1] = 1; bankUpd[1] = 1;     #srcB
            case 3:
                    vldUpd[0] = 1; bankUpd[0] = 1;     #srcA
                    vldUpd[1] = 1; bankUpd[1] = 1;     #srcB

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execatgetm__(self,ins):
        assert ins.getOp() == "ATGETM", "Expected opcode ATGETM. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        assert len(ins.getAttr()) == 1, "One attribs expected. Received " + str(len(ins.getAttr()))

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execatrelm__(self,ins):
        assert ins.getOp() == "ATRELM", "Expected opcode ATRELM. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        assert len(ins.getAttr()) == 1, "One attribs expected. Received " + str(len(ins.getAttr()))

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execdmanop__(self,ins):
        assert ins.getOp() == "DMANOP", "Expected opcode DMANOP. Received " + str(ins.getOp())
        # assert "operands" not in dir(ins) , "Zero Dst/Src/Imm/Attribs expected"
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execnop__(self,ins):
        assert ins.getOp() == "NOP", "Expected opcode NOP. Received " + str(ins.getOp())
        print (ins)
        # assert "operands" not in dir(ins) , "Zero Dst/Src/Imm/Attribs expected"
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execseminit__(self,ins):
        # assert ins.getOp() == "SEMINIT", "Expected opcode SEMINIT. Received " + str(ins.getOp())
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        # assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        # match ins.kind:
        #     case decoded_instruction.instruction_kind.ttqs:
        #         assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))
        #     case _:
        #         assert False, "Unhandled Instructin Kind"

        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 4,                   f"Four attribs expected. Received " + str(len(ins.getAttr()))
                assert hasattr(ins, 'operands'),                  f"Expected decoded instruction to have operands"
                assert hasattr(ins.operands, 'all'),              f"Expected decoded instruction operands to have attribute 'all'"
                assert "sem_sel"      in ins.operands.all.keys(), f"Could not find semaphore ID from decoded operands. Given operands: {ins.operands.all}"
                assert "sem_bank_sel" in ins.operands.all.keys(), f"Could not find semaphore bank ID from operands. Given operands: {ins.operands.all}"
                assert "init_value"   in ins.operands.all.keys(), f"Could not find semaphore initial value from operands. Given operands: {ins.operands.all}"
                assert "max_value"    in ins.operands.all.keys(), f"Could not find semaphore max value from operands. Given operands: {ins.operands.all}"

            case decoded_instruction.instruction_kind.ttwh:
                assert len(ins.getAttr()) == 3,                   f"Three attribs expected. Received " + str(len(ins.getAttr()))
                assert hasattr(ins, 'operands'),                  f"Expected decoded instruction to have operands"
                assert hasattr(ins.operands, 'all'),              f"Expected decoded instruction operands to have attribute 'all'"
                assert "sem_sel"      in ins.operands.all.keys(), f"Could not find semaphore ID from decoded operands. Given operands: {ins.operands.all}"
                assert "init_value"   in ins.operands.all.keys(), f"Could not find semaphore initial value from operands. Given operands: {ins.operands.all}"
                assert "max_value"    in ins.operands.all.keys(), f"Could not find semaphore max value from operands. Given operands: {ins.operands.all}"
            case _:
                assert False, "Unhandled Instruction Kind"

        # semaphores:
          # semaphore 0:
            # ID
            # Bank ID
            # init value
            # max value
            # current value
            # wait cond
            # ..
          # semaphore 1:
            # ID
            # Bank ID
            # init value
            # max value
            # current value
            # wait condition
          # ...

        sem_sel = ins.operands.all['sem_sel']
        if not (isinstance(sem_sel, int) and sem_sel > 0 and (sem_sel & (sem_sel - 1)) == 0):
            raise ValueError(f"sem_sel must be a positive power of 2 (one-hot encoding), got {sem_sel}")
        semaphore_id_in_bank = int(math.log2(sem_sel))   # One-hot encoding

        if (ins.kind == decoded_instruction.instruction_kind.ttqs):
            bank_id          = ins.operands.all['sem_bank_sel']
        else:
            bank_id          = 0
        semaphore_id         = semaphore_id_in_bank + (NUM_TTSEMAPHORES_PER_BANK * bank_id)
        reg_offset           = semaphore_id * NUM_VARIABLES_PER_TTSEMAPHORE

        semaphores = self.tensixSplRegs.reg['ttsemaphores']
        # semaphores = self.reg['ttsemaphores']
        semaphores[reg_offset + tt_semaphore_idx.id]             = semaphore_id_in_bank
        semaphores[reg_offset + tt_semaphore_idx.bank]           = bank_id
        semaphores[reg_offset + tt_semaphore_idx.init_value]     = ins.operands.all['init_value']
        semaphores[reg_offset + tt_semaphore_idx.max_value]      = ins.operands.all['max_value']
        semaphores[reg_offset + tt_semaphore_idx.current_value]  = ins.operands.all['init_value']
        stalled_pipes_as_bits = semaphores[reg_offset + tt_semaphore_idx.pipes_to_stall]
        thread_id_of_pipes = semaphores[reg_offset + tt_semaphore_idx.pipes_to_stall]

        if -1 == stalled_pipes_as_bits: # default value before initialisation
            if -1 != thread_id_of_pipes:
                msg  = f"- error @__seminit__: expected the thread_id_of_pipes variable value to be -1 as stalled_pipes_as_bits is {stalled_pipes_as_bits}.\n"
                msg += f"- this is because value of {stalled_pipes_as_bits} for stalled_pipes_as_bits indicates the semaphore is being initialised, hence thread_id_of_pipes is expected to be uninitialised as well.\n"
                msg += f"- instruction info: {ins}"

                if self.debug & 0x20:
                    print(msg)
                else:
                    raise Exception(msg)

            stalled_pipes_as_bits = 0 # no pipes to be stalled.
            thread_id_of_pipes = -1 # no pipes to stall, so no thread id.

        if stalled_pipes_as_bits:
            # semaphore is being reinitialised.
            # we assume the following model:
            # during initialisation, any stalled pipes will be set free.

            stalled_pipe_ids = [i for i in range(stalled_pipes_as_bits.bit_length()) if ((stalled_pipes_as_bits >> i) & 1)]
            if not stalled_pipe_ids:
                raise Exception(f"- error @__seminit__: could not obtained stalled pipe ids at execution of instruction: {ins}")

            if -1 == thread_id_of_pipes:
                msg  = f"- error @__seminit__: expected the thread_id_of_pipes variable value not to be -1 as stalled_pipes_as_bits is {stalled_pipes_as_bits}.\n"
                msg += f"- this is because value of {stalled_pipes_as_bits} for stalled_pipes_as_bits indicates the semaphore is being reinitialised\n"
                msg += f"- our present reinitialisation model assumes the following: we unstall all the previously stalled resources for given thread_id_of_pipes.\n"
                msg += f"- hence there thread_id_of_pipes is expected to return appropriate thread ID value between 0 and (number of threads - 1). given value of thread_id_of_pipes is {thread_id_of_pipes}"
                msg += f"- instruction info: {ins}"

                if self.debug & 0x20:
                    print(msg)
                else:
                    raise Exception(msg)

            ins.setDstPipes(stalled_pipe_ids) # pipes to unstall
            ins.setPipesThreadId(thread_id_of_pipes) # the thread ID of the pipes

            stalled_pipes_as_bits = 0 # no pipes to be stalled.
            thread_id_of_pipes = -1 # no pipes to stall, so no thread id.

        if self.debug & 0x20:     print(f"SEMINIT Writing SemaphoreReg{semaphore_id}={semaphores[reg_offset + tt_semaphore_idx.current_value]}")
        self.tensixSplRegs.__writeReg__(semaphore_id, semaphores[reg_offset + tt_semaphore_idx.current_value], 'ttsemaphores')

        semaphores[reg_offset + tt_semaphore_idx.pipes_to_stall] = stalled_pipes_as_bits
        semaphores[reg_offset + tt_semaphore_idx.thread_id_of_pipes] = thread_id_of_pipes

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsemget__(self,ins):
        assert ins.getOp() == "SEMGET", "Expected opcode SEMGET. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 2, "Two attribs expected. Received " + str(len(ins.getAttr()))
            case decoded_instruction.instruction_kind.ttwh:
                assert len(ins.getAttr()) == 1, "One attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 1, "One attribs expected. Received " + str(len(ins.getAttr()))

        sem_sel = ins.operands.all['sem_sel']
        if not (isinstance(sem_sel, int) and sem_sel > 0 and (sem_sel & (sem_sel - 1)) == 0):
            raise ValueError(f"sem_sel must be a positive power of 2 (one-hot encoding), got {sem_sel}")
        semaphore_id_in_bank = int(math.log2(sem_sel))   # One-hot encoding

        if (ins.kind == decoded_instruction.instruction_kind.ttqs):
            bank_id          = ins.operands.all['sem_bank_sel']
        else:
            bank_id          = 0
        semaphore_id         = semaphore_id_in_bank + (NUM_TTSEMAPHORES_PER_BANK * bank_id)
        reg_offset           = semaphore_id * NUM_VARIABLES_PER_TTSEMAPHORE

        semaphores    = self.tensixSplRegs.reg['ttsemaphores']
        init_value    = semaphores[reg_offset + tt_semaphore_idx.init_value]
        max_value     = semaphores[reg_offset + tt_semaphore_idx.max_value]
        current_value = semaphores[reg_offset + tt_semaphore_idx.current_value]

        if (semaphore_id_in_bank != semaphores[reg_offset + tt_semaphore_idx.id]) or (bank_id != semaphores[reg_offset + tt_semaphore_idx.bank]):
            msg =  f"- error: semaphore ID and bank ID mismatch. Perhaps offset is not calculated corrected, or semaphore is not initialised.\n"
            msg += f"- Semaphore ID in bank and bank ID from instruction:         {semaphore_id_in_bank}, {bank_id}.\n"
            msg += f"- Semaphore ID in bank and bank ID from offset calculations: {semaphores[reg_offset + 0]}, {semaphores[reg_offset + 1]}.\n"
            raise Exception(msg)

        if (-1 == init_value) or (-1 == max_value) or (-1 == current_value):
            raise Exception(f"- error: semaphore [ID: {semaphore_id} (ID in bank: {semaphore_id_in_bank}, bank: {bank_id})] is perhaps not initialised!")

        if (0 == current_value):
            msg  = f"- error: SEMGET called on semaphore whose current value is 0. semaphore instruction: {ins}\n"
            msg += f"  init value:    {init_value}\n"
            msg += f"  max value:     {max_value}\n"
            msg += f"  current value: {current_value}"

            if self.debug & 0x20:
                print(msg)
            else:
                raise Exception(msg)

        if current_value < init_value:
            print("WARNING: SEMAPHORE CURRENT VALUE IS LESS THAN INITIAL VALUE")

        if current_value > max_value:
            print("WARNING: SEMAPHORE CURRENT VALUE IS LESS THAN MAX VALUE")

        semaphores[reg_offset + tt_semaphore_idx.current_value] -= 1

        if semaphores[reg_offset + tt_semaphore_idx.pipes_to_stall]:
            stalled_pipes_value = semaphores[reg_offset + tt_semaphore_idx.pipes_to_stall]
            stalled_pipe_ids = [i for i in range(stalled_pipes_value.bit_length()) if ((stalled_pipes_value >> i) & 1)]
            if not stalled_pipe_ids:
                msg = f"- error: could not obtained stalled pipe ids. stalled_pipe_ids = {stalled_pipe_ids}, stalled_pipes_value = {stalled_pipes_value}. instr_info = {ins}"
                if self.debug & 0x20:
                    print(msg)
                else:
                    raise Exception(msg)

            thread_id_of_pipes = semaphores[reg_offset + tt_semaphore_idx.thread_id_of_pipes]

            if -1 == thread_id_of_pipes:
                stalled_pipes_as_bits = stalled_pipes_value
                msg  = f"- error @__execsemget__: expected the thread_id_of_pipes variable value not to be -1 as stalled_pipes_as_bits is {stalled_pipes_as_bits}.\n"
                msg += f"- this is because value of {stalled_pipes_as_bits} for stalled_pipes_as_bits indicates the semaphore is being reinitialised\n"
                msg += f"- our present reinitialisation model assumes the following: we unstall all the previously stalled resources for given thread_id_of_pipes.\n"
                msg += f"- hence there thread_id_of_pipes is expected to return appropriate thread ID value between 0 and (number of threads - 1). given value of thread_id_of_pipes is {thread_id_of_pipes}"
                msg += f"- instruction info: {ins}"

                if self.debug & 0x20:
                    print(msg)
                else:
                    raise Exception(msg)

            ins.setDstPipes(stalled_pipe_ids)
            ins.setPipesThreadId(thread_id_of_pipes)
            # print(f"SEMGET: stalled pipe ids: {stalled_pipe_ids}, thread ID of the pipes: {}. instruction: {ins}")

            semaphores[reg_offset + tt_semaphore_idx.pipes_to_stall] = 0 # Do not stall any pipes.
            semaphores[reg_offset + tt_semaphore_idx.thread_id_of_pipes] = -1 # no pipes to stall, so thread ID is -1.

        if self.debug & 0x20:       print(f"SEMGET Writing SemaphoreReg{semaphore_id}={semaphores[reg_offset + tt_semaphore_idx.current_value]}")
        self.tensixSplRegs.__writeReg__(semaphore_id, semaphores[reg_offset + tt_semaphore_idx.current_value], 'ttsemaphores')

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsempost__(self,ins):
        assert ins.getOp() == "SEMPOST", "Expected opcode SEMPOST. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 2, "Two attribs expected. Received " + str(len(ins.getAttr()))
            case decoded_instruction.instruction_kind.ttwh:
                assert len(ins.getAttr()) == 1, "One attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 1, "One attribs expected. Received " + str(len(ins.getAttr()))

        sem_sel = ins.operands.all['sem_sel']
        if not (isinstance(sem_sel, int) and sem_sel > 0 and (sem_sel & (sem_sel - 1)) == 0):
            raise ValueError(f"sem_sel must be a positive power of 2 (one-hot encoding), got {sem_sel}")
        semaphore_id_in_bank = int(math.log2(sem_sel))   # One-hot encoding

        if (ins.kind == decoded_instruction.instruction_kind.ttqs):
            bank_id          = ins.operands.all['sem_bank_sel']
        else:
            bank_id          = 0
        semaphore_id         = semaphore_id_in_bank + (NUM_TTSEMAPHORES_PER_BANK * bank_id)
        reg_offset           = semaphore_id * NUM_VARIABLES_PER_TTSEMAPHORE

        semaphores    = self.tensixSplRegs.reg['ttsemaphores']
        # semaphores    = self.reg['ttsemaphores']
        init_value    = semaphores[reg_offset + tt_semaphore_idx.init_value]
        max_value     = semaphores[reg_offset + tt_semaphore_idx.max_value]
        current_value = semaphores[reg_offset + tt_semaphore_idx.current_value]

        if (semaphore_id_in_bank != semaphores[reg_offset + tt_semaphore_idx.id]) or (bank_id != semaphores[reg_offset + tt_semaphore_idx.bank]):
            msg =  f"- error: semaphore ID and bank ID mismatch. Perhaps offset is not calculated corrected, or semaphore is not initialised.\n"
            msg += f"- Semaphore ID in bank and bank ID from instruction:         {semaphore_id_in_bank}, {bank_id}.\n"
            msg += f"- Semaphore ID in bank and bank ID from offset calculations: {semaphores[reg_offset + 0]}, {semaphores[reg_offset + 1]}."
            raise Exception(msg)

        if (-1 == init_value) or (-1 == max_value) or (-1 == current_value):
            msg  = f"- error @__execsempost__: semaphore [ID: {semaphore_id} (ID in bank: {semaphore_id_in_bank}, bank: {bank_id})] is perhaps not initialised!\n"
            msg += f"instruction:   {ins}\n"
            msg += f"init value:    {init_value}\n"
            msg += f"max value:     {max_value}\n"
            msg += f"current value: {current_value}\n"
            if self.debug & 0x20:
                print(msg)
            else:
                raise Exception(msg.rstrip())

        if current_value < init_value:
            print("WARNING: SEMAPHORE CURRENT VALUE IS LESS THAN INITIAL VALUE")

        if current_value > max_value:
            print("WARNING: SEMAPHORE CURRENT VALUE IS LESS THAN MAX VALUE")

        semaphores[reg_offset + tt_semaphore_idx.current_value] += 1

        if semaphores[reg_offset + tt_semaphore_idx.pipes_to_stall]:
            stalled_pipes_value = semaphores[reg_offset + tt_semaphore_idx.pipes_to_stall]
            stalled_pipe_ids = [i for i in range(stalled_pipes_value.bit_length()) if ((stalled_pipes_value >> i) & 1)]

            if not stalled_pipe_ids:
                msg = f"- error: could not obtained stalled pipe ids. stalled_pipe_ids = {stalled_pipe_ids}, stalled_pipes_value = {stalled_pipes_value}. instr_info = {ins}"
                if self.debug & 0x20:
                    print(msg)
                else:
                    raise Exception(msg)

            thread_id_of_pipes = semaphores[reg_offset + tt_semaphore_idx.thread_id_of_pipes]

            if -1 == thread_id_of_pipes:
                stalled_pipes_as_bits = stalled_pipes_value
                msg  = f"- error @__execsempost__. expected the thread_id_of_pipes variable value not to be -1 as stalled_pipes_as_bits is {stalled_pipes_as_bits}.\n"
                msg += f"- this is because value of {stalled_pipes_as_bits} for stalled_pipes_as_bits indicates the semaphore is being reinitialised\n"
                msg += f"- our present reinitialisation model assumes the following: we unstall all the previously stalled resources for given thread_id_of_pipes.\n"
                msg += f"- hence there thread_id_of_pipes is expected to return appropriate thread ID value between 0 and (number of threads - 1). given value of thread_id_of_pipes is {thread_id_of_pipes}"
                msg += f"- instruction info: {ins}"

                if self.debug & 0x20:
                    print(msg)
                else:
                    raise Exception(msg)

            ins.setDstPipes(stalled_pipe_ids)
            ins.setPipesThreadId(thread_id_of_pipes)

            semaphores[reg_offset + tt_semaphore_idx.pipes_to_stall] = 0 # Do not stall any pipes further.
            semaphores[reg_offset + tt_semaphore_idx.thread_id_of_pipes] = -1 # no pipes to stall, so thread ID is -1.

        if self.debug & 0x20:     print(f"SEMPOST Writing SemaphoreReg{semaphore_id}={semaphores[reg_offset + tt_semaphore_idx.current_value]}")
        self.tensixSplRegs.__writeReg__(semaphore_id, semaphores[reg_offset + tt_semaphore_idx.current_value], 'ttsemaphores')

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsemwait__(self,ins):
        assert ins.getOp() == "SEMWAIT", "Expected opcode SEMWAIT. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))
            case decoded_instruction.instruction_kind.ttwh:
                assert len(ins.getAttr()) == 3, "Three attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))

        value_if_uninitialised = -1

        sem_sel = ins.operands.all['sem_sel']
        if not (isinstance(sem_sel, int) and sem_sel > 0 and (sem_sel & (sem_sel - 1)) == 0):
            raise ValueError(f"sem_sel must be a positive power of 2 (one-hot encoding), got {sem_sel}")
        semaphore_id_in_bank = int(math.log2(sem_sel))   # One-hot encoding

        if (ins.kind == decoded_instruction.instruction_kind.ttqs):
            bank_id          = ins.operands.all['sem_bank_sel']
        else:
            bank_id          = 0
        semaphore_id         = semaphore_id_in_bank + (NUM_TTSEMAPHORES_PER_BANK * bank_id)
        reg_offset           = semaphore_id * NUM_VARIABLES_PER_TTSEMAPHORE

        semaphores    = self.tensixSplRegs.reg['ttsemaphores']
        # semaphores    = self.reg['ttsemaphores']

        if value_if_uninitialised == semaphores[reg_offset + tt_semaphore_idx.id]:
            if not value_if_uninitialised == semaphores[reg_offset + tt_semaphore_idx.bank]:
                raise Exception(f"- error: the semaphore ID is {value_if_uninitialised}, but the bank value is {semaphores[reg_offset + tt_semaphore_idx.bank]}, expected the bank value to be {semaphores[reg_offset + tt_semaphore_idx.bank]}")

            semaphores[reg_offset + tt_semaphore_idx.id]   = semaphore_id_in_bank
            semaphores[reg_offset + tt_semaphore_idx.bank] = bank_id
            # as semaphore is not initialised, we reset it.
            # during reset we set init value, max value and current value to 0.
            semaphores[reg_offset + tt_semaphore_idx.init_value]     = 0
            semaphores[reg_offset + tt_semaphore_idx.max_value]      = semaphores[reg_offset + tt_semaphore_idx.init_value]
            semaphores[reg_offset + tt_semaphore_idx.current_value]  = semaphores[reg_offset + tt_semaphore_idx.init_value]
            semaphores[reg_offset + tt_semaphore_idx.pipes_to_stall] = 0

        init_value    = semaphores[reg_offset + tt_semaphore_idx.init_value]
        max_value     = semaphores[reg_offset + tt_semaphore_idx.max_value]
        current_value = semaphores[reg_offset + tt_semaphore_idx.current_value]

        if (semaphore_id_in_bank != semaphores[reg_offset + 0]) or (bank_id != semaphores[reg_offset + 1]):
            msg =  f"- error: semaphore ID and bank ID mismatch. Perhaps offset is not calculated corrected, or semaphore is not initialised.\n"
            msg += f"- Semaphore ID in bank and bank ID from instruction:         {semaphore_id_in_bank}, {bank_id}.\n"
            msg += f"- Semaphore ID in bank and bank ID from offset calculations: {semaphores[reg_offset + 0]}, {semaphores[reg_offset + 1]}."
            raise Exception(msg)

        if (value_if_uninitialised == init_value) or (value_if_uninitialised == max_value) or (value_if_uninitialised == current_value):
            msg = f"- error: semaphore [ID: {semaphore_id} (ID in bank: {semaphore_id_in_bank}, bank: {bank_id})] is perhaps not initialised!"
            raise Exception(msg)

        if current_value < init_value:
            print("WARNING: SEMAPHORE CURRENT VALUE IS LESS THAN INITIAL VALUE")

        if current_value > max_value:
            print("WARNING: SEMAPHORE CURRENT VALUE IS LESS THAN MAX VALUE")

        wait_cond = ins.operands.all['wait_sem_cond']
        semaphores[reg_offset + tt_semaphore_idx.wait_sem_cond] = wait_cond

        stall_res = ins.operands.attributes['stall_res']

        if not isinstance(stall_res, list):
            raise Exception(f"- error: expected stall_res to be of type list, received {type(stall_res)}.")

        stall_pipes = ((1 == wait_cond) and (0 == current_value)) or ((2 == wait_cond) and (max_value == current_value))

        # srcPipes = []
        dstPipes = isaFunctions.getT3SimPipesFromStallRes(stall_res, self.pipeGrps, self.pipes) if stall_pipes else []
        if(self.debug & 0x8):
            msg  = f"instruction: {ins}\n"
            msg += f"current value: {current_value}\n"
            msg += f"max value:     {max_value}\n"
            msg += f"wait_cond:     {wait_cond}\n"
            msg += f"stall_pipes:   {stall_pipes}\n"
            msg += f"stall_res:     {stall_res}\n"
            msg += f"stall_res:     {stall_res}\n"
            msg += f"dstPipes:      {dstPipes}"

            print(msg)

        if dstPipes:
            ins.setDstPipes(dstPipes)
            pipes_as_bits = semaphores[reg_offset + tt_semaphore_idx.pipes_to_stall]
            for p in dstPipes:
                pipes_as_bits |= (1 << p)

            semaphores[reg_offset + tt_semaphore_idx.pipes_to_stall] = pipes_as_bits
            semaphores[reg_offset + tt_semaphore_idx.thread_id_of_pipes] = ins.getThread()
        else:
            semaphores[reg_offset + tt_semaphore_idx.pipes_to_stall] = 0
            semaphores[reg_offset + tt_semaphore_idx.thread_id_of_pipes] = -1

        # ins.setSrcPipes(srcPipes)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsetadcxx__(self,ins):
        assert ins.getOp() == "SETADCXX", "Expected opcode SETADCXX. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        assert len(ins.getAttr()) == 3, "Six attribs expected. Received " + str(len(ins.getAttr()))

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsetadcxy__(self,ins):
        assert ins.getOp() == "SETADCXY", "Expected opcode SETADCXY. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsetadczw__(self,ins):
        assert ins.getOp() == "SETADCZW", "Expected opcode SETADCZW. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsetc16__(self,ins):
        assert ins.getOp() == "SETC16", "Expected opcode SETC16. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 2, "Two attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 2, "Two attribs expected. Received " + str(len(ins.getAttr()))

        addr   = ins.getAttr()['setc16_reg']
        val    = ins.getAttr()['setc16_value']
        new_state = self.tensixSplRegs.writeCfgReg(addr, val, ins)

        ins.setState(new_state)

        print("In execSETC16: condChkVldUpd = ", new_state.condChkVldUpdVal, " condWriVld = ", new_state.condWriVldUpdVal, " Addr: ", hex(ins.getRelAddr()))

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsetrwc__(self,ins):
        assert ins.getOp() == "SETRWC", "Expected opcode SETRWC. Received " + str(ins.getOp())
        #TODO: Need to reset operands for instructions that set operands. Need a BU function to reset destinations/sources in operands
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        # assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))

        CLEAR_SRCA_DVALID_MASK = 0x1
        CLEAR_SRCB_DVALID_MASK = 0x2
        CLEAR_SRC_DVALID_MASK_MIN = 0x0
        CLEAR_SRC_DVALID_MASK_MAX = 0x3

        src = [] ; dst = [] ; imm = []; vldUpd ={}; bankUpd = {};

        clear_ab_vld = ins.getAttr()["clear_ab_vld"]
        if clear_ab_vld < CLEAR_SRC_DVALID_MASK_MIN or clear_ab_vld > CLEAR_SRC_DVALID_MASK_MAX:
            raise ValueError(f"{ins.getOp}: Invalid clear_ab_vld value: {clear_ab_vld}. Range: {CLEAR_SRC_DVALID_MASK_MIN} to {CLEAR_SRC_DVALID_MASK_MAX}")

        if clear_ab_vld & CLEAR_SRCA_DVALID_MASK:
            src.append(0);  vldUpd[0] = 1; bankUpd[0] = 1; #srcA

        if clear_ab_vld & CLEAR_SRCB_DVALID_MASK:
            src.append(1);  vldUpd[1] = 1; bankUpd[1] = 1; #srcB

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execstallwait__(self,ins):
        assert ins.getOp() == "STALLWAIT", "Expected opcode STALLWAIT. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 2, "Two attribs expected. Received " + str(len(ins.getAttr()))

        srcPipes = [];  dstPipes = []
        pipeLst  = [];  pipeLst2 = []

        if('stall_res' in ins.operands.all):
            if ins.operands.all['stall_res'] & 0x8:
                assert len(self.pipeGrps['UNPACK']) > 0 , "Can't find unpack in engine groups"
                pipeLst  = self.pipeGrps['UNPACK']                  # "stall_unpack",
                for p in pipeLst:
                    if(self.pipes.index(p) not in dstPipes):
                        dstPipes.append(self.pipes.index(p))
            if ins.operands.all['stall_res'] &  0x100:
                assert len(self.pipeGrps['SFPU']) > 0 , "Can't find sfpu in engine groups"
                pipeLst  = self.pipeGrps['SFPU']                    # "stall_sfpu",
                for p in pipeLst:
                    if(self.pipes.index(p) not in dstPipes):
                        dstPipes.append(self.pipes.index(p))
            if ins.operands.all['stall_res'] &  0x40:
                assert len(self.pipeGrps['MATH']) > 0 , "Can't find math in engine groups"
                pipeLst  = self.pipeGrps['MATH']                    # "stall_math",
                for p in pipeLst:
                    if(self.pipes.index(p) not in dstPipes):
                        dstPipes.append(self.pipes.index(p))
            if ins.operands.all['stall_res'] &  0x4:
                assert len(self.pipeGrps['PACK']) > 0 , "Can't find pack in engine groups"
                pipeLst  = self.pipeGrps['PACK']                    # "stall_pack",
                for p in pipeLst:
                    if(self.pipes.index(p) not in dstPipes):
                        dstPipes.append(self.pipes.index(p))
            if ins.operands.all['stall_res'] &  0x1:
                assert len(self.pipeGrps['TDMA']) > 0 , "Can't find tdma in engine groups"
                pipeLst  = self.pipeGrps['TDMA']                    # "stall_compute/tdma",
                for p in pipeLst:
                    if(self.pipes.index(p) not in dstPipes):
                        dstPipes.append(self.pipes.index(p))
            if ins.operands.all['stall_res'] &  0x80:
                assert len(self.pipeGrps['CFG']) > 0 , "Can't find cfg in engine groups"
                pipeLst  = self.pipeGrps['CFG']                    # "stall_cfg",
                for p in pipeLst:
                    if(self.pipes.index(p) not in dstPipes):
                        dstPipes.append(self.pipes.index(p))
            if ins.operands.all['stall_res'] &  0x2:
                assert len(self.pipeGrps['SYNC']) > 0 , "Can't find sync in engine groups"
                pipeLst  = self.pipeGrps['SYNC']                    # "stall_sync",
                for p in pipeLst:
                    if(self.pipes.index(p) not in dstPipes):
                        dstPipes.append(self.pipes.index(p))
            if ins.operands.all['stall_res'] &  0x20:
                assert len(self.pipeGrps['THCON']) > 0 , "Can't find thcon in engine groups"
                pipeLst  = self.pipeGrps['THCON']                    # "stall_thcon",
                for p in pipeLst:
                    if(self.pipes.index(p) not in dstPipes):
                        dstPipes.append(self.pipes.index(p))
            if ins.operands.all['stall_res'] &  0x10:
                assert len(self.pipeGrps['XMOV']) > 0 , "Can't find xmov in engine groups"
                pipeLst  = self.pipeGrps['XMOV']                     # "stall_xmov",
                for p in pipeLst:
                    if(self.pipes.index(p) not in dstPipes):
                        dstPipes.append(self.pipes.index(p))

        if sorted(dstPipes) != sorted(isaFunctions.getT3SimPipesFromStallRes(ins.operands.attributes['stall_res'], self.pipeGrps, self.pipes)):
            msg  = f"- error: {ins.getOp()} exec: mismatch in dst pipes calculations.\n"
            msg += f"  - pipes from isaFunctions.getT3SimPipesFromStallRes: {sorted(isaFunctions.getT3SimPipesFromStallRes(ins.operands.attributes['stall_res'], self.pipeGrps, self.pipes))}\n"
            msg += f"  - pipes calculated within the function:              {sorted(dstPipes)}"

            raise Exception(msg)

        waitRsrcList = ['wait_res_idx_0', 'wait_res_idx_1', 'wait_res_idx_2']
        for w in waitRsrcList:
            if(w in ins.getAttr()):
                waitRsrc = ins.getAttr()[w]
                if(0 == self.args['llkVersionTag'].group):
                    match waitRsrc:
                        # case: 0x00: srcPipes.append(0) nada,
                        case 0x01:
                            assert len(self.pipeGrps['THCON']) > 0 , "Can't find thcon in engine groups"
                            pipeLst2  = self.pipeGrps['THCON']                      # srcPipes.append(8) # thcon,
                        case 0x02|0x03|0x04|0x05|0x06|0x07:
                            assert len(self.pipeGrps['UNPACK']) > 0 , "Can't find unpack in engine groups"
                            pipeLst2  = self.pipeGrps['UNPACK']                     # upk0_idle,# upk0_l1_rds_done,# upk1_idle,# upk1_l1_rds_done,# upk2_idle,# upk2_l1_rds_done,
                        case 0x08|0x09|0x0A|0x0B:
                            assert len(self.pipeGrps['PACK']) > 0 , "Can't find pack in engine groups"
                            pipeLst2  = self.pipeGrps['PACK']                       # pck0_idle,# pck0_dst_rds_done,# pck1_idle,# pck1_dst_rds_done,
                        case 0x0C:
                            assert len(self.pipeGrps['MATH']) > 0 , "Can't find math in engine groups"
                            pipeLst2  = self.pipeGrps['MATH']                       # math,
                        # case 0x0D: srcPipes.append(0) # srcA_clr,
                        # case 0x0E: srcPipes.append(0) # srcB_clr,
                        # case 0x0F: srcPipes.append(0) # srcA_vld,
                        # case 0x10: srcPipes.append(0) # srcB_vld,
                        case 0x11:
                            assert len(self.pipeGrps['XMOV']) > 0 , "Can't find xmov in engine groups"
                            pipeLst2  = self.pipeGrps['XMOV']                        # mover,
                        # case 0x12: srcPipes.append(0) # trisc_mmio_cfg,
                        case 0x13:
                            assert len(self.pipeGrps['SFPU']) > 0 , "Can't find sfpu in engine groups"
                            pipeLst2  = self.pipeGrps['SFPU']                        # sfpu,
                        case 0x14:
                            assert len(self.pipeGrps['CFG']) > 0 , "Can't find cfg in engine groups"
                            pipeLst2  = self.pipeGrps['CFG']                        # cfg_exu,
                else:
                    match waitRsrc:
                        # case: 0x00: srcPipes.append(0) nada,
                        case 0x01:
                            assert len(self.pipeGrps['THCON']) > 0 , "Can't find thcon in engine groups"
                            pipeLst2  = self.pipeGrps['THCON']                      # srcPipes.append(8) # thcon,
                        case 0x02|0x03|0x04|0x05|0x06|0x07:
                            assert len(self.pipeGrps['UNPACK']) > 0 , "Can't find unpack in engine groups"
                            pipeLst2  = self.pipeGrps['UNPACK']                     # upk0_idle,# upk0_l1_rds_done,# upk1_idle,# upk1_l1_rds_done,# upk2_idle,# upk2_l1_rds_done,
                        case 0x08|0x09|0x0A|0x0B:
                            assert len(self.pipeGrps['PACK']) > 0 , "Can't find pack in engine groups"
                            pipeLst2  = self.pipeGrps['PACK']                       # pck0_idle,# pck0_dst_rds_done,# pck1_idle,# pck1_dst_rds_done,
                        case 0x0C:
                            assert len(self.pipeGrps['MATH']) > 0 , "Can't find math in engine groups"
                            pipeLst2  = self.pipeGrps['MATH']                       # math,
                        case 0x0D | 0x0E | 0x0F | 0x10 | 0x11 | 0x12 | 0x13 | 0x15:
                            pass
                        # case 0x0D: srcPipes.append(0) # srcA_clr,
                        # case 0x0E: srcPipes.append(0) # srcB_clr,
                        # case 0x0F: srcPipes.append(0) # srcA_vld,
                        # case 0x10: srcPipes.append(0) # srcB_vld,
                        case 0x14:
                            assert len(self.pipeGrps['XMOV']) > 0 , "Can't find xmov in engine groups"
                            pipeLst2  = self.pipeGrps['XMOV']                        # mover,
                        # case 0x15: srcPipes.append(0) # trisc_mmio_cfg,
                        case 0x16:
                            assert len(self.pipeGrps['SFPU']) > 0 , "Can't find sfpu in engine groups"
                            pipeLst2  = self.pipeGrps['SFPU']                        # sfpu,
                        case 0x17:
                            assert len(self.pipeGrps['CFG']) > 0 , "Can't find cfg in engine groups"
                            pipeLst2  = self.pipeGrps['CFG']                        # cfg_exu,

            for p in pipeLst2:
                if(self.pipes.index(p) not in srcPipes):
                    srcPipes.append(self.pipes.index(p))

        if(self.debug & 0x8):         print("DstPipes = ", dstPipes, "SrcPipes=", srcPipes, f". instruction info: {ins}")

        ins.setDstPipes(dstPipes)
        ins.setSrcPipes(srcPipes)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execwrcfg__(self,ins):
        assert ins.getOp() == "WRCFG", "Expected opcode WRCFG. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        assert len(ins.getAttr()) == 3, "Three attribs expected. Received " + str(len(ins.getAttr()))

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execzerosrc__(self,ins):
        assert ins.getOp() == "ZEROSRC", "Expected opcode ZEROSRC. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        #TODO: Remove llkVersionTag based code once LLK stabilizes
        #TODO: Check what do we need for WH and BH.
        if(0 == self.args['llkVersionTag'].group):
            assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))
        else:
            assert len(ins.getAttr()) == 7, "Seven attribs expected. Received " + str(len(ins.getAttr()))

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execzeroacc__(self,ins):
        assert ins.getOp() == "ZEROACC", "Expected opcode ZEROACC. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 5, "Five attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execclrdvalid__(self,ins):
        assert ins.getOp() == "CLEARDVALID", "Expected opcode CLEARDVALID. Received " + str(ins.getOp())
        # TODO: Check why certain clear dvalids have destinations and not sources and others have it opposite, i.e., sources and not destinations
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected " + str(dir(ins.getOperands()))
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected " + str(dir(ins.getOperands()))
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))

        src = [] ; dst = [] ; imm = []; vldUpd ={}; bankUpd = {};

        match ins.getAttr()["cleardvalid"] :
            case 1: src.append(0);  vldUpd[0] = 1; bankUpd[0] = 1; #srcA
            case 2: src.append(1);  vldUpd[1] = 1; bankUpd[1] = 1; #srcB
            case 3:
                    src.append(0);  vldUpd[0] = 1; bankUpd[0] = 1; #srcA
                    src.append(1);  vldUpd[1] = 1; bankUpd[1] = 1; #srcB

        if (ins.getAttr()["cleardvalid_S"]) :
            src.append(2);  vldUpd[2] = 1; bankUpd[2] = 1; #srcS

        # Unpacker, Math and SFPU all set Dst Valid to 1
        # Packer sets Dst Valid to 0
        match ins.getAttr()["dest_pulse_last"] :
            case 1:
                dst.append(3);  vldUpd[3] = 1; bankUpd[3] = 1;  # Unpack-to-dest
                if(self.debug & 0x8):   print("Set Dest valid for Unpack-to-dest")
            case 2:
                dst.append(3);  vldUpd[3] = 1; bankUpd[3] = 1;  # Math
                if(self.debug & 0x8):   print("Set Dest valid for math")
            case 4:
                dst.append(3);  vldUpd[3] = 1; bankUpd[3] = 1;  # SFPU
                if(self.debug & 0x8):   print("Set Dest valid for sfpu");
            case 8:
                src.append(3);  vldUpd[3] = 1; bankUpd[3] = 1;  # Packer
                if(self.debug & 0x8):   print("Clr Dest valid for packer");

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execmov__(self,ins):
        opList = ["MOVA2D", "MOVB2D", "MOVD2A", "MOVD2B", "MOVB2A"]
        assert ins.getOp() in opList, "Expected MOV* opcode. Received " + str(ins.getOp())
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"

        src = [] ; dst = [] ; imm = []; vldUpd ={}; bankUpd = {};

        match ins.getOp():
            case "MOVA2D":
                assert len(ins.getAttr()) == 5, "Five attribs expected. Received " + str(len(ins.getAttr()))
                dst.append(3);  vldUpd[3] = 0; bankUpd[3] = 0; #dst0
                src.append(0);  vldUpd[0] = 0; bankUpd[0] = 0; #srcA
            case "MOVB2D":
                assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))
                dst.append(3);  vldUpd[3] = 0; bankUpd[3] = 0; #dst0
                src.append(1);  vldUpd[1] = 0; bankUpd[1] = 0; #srcB
            case "MOVD2A":
                assert len(ins.getAttr()) == 5, "Five attribs expected. Received " + str(len(ins.getAttr()))
                dst.append(0);  vldUpd[0] = 0; bankUpd[0] = 0; #srcA
                src.append(3);  vldUpd[3] = 0; bankUpd[3] = 0; #dst0
            case "MOVD2B":
                if(0 == self.args['llkVersionTag'].group):
                    assert len(ins.getAttr()) == 5, "Five attribs expected. Received " + str(len(ins.getAttr()))
                else:
                    assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))
                dst.append(1);  vldUpd[1] = 0; bankUpd[1] = 0; #srcA
                src.append(3);  vldUpd[3] = 0; bankUpd[3] = 0; #dst0
            case "MOVB2A":
                assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))
                dst.append(0);  vldUpd[0] = 0; bankUpd[0] = 0; #srcA
                src.append(1);  vldUpd[1] = 0; bankUpd[1] = 0; #srcB

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __exectrnsp__(self,ins):
        opList = ["TRNSPSRCA", "TRNSPSRCB"]
        assert ins.getOp() in opList, "Expected TRNSPSRCA/B opcode. Received " + str(ins.getOp())
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        assert len(ins.getAttr()) == 0, "Zero attribs expected. Received " + str(len(ins.getAttr()))

        src = [] ; dst = [] ; imm = []; vldUpd ={}; bankUpd = {};

        match ins.getOp():
            case "TRNSPSRCA":
                src.append(0);  vldUpd[0] = 0; bankUpd[0] = 0; #srcA
            case "TRNSPSRCB":
                src.append(1);  vldUpd[1] = 0; bankUpd[1] = 0; #srcB

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setImm(imm)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execdsttilefacerowi__(self,ins):
        assert ins.getOp() == "SET_DST_TILE_FACE_ROW_IDX" or ins.getOp() == "INC_DST_TILE_FACE_ROW_IDX" , "Expected opcode SET/INC_DST_TILE_FACE_ROW_IDX. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        assert len(ins.getAttr()) == 3, "Three attribs expected. Received " + str(len(ins.getAttr()))

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsrctilefacerowi__(self,ins):
        assert ins.getOp() == "SET_SRC_TILE_FACE_ROW_IDX" or ins.getOp() == "INC_SRC_TILE_FACE_ROW_IDX" , "Expected opcode SET/INC_SRC_TILE_FACE_ROW_IDX. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        assert len(ins.getAttr()) == 3, "Three attribs expected. Received " + str(len(ins.getAttr()))

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execreplay__(self,ins, cycle):
        assert ins.getOp() == "REPLAY" , "Expected opcode REPLAY. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 6, "Six attribs expected. Received " + str(len(ins.getAttr()))

        # assert False, "Replay not implemented"
        print(f"TFunctional(Update) Cycle:{cycle} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{ins.getThread()} ", end='')
        ins.printInstr(ins.getThread())
        return [ins.getAttr()['load_mode'], ins.getAttr()['execute_while_loading'], ins.getAttr()['start_idx'], ins.getAttr()['len']]

        # nextRelAddr = ins.getRelAddr() + 4
        # return nextRelAddr

    def __execrmwcib__(self,ins):
        opList = ["RMWCIB0", "RMWCIB1", "RMWCIB2", "RMWCIB3"]
        assert ins.getOp() in opList , "Expected opcode RMWCIB. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 3, "Three attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 3, "Three attribs expected. Received " + str(len(ins.getAttr()))

        addr = ins.getAttr()['CfgRegAddr']
        val  = ins.getAttr()['Data'] & ins.getAttr()['Mask']

        if(self.debug & 0x8):
            print(f'exec{ins.getOp()}: Addr={hex(ins.getAttr()["CfgRegAddr"])}, Mask={hex(ins.getAttr()["Mask"])}, Val={hex(ins.getAttr()["Data"])} ' )

        oldVal  = self.tensixSplRegs.__readReg__(addr, "cfg")
        if oldVal == -1:    oldVal = 0

        match ins.getOp():
            case "RMWCIB0":   val  = oldVal | (val << 0x00000000)
            case "RMWCIB1":   val  = oldVal | (val << 8)
            case "RMWCIB2":   val  = oldVal | (val << 16)
            case "RMWCIB3":   val  = oldVal | (val << 24)
            case _:           assert False, "Expected opcode RMWCIB. Received " + str(ins.getOp())

        new_state = self.tensixSplRegs.writeCfgReg(addr, val, ins)

        ins.setState(new_state)

        if self.debug & 0x8:
            print("In execRMWCIB: condChkVldUpd = ", new_state.condChkVldUpdVal, " condWriVld = ", new_state.condWriVldUpdVal, " Addr: ", hex(ins.getRelAddr()))

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsfpload__(self,ins):
        assert ins.getOp() == "SFPLOAD" , "Expected opcode REPLAY. Received " + str(ins.getOp())
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 5,                   f"Five attribs expected. Received " + str(len(ins.getAttr()))
                assert hasattr(ins, 'operands'),                  f"Expected decoded instruction to have operands"
                assert hasattr(ins.operands, 'all'),              f"Expected decoded instruction operands to have attribute 'all'"
                assert hasattr(ins.operands, 'attributes'),       f"Expected decoded instruction operands to have attribute 'attributes'"
                assert "dest_reg_addr"  in ins.operands.attributes.keys(), f"Could not find destination register address from decoded operands. Given operands: {ins.operands.attributes}"
            case _:
                assert len(ins.getAttr()) == 5, "Five attribs expected. Received " + str(len(ins.getAttr()))

        #   Registers      | Instructions
        #                  | SFPLOAD     | SFPLOADI      | SFPARECIP | SFPSTORE
        #    Source(s)     | Dest/SRCS*  | - (imm value) | LREG[0]   | LREG[1]
        #    Destination(s)| LREGs       | LREGs         | LREG[1]   | Dest/SRCS*

        # *: If the bit 10 (MSB, litte endian order) is 1, then source is SRCS. Else it is Dest.
        # named registers: 0: srcA, 1: srcB, 2: srcS, 3: Dest.

        src_reg_id = 2 if ((ins.operands.attributes['dest_reg_addr'] >> 10) & 0b1) else 3
        update_flag = 1 if ins.operands.attributes['done'] else 0

        src = [] ; dst = [] ; imm = []; vldUpd ={}; bankUpd = {}
        src.append(src_reg_id)
        vldUpd[src_reg_id] = update_flag
        bankUpd[src_reg_id] = update_flag

        ins.setSrcInt(src)
        ins.setImm(imm)
        ins.setDstInt(dst)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsfploadi__(self,ins):
        assert ins.getOp() == "SFPLOADI" , "Expected opcode SFPLOADI. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 3,                   f"Three attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 3,                   f"Three attribs expected. Received " + str(len(ins.getAttr()))

        #   Registers      | Instructions
        #                  | SFPLOAD     | SFPLOADI      | SFPARECIP | SFPSTORE
        #    Source(s)     | Dest/SRCS*  | - (imm value) | LREG[0]   | LREG[1]
        #    Destination(s)| LREGs       | LREGs         | LREG[1]   | Dest/SRCS*

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def _validate_sfpu_attribs_and_advance(self, ins, expected_attribs: list[str], context: str = "SFPU"):
        """Helper method to validate SFPU instruction attributes and advance the PC.

        This helper performs **only** structural/semantic validation of the
        instruction's operand attributes and then advances the program counter.
        Unlike helpers such as ``_compute_num_datums_from_row_cnt_enc``, it does
        not compute or return any datum-count value; it always returns the next
        instruction address.

        Args:
            ins: The instruction to validate.
            expected_attribs: List of expected attribute names.
            context: Context string for error messages (e.g., function name).

        Returns:
            The next relative address, i.e., ``ins.getRelAddr() + 4``.

        Raises:
            AssertionError: If the instruction is missing an ``operands``
                attribute or if ``ins.operands`` is ``None``.
            AssertionError: If ``ins.operands`` is missing an ``attributes``
                attribute or if ``ins.operands.attributes`` is ``None``.
            AssertionError: If ``ins.operands.attributes`` does not provide a
                callable ``keys`` method.
            AssertionError: If the actual attribute names on
                ``ins.operands.attributes`` do not match ``expected_attribs``.
        """
        # Ensure the instruction has the expected operand/attribute structure.
        if not hasattr(ins, "operands") or ins.operands is None:
            raise AssertionError(
                f"@{context}: instruction object {type(ins).__name__} is missing required 'operands' attribute."
            )
        operands = ins.operands
        if not hasattr(operands, "attributes") or operands.attributes is None:
            raise AssertionError(
                f"@{context}: operands object {type(operands).__name__} is missing required 'attributes' attribute."
            )
        attributes = operands.attributes
        if not hasattr(attributes, "keys") or not callable(getattr(attributes, "keys", None)):
            raise AssertionError(
                f"@{context}: operands.attributes of type {type(attributes).__name__} does not support 'keys()'."
            )
        actual_attribs = list(attributes.keys())
        assert sorted(expected_attribs) == sorted(actual_attribs), (
            f"@{context}: attributes mismatch. expected attributes: {sorted(expected_attribs)}, "
            f"received: {sorted(actual_attribs)}."
        )
        return ins.getRelAddr() + 4

    def __execsfpconfig__(self,ins):
        assert ins.getOp() == "SFPCONFIG" , "Expected opcode SFPCONFIG. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 3,                   f"Three attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert len(ins.getAttr()) == 3,                   f"Three attribs expected. Received " + str(len(ins.getAttr()))

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execsfpstore__(self,ins):
        assert ins.getOp() == "SFPSTORE" , "Expected opcode REPLAY. Received " + str(ins.getOp())
        # assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        # assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 5,                   f"Five attribs expected. Received " + str(len(ins.getAttr()))
                assert hasattr(ins, 'operands'),                  f"Expected decoded instruction to have operands"
                assert hasattr(ins.operands, 'all'),              f"Expected decoded instruction operands to have attribute 'all'"
                assert hasattr(ins.operands, 'attributes'),       f"Expected decoded instruction operands to have attribute 'attributes'"
                assert "dest_reg_addr" in ins.operands.attributes.keys(), f"Could not find destination register address from decoded operands. Given operands: {ins.operands.attributes}"
                assert "done" in ins.operands.attributes.keys(),          f"Could not find done bit/field from decoded operands. Given operands: {ins.operands.attributes}"
            case _:
                assert len(ins.getAttr()) == 5, "Five attribs expected. Received " + str(len(ins.getAttr()))

        #   Registers      | Instructions
        #                  | SFPLOAD     | SFPLOADI      | SFPARECIP | SFPSTORE
        #    Source(s)     | Dest/SRCS*  | - (imm value) | LREG[0]   | LREG[1]
        #    Destination(s)| LREGs       | LREGs         | LREG[1]   | Dest/SRCS*

        # *: If the bit 10 (MSB, litte endian order) is 1, then source is SRCS. Else it is Dest.
        # named registers: 0: srcA, 1: srcB, 2: srcS, 3: Dest.

        dst_reg_id = 2 if ((ins.operands.attributes['dest_reg_addr'] >> 10) & 0b1) else 3
        update_flag = 1 if ins.operands.attributes['done'] else 0

        src = [] ; dst = [] ; imm = []; vldUpd ={}; bankUpd = {}
        dst.append(dst_reg_id)
        vldUpd[dst_reg_id] = update_flag
        bankUpd[dst_reg_id] = update_flag

        ins.setSrcInt(src)
        ins.setImm(imm)
        ins.setDstInt(dst)
        ins.setVldUpdMask(vldUpd)
        ins.setBankUpdMask(bankUpd)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def __execSFPU_MATHI12__(self,ins):
        allowedMnemonics = [
            'SFPABS', 'SFPAND', 'SFPARECIP', 'SFPCOMPC', 'SFPDIVP2',
            'SFPENCC', 'SFPEXEXP', 'SFPEXMAN', 'SFPGT', 'SFPIADD',
            'SFPLE', 'SFPLZ', 'SFPMOV', 'SFPNOT', 'SFPOR', 'SFPPOPC',
            'SFPPUSHC', 'SFPSETCC', 'SFPSETEXP', 'SFPSETMAN',
            'SFPSETSGN', 'SFPSHFT', 'SFPTRANSP', 'SFPXOR']

        if ins.getOp() not in allowedMnemonics:
            raise Exception(f"- error: given mnemonic {ins.getOp()} is not part of mnemonics with operands with tag SFPU_MATHI12")

        # For tags in groups 0 and 1, legacy arg layout applies
        if (self.args['llkVersionTag'].group in [0,1]):
            return self._validate_sfpu_attribs_and_advance(ins, ['instr_mod1', 'lreg_dest', 'lreg_c', 'imm12_math'], '__execSFPU_MATHI12__')

        # Group 2 (nov17): per-mnemonic argument layouts
        op = ins.getOp()
        if op in ['SFPCOMPC', 'SFPTRANSP']:
            return self._validate_sfpu_attribs_and_advance(ins, [], '__execSFPU_MATHI12__')
        if op in ['SFPPOPC', 'SFPPUSHC']:
            return self._validate_sfpu_attribs_and_advance(ins, ['instr_mod1'], '__execSFPU_MATHI12__')
        if op == 'SFPENCC':
            return self._validate_sfpu_attribs_and_advance(ins, ['instr_mod1', 'imm12_math'], '__execSFPU_MATHI12__')
        if op == 'SFPSETCC':
            return self._validate_sfpu_attribs_and_advance(ins, ['instr_mod1', 'lreg_c', 'imm12_math'], '__execSFPU_MATHI12__')
        if op in ['SFPDIVP2', 'SFPGT', 'SFPIADD', 'SFPLE', 'SFPSETEXP', 'SFPSETMAN', 'SFPSETSGN', 'SFPSHFT']:
            return self._validate_sfpu_attribs_and_advance(ins, ['instr_mod1', 'lreg_dest', 'lreg_c', 'imm12_math'], '__execSFPU_MATHI12__')
        if op in ['SFPABS', 'SFPEXEXP', 'SFPEXMAN', 'SFPLZ', 'SFPMOV']:
            return self._validate_sfpu_attribs_and_advance(ins, ['instr_mod1', 'lreg_dest', 'lreg_c'], '__execSFPU_MATHI12__')
        if op in ['SFPAND', 'SFPNOT', 'SFPOR', 'SFPXOR']:
            return self._validate_sfpu_attribs_and_advance(ins, ['lreg_dest', 'lreg_c'], '__execSFPU_MATHI12__')

        print(f"WARNING: in function __execSFPU_MATHI12__, advancing pc without any checks. instruction: {ins}")
        # Fallback
        return ins.getRelAddr() + 4

    def __execSFPU_MATH__(self, ins):
        allowedMnemonics = ['SFPADD', 'SFPMAD', 'SFPMUL', 'SFPMUL24']

        if ins.getOp() not in allowedMnemonics:
            raise Exception(f"- error: given mnemonic {ins.getOp()} is not part of mnemonics with operands with tag SFPU_MATH")

        # For tags in groups 0 and 1, legacy arg layout applies
        if (self.args['llkVersionTag'].group in [0,1]):
            return self._validate_sfpu_attribs_and_advance(ins, ['instr_mod1', 'lreg_dest', 'lreg_src_c', 'lreg_src_b', 'lreg_src_a'], '__execSFPU_MATH__')

        # Group 2 (nov17): per-mnemonic argument layouts.
        # NOTE: nov17 format intentionally renamed the logical register source
        # attributes from lreg_src_a/b/c to lreg_a/b/c, while keeping lreg_dest.
        # The decoder for nov17 emits these shorter names, so we must use them
        # here to stay aligned with that ISA/version. Groups 0–1 retain the
        # legacy lreg_src_* naming for backward compatibility.
        op = ins.getOp()
        if op in ['SFPADD', 'SFPMAD', 'SFPMUL']:
            return self._validate_sfpu_attribs_and_advance(ins, ['instr_mod1', 'lreg_a', 'lreg_b', 'lreg_c', 'lreg_dest'], '__execSFPU_MATH__')
        if op == 'SFPMUL24':
            return self._validate_sfpu_attribs_and_advance(ins, ['instr_mod1', 'lreg_a', 'lreg_b', 'lreg_dest'], '__execSFPU_MATH__')

        print(f"WARNING: in function __execSFPU_MATH__, advancing pc without any checks. instruction: {ins}")
        # Fallback
        return ins.getRelAddr() + 4

    def __execsfpnop__(self, ins):
        assert "SFPNOP" == ins.getOp(), "Expected opcode SFPNOP. Received " + str(ins.getOp())
        expectedAttribs = ['dest_done', 'srcs_rd_done', 'srcs_wr_done']

        assert len(ins.getAttr()) == len(expectedAttribs), f"@__execsfpnop__: {len(expectedAttribs)} attribs expected. Received {len(ins.getAttr())}, instruction: {ins}."
        assert sorted(expectedAttribs) == sorted(list(ins.operands.attributes.keys())), f"@__execsfpnop__: attributes mismatch. expected attributes: {sorted(expectedAttribs)}, received: {sorted(list(ins.operands.attributes.keys()))}."

        src = []
        dst = []
        vldUpd = {}
        bankUpd = {}

        if ins.getAttr()['dest_done']:
            src.append(isaFunctions.regIndex.dst)

        # if ins.getAttr()['srcs_rd_done']:
        #     src.append(isaFunctions.regIndex.srcS)

        if ins.getAttr()['srcs_wr_done']:
            dst.append(isaFunctions.regIndex.srcS)

        for reg_id in src:
            vldUpd[reg_id] = 1
            bankUpd[reg_id] = 1

        for reg_id in dst:
            vldUpd[reg_id] = 1
            bankUpd[reg_id] = 1

        ins.setSrcInt(src)
        ins.setDstInt(dst)
        ins.setBankUpdMask(bankUpd)
        ins.setVldUpdMask(vldUpd)

        nextRelAddr = ins.getRelAddr() + 4
        return nextRelAddr

    def execTTIns(self,ins, cycle):
        nextAddr = -1
        match ins.getOp():
            case "UNPACR0_TILE_INC":    nextAddr            = self.__execunpacrti__(ins)
            case "UNPACR1_TILE_INC":    nextAddr            = self.__execunpacrti__(ins)
            case "UNPACR2_TILE_INC":    nextAddr            = self.__execunpacrti__(ins)
            case "UNPACR_DEST_TILE_INC":nextAddr            = self.__execunpacrti__(ins)
            case "UNPACR0_FACE_INC":    nextAddr            = self.__execunpacrti__(ins)
            case "UNPACR1_FACE_INC":    nextAddr            = self.__execunpacrti__(ins)
            case "UNPACR2_FACE_INC":    nextAddr            = self.__execunpacrti__(ins)
            case "UNPACR_DEST_FACE_INC":nextAddr            = self.__execunpacrti__(ins)
            case "UNPACR0_ROW_INC":     nextAddr            = self.__execunpacrti__(ins)
            case "UNPACR1_ROW_INC":     nextAddr            = self.__execunpacrti__(ins)
            case "UNPACR2_ROW_INC":     nextAddr            = self.__execunpacrti__(ins)
            case "UNPACR_DEST_ROW_INC": nextAddr            = self.__execunpacrti__(ins)
            case "UNPACR0_TILE":        nextAddr            = self.__execunpacrt__(ins)
            case "UNPACR1_TILE":        nextAddr            = self.__execunpacrt__(ins)
            case "UNPACR2_TILE":        nextAddr            = self.__execunpacrt__(ins)
            case "UNPACR_DEST_TILE":    nextAddr            = self.__execunpacrt__(ins)
            case "UNPACR0_FACE":        nextAddr            = self.__execunpacrt__(ins)
            case "UNPACR1_FACE":        nextAddr            = self.__execunpacrt__(ins)
            case "UNPACR2_FACE":        nextAddr            = self.__execunpacrt__(ins)
            case "UNPACR_DEST_FACE":    nextAddr            = self.__execunpacrt__(ins)
            case "UNPACR0_ROW":         nextAddr            = self.__execunpacrt__(ins)
            case "UNPACR1_ROW":         nextAddr            = self.__execunpacrt__(ins)
            case "UNPACR2_ROW":         nextAddr            = self.__execunpacrt__(ins)
            case "UNPACR_DEST_ROW":     nextAddr            = self.__execunpacrt__(ins)
            case "UNPACR0_STRIDE":      nextAddr            = self.__execunpacrs__(ins)
            case "UNPACR1_STRIDE":      nextAddr            = self.__execunpacrs__(ins)
            case "UNPACR2_STRIDE":      nextAddr            = self.__execunpacrs__(ins)
            case "UNPACR_DEST_STRIDE":  nextAddr            = self.__execunpacrs__(ins)
            case "UNPACR":              nextAddr            = self.__execunpacr__(ins)
            case "UNPACR_TILE_MISC":    nextAddr            = self.__execunpacrtm__(ins)
            case "UNPACR_TILIZE":       nextAddr            = self.__execunpacrtz__(ins)
            case "UNPACR_NOP":          nextAddr            = self.__execunpacrnop__(ins)
            case "PACR0_TILE":          nextAddr            = self.__execpacrti__(ins)
            case "PACR1_TILE":          nextAddr            = self.__execpacrti__(ins)
            case "PACR2_TILE":          nextAddr            = self.__execpacrti__(ins)
            case "PACR0_FACE":          nextAddr            = self.__execpacrti__(ins)
            case "PACR1_FACE":          nextAddr            = self.__execpacrti__(ins)
            case "PACR2_FACE":          nextAddr            = self.__execpacrti__(ins)
            case "PACR0_ROW":           nextAddr            = self.__execpacrti__(ins)
            case "PACR1_ROW":           nextAddr            = self.__execpacrti__(ins)
            case "PACR2_ROW":           nextAddr            = self.__execpacrti__(ins)
            case "PACR0_TILE_INC":      nextAddr            = self.__execpacrti__(ins)
            case "PACR1_TILE_INC":      nextAddr            = self.__execpacrti__(ins)
            case "PACR2_TILE_INC":      nextAddr            = self.__execpacrti__(ins)
            case "PACR0_FACE_INC":      nextAddr            = self.__execpacrti__(ins)
            case "PACR1_FACE_INC":      nextAddr            = self.__execpacrti__(ins)
            case "PACR2_FACE_INC":      nextAddr            = self.__execpacrti__(ins)
            case "PACR0_ROW_INC":       nextAddr            = self.__execpacrti__(ins)
            case "PACR1_ROW_INC":       nextAddr            = self.__execpacrti__(ins)
            case "PACR2_ROW_INC":       nextAddr            = self.__execpacrti__(ins)
            case "PACR_STRIDE":         nextAddr            = self.__execpacr_stride__(ins)
            case "PACR_UNTILIZE":       nextAddr            = self.__execpacr_untilize__(ins)
            case "GAPOOL":              nextAddr            = self.__execgpool__(ins)
            case "GMPOOL":              nextAddr            = self.__execgpool__(ins)
            case "ELWADD":              nextAddr            = self.__execelwadd__(ins)
            case "ELWSUB":              nextAddr            = self.__execelwsub__(ins)
            case "ELWMUL":              nextAddr            = self.__execelwmul__(ins)
            case "MVMUL":               nextAddr            = self.__execmvmul__(ins)
            case "MVMULDI":             nextAddr            = self.__execmvmuldi__(ins)
            case "ATGETM":              nextAddr            = self.__execatgetm__(ins)
            case "ATRELM":              nextAddr            = self.__execatrelm__(ins)
            case "DMANOP":              nextAddr            = self.__execdmanop__(ins)
            case "NOP":                 nextAddr            = self.__execnop__(ins)
            case "SEMINIT":             nextAddr            = self.__execseminit__(ins)
            case "SEMGET":              nextAddr            = self.__execsemget__(ins)
            case "SEMPOST":             nextAddr            = self.__execsempost__(ins)
            case "SEMWAIT":             nextAddr            = self.__execsemwait__(ins)
            case "SETADCXX":            nextAddr            = self.__execsetadcxx__(ins)
            case "SETADCXY":            nextAddr            = self.__execsetadcxy__(ins)
            case "SETADCZW":            nextAddr            = self.__execsetadczw__(ins)
            case "SETC16":              nextAddr            = self.__execsetc16__(ins)
            case "SETRWC":              nextAddr            = self.__execsetrwc__(ins)
            case "STALLWAIT":           nextAddr            = self.__execstallwait__(ins)
            case "WRCFG":               nextAddr            = self.__execwrcfg__(ins)
            case "ZEROSRC":             nextAddr            = self.__execzerosrc__(ins)
            case "ZEROACC":             nextAddr            = self.__execzeroacc__(ins)
            case "CLEARDVALID":         nextAddr            = self.__execclrdvalid__(ins)
            case "MOVA2D":              nextAddr            = self.__execmov__(ins)
            case "MOVB2D":              nextAddr            = self.__execmov__(ins)
            case "MOVD2A":              nextAddr            = self.__execmov__(ins)
            case "MOVD2B":              nextAddr            = self.__execmov__(ins)
            case "MOVB2A":              nextAddr            = self.__execmov__(ins)
            case "TRNSPSRCA":           nextAddr            = self.__exectrnsp__(ins)
            case "TRNSPSRCB":           nextAddr            = self.__exectrnsp__(ins)
            case "RMWCIB0"|"RMWCIB1"|"RMWCIB2"|"RMWCIB3":
                                        nextAddr            = self.__execrmwcib__(ins)
            case "SFPLOAD":             nextAddr            = self.__execsfpload__(ins)
            case "SFPLOADI":            nextAddr            = self.__execsfploadi__(ins)
            case "SFPCONFIG":           nextAddr            = self.__execsfpconfig__(ins)
            case "SFPSTORE":            nextAddr            = self.__execsfpstore__(ins)
            case "SFPNOP":              nextAddr            = self.__execsfpnop__(ins)
            case "SET_DST_TILE_FACE_ROW_IDX":
                                        nextAddr            = self.__execdsttilefacerowi__(ins)
            case "INC_DST_TILE_FACE_ROW_IDX":
                                        nextAddr            = self.__execdsttilefacerowi__(ins)
            case "SET_SRC_TILE_FACE_ROW_IDX":
                                        nextAddr            = self.__execsrctilefacerowi__(ins)
            case "INC_SRC_TILE_FACE_ROW_IDX":
                                        nextAddr            = self.__execsrctilefacerowi__(ins)
            case "REPLAY":              nextAddr            = self.__execreplay__(ins, cycle)
            case "SFPABS" | "SFPAND" | "SFPARECIP" | "SFPCOMPC" | "SFPDIVP2" | "SFPENCC" | "SFPEXEXP" | "SFPEXMAN" | "SFPGT" | "SFPIADD" | "SFPLE" | "SFPLZ" | "SFPMOV" | "SFPNOT" | "SFPOR" | "SFPPOPC" | "SFPPUSHC" | "SFPSETCC" | "SFPSETEXP" | "SFPSETMAN" | "SFPSETSGN" | "SFPSHFT" | "SFPTRANSP" | "SFPXOR":
                nextAddr = self.__execSFPU_MATHI12__(ins)
            case "SFPADD" | "SFPMAD" | "SFPMUL" | "SFPMUL24":
                nextAddr = self.__execSFPU_MATH__(ins)
            # case "MOP":                 nextAddr            = self.__execmop__(ins)
            case _:
                print("WARNING: Unsupported TT Instruction ", ins.getOp())
                nextAddr = ins.getRelAddr() + 4
                ins.printInstr(ins.getThread())

        assert nextAddr != -1 , "Error in execution"
        if (self.debug & 0x1):
            print(f"TFunctional Cycle:{cycle} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{ins.getThread()} ", end='')
            ins.printInstr(ins.getThread())
        return nextAddr

    def __printReg__(self, type='mop'):
        assert type in self.tensixSplRegs.regTypes, "RegType not supported:" + type + "Supported RegTypes" +  self.regTypes
        regCnt = 0
        print("RegList. :", end='')
        while (regCnt < len(self.tensixSplRegs.reg[type])):
            if(self.tensixSplRegs.reg[type][regCnt] != -1):
                print (type, "[", hex(regCnt) ,"]=", hex(self.tensixSplRegs.reg[type][regCnt]),",", sep='', end='' )
            regCnt += 1
        print()

    def getMopCfgAddr(self):
        return self.tensixSplRegs.regTypeDict['mop'][1]

    #HW model of MOP
    def buildInsFromMop(self, ins):
        [mopType, bank] = self.__execmop__(ins)

        if(mopType == 1): # Double Loop
            if(bank == 1):       assert False, "dual bank MOP CFG not handled"
            instr = []
            NOP = "0x2000000"               #TODO: Remove hard-coding
            x = {
                'bank0_loop0_len'           :self.tensixSplRegs.__readReg__(64*ins.getThread() + 0,'mop'),
                'bank0_loop1_len'           :self.tensixSplRegs.__readReg__(64*ins.getThread() + 1,'mop'),
                'bank0_loop_start_instr0'   :self.tensixSplRegs.__readReg__(64*ins.getThread() + 2,'mop'),
                'bank0_loop_end_instr0'     :self.tensixSplRegs.__readReg__(64*ins.getThread() + 3,'mop'),
                'bank0_loop_end_instr1'     :self.tensixSplRegs.__readReg__(64*ins.getThread() + 4,'mop'),
                'bank0_loop_instr0'         :self.tensixSplRegs.__readReg__(64*ins.getThread() + 5,'mop'),
                'bank0_loop_instr1'         :self.tensixSplRegs.__readReg__(64*ins.getThread() + 6,'mop'),
                'bank0_loop0_last_instr'    :self.tensixSplRegs.__readReg__(64*ins.getThread() + 7,'mop'),
                'bank0_loop1_last_instr'    :self.tensixSplRegs.__readReg__(64*ins.getThread() + 8,'mop'),
                'mop_sw_ctrl'               :self.tensixSplRegs.__readReg__(64*ins.getThread() + 9,'mop')      #TODO: Need to handle
            }
            if(self.debug & 0x8):
                for t in x:
                    print(t, hex(x[t]))

            i = 0

            while(i < x['bank0_loop0_len']):
                if(self.debug & 0x8):    print("Outer Loop")
                if(str(hex(x['bank0_loop_start_instr0'])) != NOP):
                    if(self.debug & 0x8):    print("\tOuter Loop Start")
                    instr.append(x['bank0_loop_start_instr0'])
                j = 0
                while(j < x['bank0_loop1_len']):
                    if(self.debug & 0x8):    print("\tInner Loop")
                    if(j == (x['bank0_loop1_len'] - 1) and i == (x['bank0_loop0_len'] - 1)):
                        if(str(hex(x['bank0_loop_instr0'])) != NOP) and (str(hex(x['bank0_loop_instr1'])) != NOP) and (str(hex(x['bank0_loop0_last_instr'])) != NOP):
                            instr.append(x['bank0_loop_instr0'])
                            instr.append(x['bank0_loop0_last_instr'])
                            if(self.debug & 0x8):    print("\t\tInner Loop Op0")
                            if(self.debug & 0x8):    print("\t\tInner Loop Op1 Last")
                        elif(str(hex(x['bank0_loop_instr0'])) != NOP) and (str(hex(x['bank0_loop0_last_instr'])) != NOP):
                            instr.append(x['bank0_loop0_last_instr'])
                            if(self.debug & 0x8):    print("\t\tInner Loop Op0 Last")
                        else:
                            assert str(hex(x['bank0_loop_instr0'])) != NOP ,  "Shouldnt there be at least one instruction in the inner loop?"
                    elif(j == (x['bank0_loop1_len'] - 1)):
                        if(str(hex(x['bank0_loop_instr0'])) != NOP) and (str(hex(x['bank0_loop_instr1'])) != NOP) and (str(hex(x['bank0_loop1_last_instr'])) != NOP):
                            instr.append(x['bank0_loop_instr0'])
                            instr.append(x['bank0_loop1_last_instr'])
                            if(self.debug & 0x8):    print("\t\tInner Loop Op0")
                            if(self.debug & 0x8):    print("\t\tInner Loop Op1 Last")
                        elif(str(hex(x['bank0_loop_instr0'])) != NOP) and (str(hex(x['bank0_loop1_last_instr'])) != NOP):
                            instr.append(x['bank0_loop1_last_instr'])
                            if(self.debug & 0x8):    print("\t\tInner Loop Op0 Last")
                        else:
                            assert str(hex(x['bank0_loop_instr0'])) != NOP ,  "Shouldnt there be at least one instruction in the inner loop?"
                    else:
                        if(str(hex(x['bank0_loop_instr0'])) != NOP) and (str(hex(x['bank0_loop_instr1'])) != NOP):
                            instr.append(x['bank0_loop_instr0'])
                            instr.append(x['bank0_loop_instr1'])
                            if(self.debug & 0x8):    print("\t\tInner Loop Op0")
                            if(self.debug & 0x8):    print("\t\tInner Loop Op1")
                        elif(str(hex(x['bank0_loop_instr0'])) != NOP):
                            instr.append(x['bank0_loop_instr0'])
                            if(self.debug & 0x8):    print("\t\tInner Loop Op0")
                        else:
                            assert str(hex(x['bank0_loop_instr0'])) != NOP ,  "Shouldnt there be at least one instruction in the inner loop?"
                    j += 1

                i += 1

                if(i == (x['bank0_loop0_len'] ) and j == (x['bank0_loop1_len']) ):
                    if(str(hex(x['bank0_loop_end_instr0'])) != NOP):
                        instr.append(x['bank0_loop_end_instr0'])
                        if(self.debug & 0x8):    print("\tOuter Loop Op0 End")
                    if(str(hex(x['bank0_loop_end_instr1'])) != NOP):
                        instr.append(x['bank0_loop_end_instr1'])
                        if(self.debug & 0x8):    print("\tOuter Loop Op1 End")
            return instr
        else:
            assert False, "Unhandled Unpack Loop"

    def __execmop__(self,ins):
        assert ins.getOp() == "MOP" , "Expected opcode MOP. Received " + str(ins.getOp())
        assert "destinations" not in dir(ins.getOperands()) , "Zero Dst expected"
        assert "sources" not in dir(ins.getOperands()) , "Zero Src expected"
        assert "immediates" not in dir(ins.getOperands()) , "Zero Imm expected"
        match ins.kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert len(ins.getAttr()) == 4, "Four attribs expected. Received " + str(len(ins.getAttr()))
            case decoded_instruction.instruction_kind.ttwh:
                assert len(ins.getAttr()) == 3, "Three attribs expected. Received " + str(len(ins.getAttr()))
            case _:
                assert False, "Unhandled kind"

        #Two MOP Sequencers available- 0:Unpack Loop 1:Programmable double loop
        if(ins.getAttr()['mop_type'] == 0):
            assert False, "unpack z-mask loop not handled"
            x1 = ins.getAttr()['zmask_lo8_or_loop_count']
            x2 = ins.getAttr()['loop_count']
        else:
            mopType = 1
            #Handle banking of MOP CFG registers
            if(ins.kind == decoded_instruction.instruction_kind.ttqs):
                if(ins.getAttr()['done'] == 1):                 assert False, "dual bank MOP CFG not handled"
                else:           mopCfgBank = 0
            else:
                mopCfgBank = 0

        # nextRelAddr = ins.getRelAddr() + 4
        # return nextRelAddr
        return [mopType, mopCfgBank]
