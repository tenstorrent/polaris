#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
import sys

import argparse
import copy
import json
import simpy

from collections import defaultdict
from enum import Enum

import ttsim.back.tensix_neo.scratchpad as scratchpad
import ttsim.back.tensix_neo.tensixFunc as tensixFunc
import ttsim.back.tensix_neo.triscFunc as triscFunc
from ttsim.back.tensix_neo.isaFunctions import CLEARDVALID_dest_pulse_last_MASKS
from ttsim.back.tensix_neo.isaFunctions import decodeElf, decodeFn, decodeInstr, get_all_function_ranges, decodeLLK
from ttsim.back.tensix_neo.isaFunctions import instr
from ttsim.back.tensix_neo.isaFunctions import THREADMAP
from ttsim.back.tensix_neo.isaFunctions import valueStatus

NOP = "0x2000000"
enableInOrderIssue = False
enableSync      = True
enablePipeStall = True
enableStallWait = True
enableReplay    = True
if(enableSync):
    enablePipeStall = True
skipValidCheckOps          = ["STALLWAIT", "SETRWC"]
skipValidandInUseCheckOps  = ["REPLAY"]
skipValidCheckforRegs      = []
doNotFreeDstPipesforInstrs = ["SEMWAIT"]
freeDstPipesforInstrs      = ["SEMGET", "SEMINIT", "SEMPOST"]
enableThreadwisePipeStates = True
BRANCH_OPS                 = ["JAL", "JALR", "BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"]
MEMORY_LOAD_OPS            = ["LB", "LH", "LW", "LBU", "LHU", "LD", "FLW", "FLD"]
ROBBarrierInsMnemonicContain = [] # ["SETRWC", "INCRWC"]

MAXTIMER_HIGH    = 25000
MAXTIMER_LOW     = 2000

#DEBUG LEVELS
#TODO: Make these command line arguments and use it across all files
DEBUG_TENSIX_LOW_LEVEL  = 0x1
DEBUG_RISC_LOW_LEVEL    = 0x2
DEBUG_RISC_MED_LEVEL    = 0x4
DEBUG_TENSIX_MED_LEVEL  = 0x8
DEBUG_RISC_HIGH_LEVEL   = 0x10
DEBUG_TENSIX_HIGH_LEVEL = 0x20

opHist = {}

class namedStore(simpy.resources.store.Store):
    def __init__(self, n, env, capacity):
        simpy.resources.store.Store.__init__(self, env, capacity)
        self.name =  n
    def printName(self):
        return self.name

def SIMTRACE(pid, tid, ts, ph, lyrn, lyri=None):
    e = {'cat': 'basic', 'pid': pid, 'tid': tid, 'ts': ts, 'ph': ph, 'name': lyrn}
    if lyri: e['args'] = lyri
    return e

class ttReg:
    class regIndex(Enum):
        srcAIndex = 0
        srcBIndex = 1
        srcSIndex = 2
        dstIndex  = 3

    def __init__(self, env, args_dict, coreId, numThreads):
        self.numBanks       = 2         # Number of banks per register
        self.numReg         = 4         # Total number of wide registers that can be accessed by name. True # of register locations = numReg * numBanks
        self.numThreads     = numThreads# Number of threads accessing the registers
        self.coreId         = coreId
        self.regRsrcs       = []        # register resource to prevent reads or writes across threads at the same time
        self.valids         = []        # to stall reads/writes. valid = 0 --> unstall write, valid = 1 --> unstall read
        self.bankSel        = []        # mulltiple banks for register controlled by call to getNextBankId
        self.condCheckValid = []
        self.condWriteValid = []
        self.accThId        = []
        self.accCntToValid  = []

        self.inUse          = []
        for r in range(self.numReg):
            valid   = []
            regRsrc = []
            inUse   = []
            accThId = []
            accCntToValid = []
            for b in range(self.numBanks):
                valid.append(0)
                regRsrc.append(simpy.Resource(env))
                inUse.append(False)
                accThId.append(-1)
                accCntToValid.append(-1)

            self.valids.append(valid)
            self.regRsrcs.append(regRsrc)
            self.inUse.append(inUse)
            self.accThId.append(accThId)
            self.accCntToValid.append(accCntToValid)

        for r in range(self.numReg):
            bSel    = []
            condCheckValid = []
            condWriteValid = []
            orderIndex  = 0
            for c in range(THREADMAP.NUM_CONTEXTS.value):
                bSel.append(0)
                if(args_dict['orderScheme'] is None or r >= len(args_dict['orderScheme']) or len(args_dict['orderScheme'][r]) == 0 ):
                    condCheckValid.append(-1)                       # waitPolarity = -1 if thread is not found in order scheme of register
                    condWriteValid.append(-1)                       # waitPolarity = -1 if thread is not found in order scheme of register
                elif(c ==  args_dict['orderScheme'][r][orderIndex]):
                    condCheckValid.append(orderIndex);                 # waitpolarity = Order Index if thread is found in order scheme of register
                    if(orderIndex == len(args_dict['orderScheme'][r]) - 1): #if orderIndex has exceeded maximum, reset it
                        orderIndex = 0
                        condWriteValid.append(orderIndex);
                    else:
                        orderIndex +=1;                                 # toggle to next thread
                        condWriteValid.append(orderIndex);
                else:
                    condCheckValid.append(-1)                           # waitPolarity = -1 if thread is not found in order scheme of register
                    condWriteValid.append(-1)                           # waitPolarity = -1 if thread is not found in order scheme of register

            self.bankSel.append(bSel)
            self.condCheckValid.append(condCheckValid)
            self.condWriteValid.append(condWriteValid)

        print("condCheckValid=", self.condCheckValid)
        print("condWriteValid=", self.condWriteValid)

        self.env = env

    def getRegIndex(self, regName):
        if regName == "srcA":
            return self.regIndex.srcAIndex.value
        elif regName == "srcB":
            return self.regIndex.srcBIndex.value
        elif regName == "srcS":
            return self.regIndex.srcSIndex.value
        elif regName == "dst":
            return self.regIndex.dstIndex.value
        else:
            assert False, "Unknown register name: " + str(regName)
    def peekCurrBankId(self, r, p):
        return self.bankSel[r][p]

    def getNextBankId(self, r, p):
        currBank    = self.bankSel[r][p]
        self.bankSel[r][p] = (self.bankSel[r][p] + 1) % self.numBanks
        return currBank

    #mode: 0 = no update, 1 = valid only, 2 = inUse Only, 3 = valid + inUse
    def writeValid(self, r, t, val, vMask, bMask, mode, debug=0):
        assert val>=0 , "Write Valid condition Invalid:" + str(val)
        assert self.peekCurrBankId(r,t) < self.numBanks , "Unknown Bank Id"
        assert not(vMask and not bMask), "valid and bank update go hand in hand "+ str(vMask) + "," + str(bMask)
        assert not(not vMask and bMask), "valid and bank update go hand in hand "+ str(vMask) + "," + str(bMask)

        if( self.valids[r][self.peekCurrBankId(r,t)] == val):
            if(mode == 2 or mode == 3):
                if(debug & DEBUG_RISC_HIGH_LEVEL):
                    if(self.inUse[r][self.peekCurrBankId(r,t)] == 0):       print("WARNING: Trying to reset a register that's not in use: " + str(r) + "," + str(self.peekCurrBankId(r,t)) + ",1")
                else:
                    assert self.inUse[r][self.peekCurrBankId(r,t)] == 1 , "WARNING: Trying to reset a register that's not in use: " + str(r) + "," + str(self.peekCurrBankId(r,t)) + ",1"
                if(debug & DEBUG_TENSIX_MED_LEVEL):       print(f"Cycle:{str(self.env.now)} Thread[{t}], Reset inUse threads[{t}, {self.accThId[r][self.peekCurrBankId(r,t)]}] for Reg[{r}][{self.peekCurrBankId(r,t)}] - Valid already set to {val}")
                self.inUse[r][self.peekCurrBankId(r,t)] = 0
            else:
                if(debug & DEBUG_TENSIX_MED_LEVEL):       print(f"Cycle:{str(self.env.now)} Thread[{t}], Skip Valids threads[{t}, {self.accThId[r][self.peekCurrBankId(r,t)]}] for Reg[{r}][{self.peekCurrBankId(r,t)}] - Valid already set to {val}")

            self.accThId[r][self.peekCurrBankId(r,t)]= t
            if(not vMask and not bMask):
                self.accCntToValid[r][self.peekCurrBankId(r,t)]+= 1
            else:
                self.accCntToValid[r][self.peekCurrBankId(r,t)]= 0

            if(debug & DEBUG_TENSIX_MED_LEVEL):       print(f"Cycle:{str(self.env.now)} Thread[{t}], Valids Update threads[{t}, {self.accThId[r][self.peekCurrBankId(r,t)]}] for Reg[{r}][{self.peekCurrBankId(r,t)}] - Valid already set to {val}")
            return True

        if(not vMask and not bMask):
            if(mode == 2 or mode == 3):
                if(debug & DEBUG_RISC_HIGH_LEVEL):
                    if(self.inUse[r][self.peekCurrBankId(r,t)] == 0):       print("WARNING: Trying to reset a register that's not in use: " + str(r) + "," + str(self.peekCurrBankId(r,t)) + ",1")
                else:
                    assert self.inUse[r][self.peekCurrBankId(r,t)] == 1 , "WARNING: Trying to reset a register that's not in use: " + str(r) + "," + str(self.peekCurrBankId(r,t)) + ",1"
                if(debug & DEBUG_TENSIX_MED_LEVEL):       print(f"Cycle:{str(self.env.now)} Thread[{t}], Reset inUse threads[{t}, {self.accThId[r][self.peekCurrBankId(r,t)]}] for Reg[{r}][{self.peekCurrBankId(r,t)}] - Current valid val = {self.valids[r][self.peekCurrBankId(r,t)]} vMask={vMask}, bMask={bMask}")
                self.inUse[r][self.peekCurrBankId(r,t)] = 0
            else:
                if(debug & DEBUG_TENSIX_MED_LEVEL):       print(f"Cycle:{str(self.env.now)} Thread[{t}], Skip Valids threads[{t}, {self.accThId[r][self.peekCurrBankId(r,t)]}] for Reg[{r}][{self.peekCurrBankId(r,t)}] - Current valid val = {self.valids[r][self.peekCurrBankId(r,t)]} vMask={vMask}, bMask={bMask}")
            self.accThId[r][self.peekCurrBankId(r,t)]       = t
            self.accCntToValid[r][self.peekCurrBankId(r,t)]+= 1

            if(debug & DEBUG_TENSIX_MED_LEVEL):       print(f"Cycle:{str(self.env.now)} Thread[{t}], Valids Update threads[{t}, {self.accThId[r][self.peekCurrBankId(r,t)]}] for Reg[{r}][{self.peekCurrBankId(r,t)}] - Current valid val = {self.valids[r][self.peekCurrBankId(r,t)]} vMask={vMask}, bMask={bMask} accCntToValid={self.accCntToValid[r][self.peekCurrBankId(r,t)]}")
            return True

        rsrc    = self.regRsrcs[r][self.peekCurrBankId(r,t)]
        with rsrc.request() as req:
            yield req
            match mode:
                case 0: #Skip Valids / Skip InUse
                    if(debug & DEBUG_TENSIX_LOW_LEVEL) or (debug & DEBUG_TENSIX_MED_LEVEL) or (debug & DEBUG_TENSIX_HIGH_LEVEL):                 print(f"Cycle:{str(self.env.now)} Thread[{t}], Skip all updates for [{r}][{self.peekCurrBankId(r,t)}] accCntToValid={self.accCntToValid[r][self.peekCurrBankId(r,t)]}")
                    self.accThId[r][self.peekCurrBankId(r,t)]       = t
                    self.accCntToValid[r][self.peekCurrBankId(r,t)]+= 1

                    return True
                case 1: #Update Valids / Skip InUse
                    if(debug & DEBUG_TENSIX_LOW_LEVEL) or (debug & DEBUG_TENSIX_MED_LEVEL) or (debug & DEBUG_TENSIX_HIGH_LEVEL):                 print(f"Cycle:{str(self.env.now)} Thread[{t}], Set Valids threads[{t}, {self.accThId[r][self.peekCurrBankId(r,t)]}] for Reg[{r}][{self.peekCurrBankId(r,t)}] = {val} accCntToValid=0")
                    self.accThId[r][self.peekCurrBankId(r,t)]       = t
                    self.accCntToValid[r][self.peekCurrBankId(r,t)] = 0

                    self.valids[r][self.getNextBankId(r,t)] = val

                    return True
                case 2: #Skip Valids / Update InUse
                    if(debug & DEBUG_TENSIX_LOW_LEVEL) or (debug & DEBUG_TENSIX_MED_LEVEL) or (debug & DEBUG_TENSIX_HIGH_LEVEL):
                        if(self.inUse[r][self.peekCurrBankId(r,t)] == 0):               print(f"Cycle:{str(self.env.now)} WARNING: Trying to reset a register that's not in use: {r}, {self.peekCurrBankId(r,t)}, 1")
                    else:
                        assert self.inUse[r][self.peekCurrBankId(r,t)] != 0, f"Trying to reset a register in use that's already in use: {r}, {self.peekCurrBankId(r,t)}, 1"

                    if(debug & DEBUG_TENSIX_LOW_LEVEL) or (debug & DEBUG_TENSIX_MED_LEVEL) or (debug & DEBUG_TENSIX_HIGH_LEVEL):
                        print(f"Cycle:{str(self.env.now)} Thread[{t}], Reset inUse threads[{t}, {self.accThId[r][self.peekCurrBankId(r,t)]}] for Reg[{r}][{self.peekCurrBankId(r,t)}] - Current valid val = {self.valids[r][self.peekCurrBankId(r,t)]} accCntToValid={self.accCntToValid[r][self.peekCurrBankId(r,t)]}")

                    self.accThId[r][self.peekCurrBankId(r,t)]       = t
                    self.accCntToValid[r][self.peekCurrBankId(r,t)]+= 1

                    self.inUse[r][self.peekCurrBankId(r,t)] = 0

                    return True
                case 3: #Update Valids / Update InUse
                    if(debug & DEBUG_TENSIX_LOW_LEVEL) or (debug & DEBUG_TENSIX_MED_LEVEL) or (debug & DEBUG_TENSIX_HIGH_LEVEL):
                        if(self.inUse[r][self.peekCurrBankId(r,t)] == 0):               print(f"Cycle:{str(self.env.now)} WARNING: Trying to reset a register that's not in use: {r}, {self.peekCurrBankId(r,t)}, 1")
                    else:
                        assert self.inUse[r][self.peekCurrBankId(r,t)] != 0, "Trying to reset a register in use that's already in use: " + str(r) + "," + str(self.peekCurrBankId(r,t)) + ",1"

                    self.accThId[r][self.peekCurrBankId(r,t)]       = t
                    self.accCntToValid[r][self.peekCurrBankId(r,t)] = 0

                    if(debug & DEBUG_TENSIX_LOW_LEVEL) or (debug & DEBUG_TENSIX_MED_LEVEL) or (debug & DEBUG_TENSIX_HIGH_LEVEL):
                        print(f"Cycle:{str(self.env.now)} Thread[{t}], Set Valids and Reset inUse threads[{t}, {self.accThId[r][self.peekCurrBankId(r,t)]}] for Reg[{r}][{self.peekCurrBankId(r,t)}] - Current valid val = {self.valids[r][self.peekCurrBankId(r,t)]} New valid val={val} accCntToValid=0")

                    self.inUse[r][self.peekCurrBankId(r,t)]         = 0
                    self.valids[r][self.getNextBankId(r,t)]         = val

                    return True
                case _:
                    assert False, "Unknown mode for writeValid"

    #mode: 0 = no check, 1 = valid only, 2 = inUse Only, 3 = valid + inUse
    def checkValid(self, r, t, v, mode, debug, instr_info = None):
        assert v>=0 , f"Thread{t} Check Valid condition Invalid for reg{r}{v}"
        rsrc        = self.regRsrcs[r][self.peekCurrBankId(r,t)]
        maxTimer    = maxTimerValue = MAXTIMER_HIGH
        while(True):
            with rsrc.request() as req:
                yield req
                match mode:
                    case 0: #Skip Valids / Skip InUse
                        if(debug & DEBUG_TENSIX_MED_LEVEL):                        print(f"Cycle:{str(self.env.now)} Thread[{t}], Skip all checks for [{r}][{self.peekCurrBankId(r,t)}] mode={mode} checkValue={v}")
                        return True
                    case 1: #Update Valids / Skip InUse
                        if(self.valids[r][self.peekCurrBankId(r,t)] == v) and (self.inUse[r][self.peekCurrBankId(r,t)] == 0):
                            if(debug & DEBUG_TENSIX_MED_LEVEL):                    print(f"Cycle:{str(self.env.now)} Thread[{t}], Valids (inter thread=[{t},{self.accThId[r][self.peekCurrBankId(r,t)]}]) condition for [{r}][{self.peekCurrBankId(r,t)}] met, mode={mode}")

                            return True
                        else:
                            if(debug & DEBUG_TENSIX_MED_LEVEL):
                                print(f"DEBUG INFO: Cycle:{str(self.env.now)} Thread[{t}], Reg[{r}][{self.peekCurrBankId(r,t)}] valid={self.valids[r][self.peekCurrBankId(r,t)]}, inUse={self.inUse[r][self.peekCurrBankId(r,t)]}, mode={mode}, checkValue={v}, instr_info={instr_info}")

                    case 2: #Skip Valids / Update InUse
                        if(self.inUse[r][self.peekCurrBankId(r,t)] != 1 ):
                            if(debug & DEBUG_TENSIX_MED_LEVEL):                        print(f"Cycle:{str(self.env.now)} Thread[{t}], set inUse  (inter thread=[{t},{self.accThId[r][self.peekCurrBankId(r,t)]}]) for [{r}][{self.peekCurrBankId(r,t)}] mode={mode}")
                            self.inUse[r][self.peekCurrBankId(r,t)] = 1

                            return True
                        else:
                            if(debug & DEBUG_TENSIX_MED_LEVEL):
                                print(f"DEBUG INFO: Cycle:{str(self.env.now)} Thread[{t}], Reg[{r}][{self.peekCurrBankId(r,t)}] valid={self.valids[r][self.peekCurrBankId(r,t)]}, inUse={self.inUse[r][self.peekCurrBankId(r,t)]}, mode={mode}, checkValue={v}, instr_info={instr_info}")

                    case 3: #Update Valids / Update InUse
                        if(self.valids[r][self.peekCurrBankId(r,t)] == v) and (self.inUse[r][self.peekCurrBankId(r,t)] == 0):
                            if(debug & DEBUG_TENSIX_MED_LEVEL):
                                print(f"Cycle:{str(self.env.now)} Thread[{t}], Valids (inter thread=[{t},{self.accThId[r][self.peekCurrBankId(r,t)]}]) condition for [{r}][{self.peekCurrBankId(r,t)}] met, mode={mode}")
                            self.inUse[r][self.peekCurrBankId(r,t)] = 1

                            if(debug & DEBUG_TENSIX_MED_LEVEL):                    print(f"Cycle:{str(self.env.now)} Thread[{t}], set inUse  (inter thread=[{t},{self.accThId[r][self.peekCurrBankId(r,t)]}]) for [{r}][{self.peekCurrBankId(r,t)}] mode={mode}")
                            return True
                        else:
                            if(debug & DEBUG_TENSIX_MED_LEVEL):
                                print(f"Cycle:{str(self.env.now)} Thread[{t}], Valids condition for [{r}][{self.peekCurrBankId(r,t)}] NOT met yet, valid={self.valids[r][self.peekCurrBankId(r,t)]}, inUse={self.inUse[r][self.peekCurrBankId(r,t)]}, inter thread=[{t},{self.accThId[r][self.peekCurrBankId(r,t)]}] mode={mode}, checkValue={v}") 
                    case _:
                        assert False, "Unknown mode for checkValid"

            yield self.env.timeout(1)
            maxTimer = maxTimer - 1
            if(maxTimer == 0):
                msg = f"WARNING: Cycle:{self.env.now} TCore{self.coreId} Thread{t} Timeout {maxTimerValue} reached for valid check Reg={r} Bank={self.peekCurrBankId(r,t)},Valid={self.valids[r][self.peekCurrBankId(r,t)]} Condition checked={v}"
                if(debug & DEBUG_TENSIX_HIGH_LEVEL):   print(msg); return True
                assert maxTimer > 0, msg

    def writeCondValid(self, r, t, condChkVldVal, condWriVldVal, debug=DEBUG_TENSIX_MED_LEVEL):

        # TODO: Can this -2 for no update be changed to -1
        assert condChkVldVal >= valueStatus.IGNORE and condChkVldVal <= 3, "False,condChkVldVal=" + str(condChkVldVal)
        assert condWriVldVal >= valueStatus.IGNORE and condWriVldVal <= 3, "False,condWriVldVal=" + str(condWriVldVal)
        if(condChkVldVal != valueStatus.IGNORE):
            rsrc    = self.regRsrcs[r][self.peekCurrBankId(r,t)]
            with rsrc.request() as req:
                yield req
                self.condCheckValid[r][t] = condChkVldVal
                self.condWriteValid[r][t] = condWriVldVal
                if debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Updating condCheckValid[{r}][{t}] from {self.condCheckValid[r][t]} to {condChkVldVal}")
                if debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Updating condWriteValid[{r}][{t}] from {self.condWriteValid[r][t]} to {condWriVldVal}")

        return True

    def readValid(self, r, t):
        v   = self.valids[r][self.peekCurrBankId(r,t)]
        return v

    def readCurrBank(self, r, t):
        v   = self.peekCurrBankId(r,t)
        return v

    def printState(self):
        print(self.valids)

class riscRegState:
    def __init__(self, env, coreId, args_dict, numThreads):
        self.env = env
        self.coreId = coreId
        self.args_dict = args_dict

        self.inUse= []
        self.valid= []
        self.regRsrc = []
        for t in range(numThreads):
            self.inUse.append([False for _ in range(32)])
            self.valid.append([0 for _ in range(32)])
            self.regRsrc.append([simpy.Resource(env) for _ in range(32)])

    def setInUse(self, t, r, v):
        assert r>=0 and r<32, "Invalid risc reg number"
        assert v==0 or v==1, "Invalid inUse value"
        rsrc        = self.regRsrc[t][r]
        while(True):
            with rsrc.request() as req:
                yield req
                assert self.inUse[t][r] != v, f"Trying to set a register Thread[{t}] Reg[{r}] to {v} but it's already {v}"
                self.inUse[t][r] = v
                return

    def setValid(self, t, r, v):
        assert r>=0 and r<32, "Invalid risc reg number"
        assert v>=0 , "Invalid valid value"
        self.valid[t][r] = v

    def checkInUse(self, t, r):
        assert r>=0 and r<32, "Invalid risc reg number"
        rsrc        = self.regRsrc[t][r]
        while(True):
            with rsrc.request() as req:
                yield req
                if(self.inUse[t][r] == False):
                    return False
                else:
                    yield self.env.timeout(1)

    def checkValid(self, t, r):
        assert r>=0 and r<32, "Invalid risc reg number"
        return self.valid[t][r]

    """
    def resetInUse(self, t, r):
        assert r>=0 and r<32, "Invalid risc reg number"
        assert self.inUse[t][r] == True, f"Trying to reset a register Thread[{t}] Reg[{r}] that's not in use"
        rsrc        = self.regRsrc[t][r]
        while(True):
            with rsrc.request() as req:
                print(f'Cycle:{self.env.now} Thread[{t}], Reset inUse (inprogress) threads[{t},] for Reg[{r}] ')
                yield req
                self.inUse[t][r] = False
                print(f'Cycle:{self.env.now} Thread[{t}], Reset inUse (done) threads[{t},] for Reg[{r}] ')
                return
    """

    def resetValid(self, t, r, v):
        assert r>=0 and r<32, "Invalid risc reg number"
        assert v>=0 , "Invalid valid value"
        assert self.valid[t][r] == v, "Trying to reset a register that's not valid"
        self.valid[t][r] = 0

class pipeResource:
    def __init__(self, env, args, coreId):
        self.env        = env
        self.coreId     = coreId
        self.numPipes   = len(args['engines'])
        self.numThreads = args['input']['tc'+str(self.coreId)]['numThreads']

        self.currState  = {}
        self.resState   = []

        for i in range(self.numPipes):
            self.resState.append(simpy.Resource(env))
            self.currState[i] = [0 for _ in range(self.numThreads)]

    def readRsrcState(self, pipe_id, thread_id):
        assert pipe_id >= 0, f"@readRsrcState: Pipe ID can not be negative. Given value: {pipe_id}"
        assert pipe_id < self.numPipes, f"@readRsrcState: Given pipe ID {pipe_id} is greater or equal to number of pipes {self.numPipes}"
        assert thread_id >= 0, f"@readRsrcState: Thread ID can not be negative. Given value: {thread_id}"
        assert thread_id < self.numThreads, f"@readRsrcState: Given thread ID {thread_id} is greater or equal to number of threads {self.numThreads}"

        return self.currState[pipe_id][thread_id]

    def setRsrcState(self, pipe_id, thread_id, v, debug, instr_info = None):
        assert pipe_id >= 0, f"@setRsrcState: Pipe ID can not be negative. Given value: {pipe_id}"
        assert pipe_id < self.numPipes, f"@setRsrcState: Given pipe ID {pipe_id} is greater or equal to number of pipes {self.numPipes}"
        assert thread_id >= 0, f"@setRsrcState: Thread ID can not be negative. Given value: {thread_id}"
        assert thread_id < self.numThreads, f"@setRsrcState: Given thread ID {thread_id} is greater or equal to number of threads {self.numThreads}"

        rsrc = self.resState[pipe_id]
        maxTimer = maxTimerValue = MAXTIMER_HIGH
        while(True):
            with rsrc.request() as req:
                yield req
                if(self.currState[pipe_id][thread_id] == v):
                    if debug & DEBUG_TENSIX_HIGH_LEVEL:  print(f"Cycle:{self.env.now} TCore{self.coreId} Thread{thread_id} pipe ID = {pipe_id}, thread ID = {thread_id}. Current resource state = {self.currState[pipe_id][thread_id]}. Condition not met. Value to set = {v}. instr_info: {instr_info}")
                    yield self.env.timeout(1)
                    maxTimer = maxTimer - 1
                    msg = f"WARNING: Cycle:{self.env.now} TCore{self.coreId} Thread{thread_id} Timeout {maxTimerValue} reached for pipe ID = {pipe_id}, thread ID = {thread_id}. Current resource state = {self.currState[pipe_id][thread_id]}. Condition checked (value to set) = {v}. instr_info = {instr_info}"
                    if(debug & DEBUG_TENSIX_HIGH_LEVEL):
                        if(maxTimer == 0):
                            print(msg)
                            return False
                    assert maxTimer > 0, msg
                else:
                    if not enableThreadwisePipeStates:
                        for th_id in range(self.numThreads):
                            self.currState[pipe_id][th_id] = v
                    else:
                        self.currState[pipe_id][thread_id] = v
                        if debug & DEBUG_TENSIX_HIGH_LEVEL:  print(f"Cycle:{self.env.now} TCore{self.coreId} Thread{thread_id} Change state of the resource pipe ID = {pipe_id}, thread ID = {thread_id} to value {v}. instr_info: {instr_info}")

                    return True

    def checkRsrcState(self, pipe_id, thread_id, v, debug, c = 0, instr_info = None):
        assert pipe_id >= 0, f"@checkRsrcState: Pipe ID can not be negative. Given value: {pipe_id}"
        assert pipe_id < self.numPipes, f"@checkRsrcState: Given pipe ID {pipe_id} is greater or equal to number of pipes {self.numPipes}"
        assert thread_id >= 0, f"@checkRsrcState: Thread ID can not be negative. Given value: {thread_id}"
        assert thread_id < self.numThreads, f"@checkRsrcState: Given thread ID {thread_id} is greater or equal to number of threads {self.numThreads}"

        rsrc = self.resState[pipe_id]
        maxTimer = maxTimerValue = MAXTIMER_HIGH
        cyc     = 0
        while(True):
            with rsrc.request() as req:
                yield req
                # if(self.currState[p] == v):     return True
                if((self.currState[pipe_id][thread_id] == v) and (cyc < c)):
                    if debug & DEBUG_TENSIX_HIGH_LEVEL:  print(f"Cycle:{self.env.now} TCore{self.coreId} Thread{thread_id} pipe ID = {pipe_id}. Idle for {cyc} cycles. instr_info: {instr_info}")
                    cyc += 1
                elif(self.currState[pipe_id][thread_id] == v and cyc >= c):
                    if debug & DEBUG_TENSIX_HIGH_LEVEL:  print(f"Cycle:{self.env.now} TCore{self.coreId} Thread{thread_id} pipe ID = {pipe_id}. Idle for {cyc} cycles. Condition met. instr_info: {instr_info}")
                    return True
                else:
                    if debug & DEBUG_TENSIX_HIGH_LEVEL:  print(f"Cycle:{self.env.now} TCore{self.coreId} Thread{thread_id} pipe ID = {pipe_id}. Idle for {cyc} cycles. Condition not met. instr_info: {instr_info}")
                    cyc = 0

            yield self.env.timeout(1)
            maxTimer = maxTimer - 1

            if(maxTimer == 0):
                msg = f"WARNING: Cycle:{self.env.now} TCore{self.coreId} TCore{self.coreId} Thread{thread_id} Timeout {maxTimerValue} reached for pipe ID = {pipe_id}, thread ID = {thread_id}. Current resource state = {self.currState[pipe_id][thread_id]}. Condition checked (value to check) = {v}. instr_info = {instr_info}"
                if(debug & DEBUG_TENSIX_HIGH_LEVEL):   print(msg); return True
                assert maxTimer > 0, msg

    def __str__(self):
        printString = ""
        for k in self.currState.keys():
            printString += "currState[" + str(k) + "]=" + str(self.currState[k]) + "\n"
        return(printString)

class replayState:
    def __init__(self, env, coreId, thread):
        self.env            = env
        self.rMode          = 0
        self.prevrMode      = self.rMode
        self.replayStartIdx = -1
        self.replayLen      = -1

        self.replayBatchLen = -1
        self.rModeRsrc = simpy.Resource(env)

        self.replayList = []
        for i in range(32):
            self.replayList.append(instr())

        self.coreId         = coreId
        self.threadId       = thread

    def updateRMode(self, replayAttr, debug):

        assert len(replayAttr) == 5 , "Replay Attribs should be five"
        load            = 0
        exec_while_load = 0
        replayStartIdx  = -1
        replayLen       = 0

        rsrc = self.rModeRsrc
        with rsrc.request() as req:
            yield req

            load            = replayAttr[0]
            exec_while_load = replayAttr[1]
            replayStartIdx  = replayAttr[2]
            replayLen       = replayAttr[3]
            replayCurrThread= replayAttr[4]


            self.prevrMode      = self.rMode
            self.replayStartIdx = replayStartIdx
            self.replayLen      = replayLen
            assert replayCurrThread == self.threadId, "Thread Mismatch: replayState Thread= " + self.threadId + ", Ins Thread= " + replayCurrThread
            self.replayBatchLen = 0

            if(load == 1 and exec_while_load == 1):             self.rMode = 2    # Load and Execute
            elif(load == 1 ):                                   self.rMode = 1    # Load
            elif(load == 0 and exec_while_load == 0):           self.rMode = 3    # Execute
            else:                                               assert False, "What's the use of replay then? Load:" + load + ",Exec While Loading:" + exec_while_load
            if(debug & DEBUG_TENSIX_MED_LEVEL):          print(f"Cycle:{self.env.now} TCore{self.coreId} Thread: {3} Updating RMode  from {1}  to {2}".format(self.env.now, self.prevrMode, self.rMode, self.threadId, self.coreId))

            return True


    def readRMode(self):
        rsrc = self.rModeRsrc
        with rsrc.request() as req:
            yield req
            return self.rMode

    def getRMode(self):
        return self.rMode

    def getPrevRMode(self):
        return self.prevrMode

    def printReplayList(self):
        print("-----Replay List------")
        for ins in self.replayList:
            if(ins.getRelAddr() >=0):            ins.printInstr(self.threadId)
        print("-----Replay List------")

    def loadReplayList(self, ins, debug):
        assert self.replayBatchLen >= 0 and self.replayBatchLen < self.replayLen , "Illegal Batch Length " + str(self.replayBatchLen) + ", Len=" + str(self.replayLen)
        assert ins.getThread() == self.threadId, "Replay Loading a different thread. Is it legal? Instruction thread = " + str(ins.getThread()) + " Replay Buffer[" + str(self.threadId) + "]"
        rsrc = self.rModeRsrc
        with rsrc.request() as req:
            yield req
            self.replayList[self.replayStartIdx + self.replayBatchLen] = ins
            if(self.replayBatchLen == self.replayLen - 1):
                if(debug & DEBUG_TENSIX_MED_LEVEL):         self.printReplayList()
                self.prevrMode      = self.rMode
                self.rMode          = 0
                self.replayBatchLen = 0
                self.replayStartIdx = -1
                if(debug & DEBUG_TENSIX_MED_LEVEL):         print(f"Cycle:{self.env.now} TCore{self.coreId} Thread [{3}] Setting RMode from {1}  to {2}".format(self.env.now, self.prevrMode, self.rMode, self.threadId, self.coreId))
            else:
                self.replayBatchLen+=1

            return True

    def execReplayList(self, debug):
        # rsrc = self.rModeRsrc
        # with rsrc.request() as req:
        #     yield req
        assert self.replayBatchLen >= 0 and self.replayBatchLen < self.replayLen , "Illegal Batch Length " + str(self.replayBatchLen) + ", Len=" + str(self.replayLen)
        ins = self.replayList[self.replayStartIdx + self.replayBatchLen]
        assert ins.getThread() == self.threadId, "Replay Executing a different thread. Is it legal? Instruction thread = " + str(ins.getThread()) + " Replay Buffer[" + str(self.threadId) + "]"
        if(self.replayBatchLen == self.replayLen - 1):
            self.prevrMode      = self.rMode
            self.rMode          = 0
            self.replayBatchLen = 0
            self.replayStartIdx = -1
        else:
            if(debug & DEBUG_TENSIX_MED_LEVEL):     print(f"Cycle:{self.env.now} TCore{self.coreId} Thread [{3}] Setting RMode from {1}  to {2}".format(self.env.now, self.prevrMode, self.rMode, self.threadId, self.coreId))
            self.replayBatchLen+=1

        return ins

class rob:
    def __init__(self,env, c, t):
        self.env        = env
        self.insId      = -1
        self.idRob      = []
        self.valRob     = []
        self.robRsrc    = simpy.Resource(env)
        self.coreId     = c
        self.threadId   = t

    def getCurrId(self): return self.insId
    def incId(self): self.insId += 1
    def decId(self): self.insId -= 1

    def appendRob(self, ins, debug):
        rsrc    = self.robRsrc
        with rsrc.request() as req:
            yield req
            self.incId()
            if debug & DEBUG_TENSIX_HIGH_LEVEL:
                print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{ins.getThread()} insId{self.getCurrId()} Instruction:{ins.getOp()} Inserting (inprogress) into ROB ")
            self.idRob.append(self.getCurrId())
            self.valRob.append(ins)
            if debug & DEBUG_TENSIX_MED_LEVEL:
                print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{ins.getThread()} insId{self.getCurrId()} Instruction:{ins.getOp()} Inserting (done) into ROB at position={self.findRob(self.getCurrId(),debug)}")
            return(self.getCurrId())

    def popRob(self, val, debug):
        maxTimer = maxTimerValue = MAXTIMER_HIGH
        assert val >= 0 , f"Thread[{self.threadId} Illegal insId{val}"
        rsrc    = self.robRsrc
        while(True):
            with rsrc.request() as req:
                yield req
                idRobIndex  = self.findRob(val,debug)
                insOp       = self.valRob[idRobIndex].getOp()
                insAddr     = hex(self.valRob[idRobIndex].getRelAddr())
                if idRobIndex == 0:
                    if debug & DEBUG_TENSIX_HIGH_LEVEL:    print(f"Cycle:{self.env.now} Addr:{insAddr} TCore{self.coreId} Thread{self.threadId} insId{val} Instruction:{insOp} Removing (inprogress) from ROB at position={idRobIndex}. maxTimer={maxTimerValue - maxTimer}")
                    self.idRob.remove(val)
                    del self.valRob[idRobIndex]
                    if debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{insAddr} TCore{self.coreId} Thread{self.threadId} insId{val} Instruction:{insOp} Removing (done)       from ROB at position={idRobIndex}. maxTimer={maxTimerValue - maxTimer}")
                    return True
                else:
                    if debug & DEBUG_TENSIX_HIGH_LEVEL:    print(f"Cycle:{self.env.now} Addr:{insAddr} TCore{self.coreId} Thread{self.threadId} insId{val} Instruction:{insOp} Removing (failed)     from ROB at position={idRobIndex}")

            yield self.env.timeout(1)
            maxTimer = maxTimer - 1
            if(maxTimer == 0):
                msg = f"WARNING: Cycle:{self.env.now} Addr:{insAddr} TCore {self.coreId} Thread{self.threadId} Timeout {maxTimerValue} reached for pop from ROB insId{val} Position={idRobIndex}"
                if(debug & DEBUG_TENSIX_HIGH_LEVEL):   print(msg); return True
                assert maxTimer > 0, msg

    def headOfRob(self,val,debug, instr_info = None):
        maxTimer = maxTimerValue = MAXTIMER_HIGH
        assert val >= 0 , f"Thread{self.threadId} Illegal insId{val}"
        rsrc    = self.robRsrc
        while(True):
            with rsrc.request() as req:
                yield req
                idRobIndex  = self.findRob(val,debug)
                insOp       = self.valRob[idRobIndex].mnemonic
                insAddr     = hex(self.valRob[idRobIndex].getRelAddr())

                if debug & DEBUG_TENSIX_HIGH_LEVEL:    print(f"Cycle:{self.env.now} Addr:{insAddr} TCore{self.coreId} Thread{self.threadId} insId{val} Instruction:{insOp} Check head of ROB (inprogress) at position={idRobIndex}")
                if idRobIndex == 0:
                    if debug & DEBUG_TENSIX_MED_LEVEL: print(f"Cycle:{self.env.now} Addr:{insAddr} TCore{self.coreId} Thread{self.threadId} insId{val} Instruction:{insOp} Check head of ROB (done)       at position={idRobIndex}. maxTimer={maxTimerValue - maxTimer}")
                    return True

            yield self.env.timeout(1)
            maxTimer = maxTimer - 1
            if(maxTimer == 0):
                msg = f"WARNING: Cycle:{self.env.now} Addr:{insAddr} TCore{self.coreId} Thread{self.threadId} Timeout {maxTimerValue} reached for headOfRob with insId{val} Position=idRobIndex"
                if(debug & DEBUG_TENSIX_HIGH_LEVEL):   print(msg); return True
                assert maxTimer > 0, msg


    def removeRob(self, val, debug):
        maxTimer = maxTimerValue = MAXTIMER_HIGH
        assert val >= 0 , f"Thread{self.threadId} Illegal insId{val}"
        rsrc    = self.robRsrc
        while(True):
            with rsrc.request() as req:
                yield req
                idRobIndex  = self.findRob(val,debug)
                insOp       = self.valRob[idRobIndex].getOp()
                insAddr     = hex(self.valRob[idRobIndex].getRelAddr())

                if debug & DEBUG_TENSIX_HIGH_LEVEL:    print(f"Cycle:{self.env.now} Addr {insAddr} TCore{self.coreId} Thread{self.threadId} insId{val} Instruction:{insOp} Removing (inprogress) from ROB at position={idRobIndex}. maxTimer={maxTimerValue - maxTimer}")
                self.idRob.remove(val)
                del self.valRob[idRobIndex]
                if debug & DEBUG_TENSIX_MED_LEVEL:    print(f"Cycle:{self.env.now} Addr {insAddr} TCore{self.coreId} Thread{self.threadId} insId{val} Instruction:{insOp} Removing (done) from ROB at position={idRobIndex}. maxTimer={maxTimerValue - maxTimer}")
                return True

    def findRob(self,val,debug):
        if debug & DEBUG_TENSIX_HIGH_LEVEL:     print(f"Cycle:{self.env.now} TCore{self.coreId} Thread{self.threadId} Find in ROB insId{val}. Position={self.idRob.index(val)} ")
        assert val in self.idRob , f"Can't find TCore{self.coreId} Thread{self.threadId} insId{val} in ROB"
        return self.idRob.index(val)

class thread:
    def __init__(self,args_dict, env, kernelName, coreId, threadId, threadFunc, tensixFunc, ttFifo, ttBuffer, rState, ttReg, riscReg, pipes, replayState, insROB, traceEventList):
        self.env = env
        self.args_dict  = args_dict
        self.coreId     = coreId

        self.riscPipeDepth = 3 if('riscPipeDepth' not in args_dict) else args_dict['riscPipeDepth'] - 1
        self.branchMisPredictLat = 3 if('branchMisPredictPenalty' not in args_dict) else args_dict['branchMisPredictPenalty']
        self.enableRiscPM  = True if('enableRiscPM' not in args_dict) else args_dict['enableRiscPM']
        self.enableScoreboardCheckforRegs = True if('enableScoreboardCheckforRegs' not in args_dict) else args_dict['enableScoreboardCheckforRegs']
        self.enableForwardingforRegs = True if('enableForwardingforRegs' not in args_dict) else args_dict['enableForwardingforRegs']

        self.inputBuff  = simpy.Store(self.env, capacity=2)
        self.mopBuff    = simpy.Store(self.env, capacity=1)
        self.instrBuff  = simpy.Store(self.env, capacity=2)
        self.kernelsToExecute = simpy.Store(self.env, capacity=1)
        self.pcLock           = simpy.Resource(self.env, capacity=1)

        self.riscCheckRegBuff  = simpy.Store(self.env, capacity=1)
        self.riscExecBuff = simpy.Store(self.env, capacity=1)
        self.riscExecTrkBuff = simpy.Store(self.env, capacity=self.riscPipeDepth)
        self.riscResetRegBuff = simpy.Store(self.env, capacity=1)

        self.procThread = self.env.process(self.execThread())
        self.procIns    = self.env.process(self.execIns())
        self.procRIns1  = self.env.process(self.execCheckAndSetRiscReg(self.riscPipeDepth))
        self.procRIns2  = self.env.process(self.execResetRiscReg())
        self.procMop    = self.env.process(self.mopDecode())
        self.procArb    = self.env.process(self.tArbiter(ttFifo, ttBuffer, replayState, insROB, traceEventList))

        self.debug      = args_dict['debug']

        testPath    = self.args_dict['input']['tc' + str(self.coreId)]['th' + str(threadId) + 'Path']
        elfName     = self.args_dict['input']['tc' + str(self.coreId)]['th' + str(threadId) + 'Elf']

        self.startKernel        = kernelName
        self.name               = elfName.replace(".elf","")
        self.threadId           = threadId
        self.threadListofLists  = {}
        # print(self.name)

        # Control Flow
        # Stage 1: Create a dict with [kernel][addr] --> ins
        self.threadListofListsWithAddr  = {}
        self.pc                         = 0
        self.prevPC                     = 0
        self.startAddr                  = -1
        self.endAddr                    = -1

        self.allFnsRanges               = []

        if(self.args_dict['input']['syn']):
            # equivalent to get_all_function_ranges functionality for synthetic tests
            self.allFnsRanges       = self.args_dict['input']['syn' + str(threadId) + 'Range']
            # equivalent to decodeElf functionality for synthetic tests
            beginOffset = 100000000000
            for i in range(len(self.allFnsRanges)):     beginOffset = min(self.allFnsRanges[i][1],beginOffset)
            for i in range(len(self.allFnsRanges)):
                kernel      = self.allFnsRanges[i][0]
                startAddr   = self.allFnsRanges[i][1]
                endAddr     = self.allFnsRanges[i][2]
                arch = 'ttqs' if('arch' not in self.args_dict) else self.args_dict['arch']
                insDict  = decodeLLK(testPath, elfName, kernel, arch, startAddr-beginOffset, endAddr-beginOffset)
                assert len(insDict) == 1, f"Error in decodeLLK for {testPath}, {elfName}, {kernel}, {startAddr}, {endAddr}"
                for k,v in insDict.items():
                    self.threadListofLists[k] = v
        else:
            self.allFnsRanges       = get_all_function_ranges(testPath, elfName)
            self.threadListofLists  = decodeElf(testPath, elfName, self.args_dict['ttISA'])

        print(self.allFnsRanges)
        numFunctions            = len(self.allFnsRanges)

        for i in range(len(self.allFnsRanges)):
            kernel      = self.allFnsRanges[i][0]
            startAddr   = self.allFnsRanges[i][1]
            endAddr     = self.allFnsRanges[i][2]
            print("Kernel:", kernel)
            print("Thread ID: ", self.threadId, ", Kernel:", kernel, ", StartAddress = ", hex(self.allFnsRanges[i][1]), ",EndAddress=", hex(self.allFnsRanges[i][2]))
            print("Length of Kernel=", len(self.threadListofLists[kernel] ))
            # Start Kernel - Start Address
            # if(kernel == self.startKernel):
            #     self.startAddr  = self.pc = startAddr
            #     self.endAddr    = endAddr
            #     self.prevPC     = 0
            #     self.startKernelIdx = i

            self.threadListofListsWithAddr[kernel] ={}
            offset = 0
            for insIndex in range(len(self.threadListofLists[kernel])):
                insDec      = self.threadListofLists[kernel][insIndex]
                ins         = instr(insDec)

                ins.setCoreId(self.coreId)
                ins.setThread(self.threadId)

                # To be removed once its handled by bu-playground
                ins.setRelAddr(startAddr + offset)

                ins.printInstr(self.threadId)
                self.threadListofListsWithAddr[kernel][startAddr + offset]   = ins
                offset      += 4

            assert startAddr + offset == endAddr, f"Kernel:{kernel} Thread{self.threadId} Error in Instruction Address assignment {startAddr + offset},{endAddr}"

        print("Number of Instructions in Thread", threadId, "=", len(self.threadListofLists[self.startKernel]))
        if(len(self.threadListofLists[self.startKernel]) == 0):
            sys.exit(1,"Empty kernel")

        self.env.process(self.updateKernelsToExecute())

        self.ttInstr = []

        self.tFunc      = threadFunc
        self.tensixFunc = tensixFunc

        self.rState   = rState
        self.ttThreadReg= ttReg
        self.riscReg   = riscReg
        self.pipes      = pipes

        # Per Thread Register
        self.mopAddr = copy.deepcopy(self.tensixFunc.getMopCfgAddr())
        print("MOP_CFG_ADDR[",self.name, "]=",hex(self.mopAddr))

        #Stats
        self.rInstructions         = 0
        self.ttInstructions        = 0
        self.sumRiscLatency       = 0
        self.sumRiscStallLatency  = 0

        self.addrHist               = {}


    def findIns(self, targetAddr):
        # Find Kernel. Find Address within kernel. Return Instruction
        if(targetAddr>= self.allFnsRanges[self.startKernelIdx][1] and
            targetAddr<= (self.allFnsRanges[self.startKernelIdx][2] - 1)
           ):
            if(targetAddr in self.addrHist):    self.addrHist[targetAddr] += 1
            else:                               self.addrHist[targetAddr] = 1

            if targetAddr not in self.threadListofListsWithAddr[self.startKernel].keys():
                msg  = f"- error: could not find targetAddr in threadListofListsWithAddr[self.startKernel].keys()\n"
                msg += f"- self.startKernel: {self.startKernel}\n"
                msg += f"- self.threadId: {self.threadId}\n"
                msg += f"- target address: {targetAddr}, {hex(targetAddr)}\n"
                msg += f"- start address:  {self.startAddr}, {hex(self.startAddr)}\n"
                msg += f"- end address:    {self.endAddr}, {hex(self.endAddr)}"
                msg += f"- keys:           {self.threadListofListsWithAddr[self.startKernel].keys()}"

                raise Exception(msg)

            instruction = copy.deepcopy(self.threadListofListsWithAddr[self.startKernel][targetAddr])
            if self.debug & DEBUG_TENSIX_HIGH_LEVEL:   print(f'Cycle:{self.env.now} TCore{self.coreId} Thread{self.threadId} Address:{hex(targetAddr)} ObjectId:{hex(id(instruction))} Found Instruction in kernel {self.startKernel}: {instruction.mnemonic} ')
            return(instruction)
        else:
            for i in range(len(self.allFnsRanges)):
                kernel      = self.allFnsRanges[i][0]
                startAddr   = self.allFnsRanges[i][1]
                endAddr     = self.allFnsRanges[i][2]

                # Start Kernel - Start Address
                if(targetAddr >= self.allFnsRanges[i][1] and
                    (targetAddr <= self.allFnsRanges[i][2] - 1)
                ):
                    # print("In another Kernel = ", kernel, hex(targetAddr), hex(startAddr), hex(endAddr))
                    if(targetAddr in self.addrHist):    self.addrHist[targetAddr] += 1
                    else:                               self.addrHist[targetAddr] = 1

                    instruction = copy.deepcopy(self.threadListofListsWithAddr[kernel][targetAddr])
                    if self.debug & DEBUG_TENSIX_HIGH_LEVEL:   print(f'Cycle:{self.env.now} TCore{self.coreId} Thread{self.threadId} Address:{hex(targetAddr)} ObjectId:{hex(id(instruction))} Found Instruction in kernel {kernel}: {instruction.mnemonic} ')
                    return(instruction)

        print("Could not find instruction at ", hex(targetAddr))
        print(self.tFunc.__printReg__())
        sys.exit(1)

    def execKernel(self):
        insIndex    = 0
        addr    = self.startAddr
        if(self.debug & DEBUG_RISC_HIGH_LEVEL):            print(self.env.now, self.name, "Start")

        # Control Flow
        # Stage 2: Address based access
        while( self.pc != 0 ):                     # Last Ret, PC is set to zero
            if(self.pc != self.prevPC):
                ins     = self.findIns(self.pc)
                if(self.debug & DEBUG_RISC_HIGH_LEVEL) and ( self.pc- self.prevPC != 4) and (self.prevPC != 0):                print("Cycle:", self.env.now," Thread[",self.threadId, "] Jump Instruction executed ", hex(self.prevPC), ",", hex(self.pc), ",", self.pc - self.prevPC, sep='')
                self.prevPC = self.pc
                if self.debug & DEBUG_TENSIX_HIGH_LEVEL:
                    print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} ObjectId:{hex(id(ins))} Fetched Instruction:{ins.mnemonic} from Program Counter")
                yield self.inputBuff.put(ins)
            else:
                yield self.env.timeout(1)

        assert self.findIns(self.prevPC).mnemonic == "JALR" or (self.findIns(self.prevPC).mnemonic == "JAL" and self.findIns(self.prevPC).getImm()[0] == 0), f"Last instruction not JALR? or JAL with imm=0?:{self.findIns(self.prevPC).getImm()[0]} Op:{self.findIns(self.prevPC).mnemonic},Imm:{self.findIns(self.prevPC).getImm()}"
        print(f"TCore{self.coreId} Thread{self.threadId} End of Kernel:{hex(self.endAddr)} PC={hex(self.pc)} prevPC={hex(self.prevPC)} LastOp={self.findIns(self.prevPC).mnemonic}")

        return True

    def updateKernelsToExecute(self):

        self.kernelsToExecuteNames = [fn_name for fn_name, _, _ in self.allFnsRanges if fn_name.startswith("_GLOBAL__sub_I_")]
        self.kernelsToExecuteNames.append(self.startKernel)

        print(f"- Kernels to execute in thread {self.threadId}: {self.kernelsToExecuteNames}")

        for kernelName in self.kernelsToExecuteNames:
            kernelIdx = [i for i in range(len(self.allFnsRanges)) if self.allFnsRanges[i][0] == kernelName]
            if 0 == len(kernelIdx):
                msg = f"- error: could not find kernel {kernelName} in binary file associated thread {self.threadId}\n"
                msg += f"- available kernels:\n"
                for name, _, _ in self.allFnsRanges:
                    msg += f"  - {name}\n"
                raise Exception(msg.rstrip())
            elif 1 != len(kernelIdx):
                msg = f"- error: found multiple matches for kernel {kernelName} in binary file associated thread {self.threadId}\n"
                msg += f"- available kernels:\n"
                for name, _, _ in self.allFnsRanges:
                    msg += f"  - {name}\n"
                raise Exception(msg.rstrip())

            kernelIdx = kernelIdx[0] # expected length is 1.

            kernelToExecuteDict = dict()
            kernelToExecuteDict["idx"]       = kernelIdx
            kernelToExecuteDict["name"]      = self.allFnsRanges[kernelIdx][0]
            kernelToExecuteDict["startAddr"] = self.allFnsRanges[kernelIdx][1]
            kernelToExecuteDict["endAddr"]   = self.allFnsRanges[kernelIdx][2]

            yield self.kernelsToExecute.put(kernelToExecuteDict)

    def execThread(self):
        while (True):
            kernelToExcute = yield self.kernelsToExecute.get()
            with self.pcLock.request() as req:
                yield req

                self.startAddr      = kernelToExcute["startAddr"]
                self.pc             = kernelToExcute["startAddr"]
                self.endAddr        = kernelToExcute["endAddr"]
                self.prevPC         = 0
                self.startKernelIdx = kernelToExcute["idx"]
                self.startKernel    = kernelToExcute["name"]

                proc = self.env.process(self.execKernel())
                yield proc

    def execIns(self):
        while(True):
            ins = yield self.inputBuff.get()
            if self.debug & DEBUG_TENSIX_HIGH_LEVEL:
                print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} ObjectId:{hex(id(ins))} Fetched Instruction:{ins.mnemonic} from Input Buffer")
            if(ins.isMop() or ins.isTT()):
                yield self.mopBuff.put(ins)
            else:
                if(self.enableRiscPM):
                    yield self.riscCheckRegBuff.put(ins)

                nextAddr = self.tFunc.execRIns(ins, self.env.now)

                ### Handle all writeCfg ##
                # TODO: Change to a condition on write to CFG
                if ins.mnemonic  == "SW":
                    bankCurrVal = self.ttThreadReg.readCurrBank(3,ins.getContext())
                    vldCurrVal  = self.ttThreadReg.readValid(3,ins.getContext())
                    vldUpd      = {}
                    bankUpd     = {}
                    if(ins.hasBankUpdMask(3)):
                        bankNewVal  = ins.getBankUpdMask(3)
                        if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"Writing to Thread[{self.threadId}] Context={ins.getContext()} bank[3][{ins.getThread()}]= {bankNewVal}. NewVal = {bankCurrVal}")
                        if(bankCurrVal != bankNewVal):
                            bankUpd[3] = 1; vldUpd[3] = 1
                            ins.setVldUpdMask(vldUpd)
                            ins.setBankUpdMask(bankUpd)

                            if 3 not in skipValidCheckforRegs:
                                if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"Post skipValidCheck Writing to Thread[{self.threadId}] Context={ins.getContext()} bank[3][{ins.getThread()}]= {bankNewVal}. NewVal = {bankCurrVal}")
                                yield self.env.process(
                                    self.ttThreadReg.writeValid(3,                     # dstList[i]
                                                        ins.getContext(),
                                                        1 - vldCurrVal,
                                                        ins.getVldUpdMask(3),
                                                        ins.getBankUpdMask(3),
                                                        1,                      # mode = 1: Only Valids/Bank Update
                                                        self.args_dict['debug'])
                                        )

                    #Cond[Chk/Wri]Vld Update
                    condVldUpdCheck = []
                    for i in range(self.args_dict['input']['tc' + str(self.coreId)]['numThreads']):
                        # Currently only programming for reg 3 (destination register) is supported
                        if(ins.hasCondChkVldUpd(3, i)):
                            assert ins.hasCondWriVldUpd(3, i) , "Mismatch between condChkVld and condWriVld of destination"
                            print("writeCondValid: " , ins.getCondChkVldUpd(3, i), ins.getCondWriVldUpd(3, i))
                            condVldUpdCheck.append(self.env.process(
                                self.ttThreadReg.writeCondValid(3, i, ins.getCondChkVldUpd(3, i), ins.getCondWriVldUpd(3, i))
                            ))

                    checkTime = self.env.now
                    if(len(condVldUpdCheck) >0 ):
                        if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} Thread{ins.getThread()} insId{ins.getInsId()} Instruction:{ins.getOp()} CondValids Update Condition (inprogress) ".format(ins.mnemonic, -1, ins.getThread(), hex(ins.getRelAddr()), self.env.now))
                        yield simpy.events.AllOf(self.env, condVldUpdCheck)
                        if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} Thread{ins.getThread()} insId{ins.getInsId()} Instruction:{ins.getOp()} CondValids Update Condition (done) StallTime:{5}".format(ins.mnemonic, -1 , ins.getThread(), hex(ins.getRelAddr()), self.env.now, (self.env.now - checkTime)))
                        if self.debug & DEBUG_TENSIX_MED_LEVEL:
                            print("condCheckValid: ", end='')
                            for i in range(4):  #TODO: Replace with number of registers = 4
                                print(self.ttThreadReg.condCheckValid[i],  end =';')
                            print()
                            print("condWriteValid: ", end='')
                            for i in range(4):  #TODO: Replace with number of registers = 4
                                print(self.ttThreadReg.condWriteValid[i],  end =';')
                            print()

                # Intercept intstruction buffer writes
                decodedInsDec = self.tFunc.decodeInstructionBufMem()
                if decodedInsDec is not None:
                    decodedIns  = instr(decodedInsDec)
                    decodedIns.setRelAddr(ins.getRelAddr())
                    decodedIns.setCoreId(self.coreId)
                    decodedIns.setThread(ins.getThread())

                    if(self.debug & DEBUG_TENSIX_MED_LEVEL):   print(f"INSTRBUF: {decodedIns.printInstr(self.threadId)} ")
                    yield self.mopBuff.put(decodedIns)
                    if(self.debug & DEBUG_TENSIX_MED_LEVEL):   print("Placing into MopBuffer ", decodedIns.mnemonic)

                    # To avoid double increment of PC on instruction buffer insertion, PC is not incremented here. PC should be incremented by the TT instruction inserted into mopBuff
                    # Assumes 1 SW : 1 TT Ins mapping
                    # if 1 SW : n TT Ins mapping, major changes in control flow management needed.
                    # If 1 SW : 1 macro TT (MOP/REPLAY), that should work as MOP/REPLAY will block PC

                else:
                    self.prevPC = self.pc
                    self.pc = nextAddr
                    self.rInstructions += 1

            yield self.env.timeout(1)

    def execCheckAndSetRiscReg(self, execLatency):
        while(True):
            ins = yield self.riscCheckRegBuff.get()
            startTime = self.env.now

            if ins.getOp() in BRANCH_OPS:
                yield self.env.timeout(self.branchMisPredictLat)   # Branch Delay

            # Check Scoreboard for source and destination registers
            scoreBoardCheck = []
            if(len(ins.getSrcInt()) != 0):
                srcList = ins.getSrcInt()
                for i in range(len(srcList)):
                    if (srcList[i] != 0 and self.enableScoreboardCheckforRegs):
                        scoreBoardCheck.append(self.env.process(self.riscReg.checkInUse(self.threadId, srcList[i])))
            if(len(ins.getDstInt()) != 0):
                dstList = ins.getDstInt()
                for i in range(len(dstList)):
                    if (dstList[i] != 0 and self.enableScoreboardCheckforRegs):
                        scoreBoardCheck.append(self.env.process(self.riscReg.checkInUse(self.threadId, dstList[i])))
            if(len(scoreBoardCheck) >0 ):    yield simpy.events.AllOf(self.env, scoreBoardCheck)

            if self.debug & DEBUG_TENSIX_HIGH_LEVEL:
                print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} Instruction:{ins.getOp()} ObjectId:{hex(id(ins))} CheckInUse-RISC (done):{ins.mnemonic}. Stall Time:{self.env.now - startTime} ")

            #Set Scoreboard for destination registers
            scoreBoardSet = []
            if(len(ins.getDstInt()) != 0):
                dstList = ins.getDstInt()
                for i in range(len(dstList)):
                    #1. Skip register 0
                    #2. If forwarding enabled, enable scoreboard only for loads
                    #3. If scoreboard disabled, skip
                    if (dstList[i] != 0 and
                        (not self.enableForwardingforRegs or (self.enableForwardingforRegs and ins.getOp() in MEMORY_LOAD_OPS)) and
                        self.enableScoreboardCheckforRegs):
                        scoreBoardSet.append(self.env.process(self.riscReg.setInUse(self.threadId, dstList[i], 1)))
            if(len(scoreBoardSet) >0 ):    yield simpy.events.AllOf(self.env, scoreBoardSet)

            yield self.env.timeout(1)
            if self.debug & DEBUG_TENSIX_MED_LEVEL:
                print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} Instruction:{ins.getOp()} ObjectId:{hex(id(ins))} SetInUse-RISC (done):{ins.mnemonic} srcList={ins.getSrcInt()} dstList={ins.getDstInt()}: Stall Time:{self.env.now - startTime} ")

            yield self.riscExecTrkBuff.put(ins)
            self.sumRiscStallLatency += (self.env.now - startTime)
            self.env.process(self.executeRiscInstruction(ins, execLatency, startTime))

    def executeRiscInstruction(self, ins, latency, startTime):
        if self.debug & DEBUG_TENSIX_HIGH_LEVEL:
            print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} Instruction:{ins.getOp()} ObjectId:{hex(id(ins))} Execution-RISC (inprogress):{ins.mnemonic}")

        yield self.env.timeout(latency)
        ins = yield self.riscExecTrkBuff.get()
        yield self.riscResetRegBuff.put(ins)

        self.sumRiscLatency += (self.env.now - startTime + 1)
        if self.debug & DEBUG_TENSIX_MED_LEVEL:
            print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} Instruction:{ins.getOp()} ObjectId:{hex(id(ins))} Execution-RISC (done):{ins.mnemonic}. Latency:{self.env.now - startTime} ")

        return

    def execResetRiscReg(self):
        while(True):
            ins = yield self.riscResetRegBuff.get()
            if self.debug & DEBUG_TENSIX_HIGH_LEVEL:
                print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} Instruction:{ins.getOp()} ObjectId:{hex(id(ins))} ResetInUse-RISC (inprogress):{ins.mnemonic} srcList={ins.getSrcInt()} dstList={ins.getDstInt()}")

            scoreBoardSet = []
            if(len(ins.getDstInt()) != 0):
                dstList = ins.getDstInt()
                for i in range(len(dstList)):
                    #1. Skip register 0
                    #2. If forwarding enabled, enable scoreboard only for loads
                    #3. If scoreboard disabled, skip
                    if (dstList[i] != 0 and
                        (not self.enableForwardingforRegs or (self.enableForwardingforRegs and ins.getOp() in MEMORY_LOAD_OPS)) and
                        self.enableScoreboardCheckforRegs):
                        scoreBoardSet.append(self.env.process(self.riscReg.setInUse(self.threadId, dstList[i], 0)))

            if(len(scoreBoardSet) > 0 ):    yield simpy.events.AllOf(self.env, scoreBoardSet)
            yield self.env.timeout(1)
            if self.debug & DEBUG_TENSIX_MED_LEVEL:
                print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} Instruction:{ins.getOp()} ObjectId:{hex(id(ins))} ResetInUse-RISC (done):{ins.mnemonic} from Instruction Buffer srcList={ins.getSrcInt()} dstList={ins.getDstInt()}")

    def mopDecode(self):
        while(True):
            ins = yield self.mopBuff.get()
            if(ins.isMop() ):
                if(self.debug & DEBUG_TENSIX_LOW_LEVEL):
                    print("Cycle:", self.env.now, " ", end='')
                    ins.printInstr(self.threadId)
                l = self.tensixFunc.buildInsFromMop(ins)

                i = 0
                for ll in l:
                    if(type(ll) is list):
                        for lll in ll:
                            self.ttInstr.append(lll)
                            decodedInsDec = decodeInstr(lll, ins.kind, True, ttISA = self.args_dict['ttISA'])
                            decodedIns  = instr(decodedInsDec)
                            decodedIns.setRelAddr(ins.getRelAddr())
                            decodedIns.setCoreId(self.coreId)
                            decodedIns.setThread(ins.getThread())
                            self.instrBuff.put(decodeInstr(lll, ins.kind, True, ttISA = self.args_dict['ttISA']))
                            if(self.debug & DEBUG_TENSIX_MED_LEVEL):     decodedIns.printInstr(self.threadId)
                            yield self.env.timeout(1)
                    else:
                        self.ttInstr.append(ll)
                        decodedInsDec = decodeInstr(ll, ins.kind, True, ttISA = self.args_dict['ttISA'])
                        decodedIns  = instr(decodedInsDec)
                        decodedIns.setRelAddr(ins.getRelAddr())
                        decodedIns.setCoreId(self.coreId)
                        decodedIns.setThread(ins.getThread())
                        if(self.debug & DEBUG_TENSIX_MED_LEVEL):     print(f"MOP: Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{ins.getThread()} ", end=''); print(decodedIns.printInstr(self.threadId))
                        self.instrBuff.put(decodedIns)
                        yield self.env.timeout(1)
                    i += 1
                if(self.debug & DEBUG_TENSIX_MED_LEVEL):
                    print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{ins.getThread()} MOP Decode completed. Total TT Instructions generated={i} from MOP Instruction")
            elif(ins.isTT()):
                self.instrBuff.put(ins)
            else:
                print("Expected TT instruction. Received", print(ins))
            yield self.env.timeout(2)
            self.prevPC = self.pc
            self.pc = self.prevPC + 4

    def _stall_dst_pipes(self, pipeIns):
        # Stall destination pipes
        pipeCondCheck0 = []
        if pipeIns.getOp() in freeDstPipesforInstrs:
            # for SEMGET/POST/INIT instructions the destination pipes are already stalled. We only check that they are indeed stalled. Here we wait until they are stalled, instead of one time check.
            # check that all dst pipes are stalled.
            for i in range(len(pipeIns.getDstPipes())):
                pipeCondCheck0.append(self.env.process(
                self.rState.checkRsrcState(pipeIns.getDstPipes()[i], pipeIns.getPipesThreadId(), 1, self.debug, 2, instr_info = f"@tensixPipe. cycle: {self.env.now}. check if dst pipes (i = {i}, pipe id = {pipeIns.getDstPipes()[i]}) are busy for instruction: {pipeIns}"))
                )
        else:
            # for other instructions we stall the dst pipes
            # Stall Dst Pipes
            for i in range(len(pipeIns.getDstPipes())):
                pipeCondCheck0.append(self.env.process(
                    self.rState.setRsrcState(pipeIns.getDstPipes()[i], pipeIns.getPipesThreadId(), 1, self.debug, instr_info = f"@tensiPipe. cycle: {self.env.now}. set dst pipe (i = {i}, pipe id = {pipeIns.getDstPipes()[i]}) busy for instruction: {pipeIns}"))
                    )
                if(self.debug & DEBUG_RISC_HIGH_LEVEL):
                    print(f"Cycle:{self.env.now}, Addr: {hex(pipeIns.getRelAddr())}, Thread{pipeIns.getThread()}: Trying to stall pipe[{pipeIns.getDstPipes()[i]}] on pipe thread ID = {pipeIns.getPipesThreadId()} as part of {pipeIns.mnemonic}, Len of pipeCondCheck0 = {len(pipeCondCheck0)}, Len of dstPipes = {len(pipeIns.getDstPipes())}. pipeInstr_info: {pipeIns}")

        yield simpy.events.AllOf(self.env, pipeCondCheck0)
        if (self.debug & DEBUG_TENSIX_MED_LEVEL) and (len(pipeIns.getDstPipes()) > 0):
            print("Cycle:{2} Thread[{0}]: StallPipe set for {1} , Val={3}".format(pipeIns.getThread(), pipeIns.mnemonic, self.env.now, self.rState.readRsrcState(pipeIns.getDstPipes()[0], pipeIns.getPipesThreadId())))

    def _check_src_pipes(self, pipeIns):
        # Check source pipes are free
        pipeCondCheck1      = []
        for i in range(len(pipeIns.getSrcPipes())):
            pipeCondCheck1.append(self.env.process(
                self.rState.checkRsrcState(pipeIns.getSrcPipes()[i], pipeIns.getPipesThreadId(), 0, self.debug, 2, instr_info = f"@tensiPipe. cycle: {self.env.now}. check if src pipes (i = {i}, pipe id = {pipeIns.getSrcPipes()[i]}) are free for instruction: {pipeIns}"))
            )
            if(self.debug & DEBUG_TENSIX_MED_LEVEL):                      print(f"Cycle:{self.env.now} Addr:{4},Thread[{0}]: Waiting for pipe[{1}] thread [{6}] as part of {2}, Len of pipeCondCheck1={4}, Len of srcPipes={5}".format(pipeIns.getThread(), pipeIns.getSrcPipes()[i],  pipeIns.mnemonic, self.env.now, len(pipeCondCheck1), len(pipeIns.getSrcPipes()), hex(pipeIns.getRelAddr()), pipeIns.getPipesThreadId()))

        yield simpy.events.AllOf(self.env, pipeCondCheck1)
        if (self.debug & DEBUG_TENSIX_MED_LEVEL) and (len(pipeIns.getSrcPipes()) > 0):
            print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} Thread{pipeIns.getThread()}: WaitRes completed for {3}".format(self.env.now, hex(pipeIns.getRelAddr()), pipeIns.getThread(), pipeIns.mnemonic))

    def _wait_on_exe_pipe(self, pipeIns):
        # Wait on execution pipe to be free
        exe_pipe_id = self.pipes.index(self.targetResource(pipeIns))
        if (exe_pipe_id in pipeIns.getDstPipes()) and (pipeIns.getThread() == pipeIns.getPipesThreadId()):
            if (self.debug & DEBUG_TENSIX_MED_LEVEL):
                msg  = f"Cycle:{self.env.now}. @tensixPipe. instr_info: {pipeIns}\n"
                msg += f"- execution pipe id for instruction {pipeIns.getOp()} is {exe_pipe_id}\n"
                msg += f"- stalled dst pipe ids are: {pipeIns.getDstPipes()}\n"
                msg += f"- pipe thread id for instruction {pipeIns.getOp()} is {pipeIns.getPipesThreadId()}\n"
                msg += f"- as execution pipe and thread is included in dst pipes and corresponding thread, we skip the following:\n"
                msg += f"- skip checking if execution pipe if free for the insutruction\n"
                msg += f"- do not set execution pipe busy (as it is already busy)"
                msg += f"- dst resource state:\n"
                for i in range(len(pipeIns.getDstPipes())):
                    msg += f"  - pipe id: {pipeIns.getDstPipes()[i]}, current status: {self.rState.readRsrcState(pipeIns.getDstPipes()[i], pipeIns.getThread())}\n"

                print(msg.rstrip())
        else:
            # Wait on Exe Pipe Free
            yield(self.env.process(self.rState.checkRsrcState(exe_pipe_id, pipeIns.getThread(), 0, self.debug, instr_info = f"@tensixPipe. cycle: {self.env.now}. check if exe pipe (id = {exe_pipe_id}) are free for instruction: {pipeIns}")))
            if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} pipe:{5} Available.".format(pipeIns.mnemonic, pipeIns.getInsId(), pipeIns.getThread(), hex(pipeIns.getRelAddr()), self.env.now, pipeIns.getExPipe()))

            # Exe Pipe Busy
            yield self.env.process(self.rState.setRsrcState(exe_pipe_id, pipeIns.getThread(), 1, self.debug, instr_info = f"@tensiPipe. cycle: {self.env.now}. set exe pipe (id = {exe_pipe_id}) busy for instruction: {pipeIns}"))
            if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} pipe:{5} Set Busy.".format(pipeIns.mnemonic, pipeIns.getInsId(), pipeIns.getThread(), hex(pipeIns.getRelAddr()), self.env.now, pipeIns.getExPipe()))

    #TODO: Merge both _stall_and_check_pipes functions
    def _stall_and_check_pipes(self, pipeIns):
    # Handles pipe stall and resource checks
    # Reuse for both barrier and non-barrier instructions
        # Stall Dst Pipes
        yield from self._stall_dst_pipes(pipeIns)
        # Check Src Pipes
        yield from self._check_src_pipes(pipeIns)
        # Wait on Exe Pipe Free
        yield from self._wait_on_exe_pipe(pipeIns)

    #TODO: Merge both _check_valids functions
    def _check_valids(self, pipeIns):
        # Handles valid/inUse checks for src/dst registers
        validCondCheck          = []

        srcList = []
        dstList = []
        mode    =  0
        # Wait on Src/Dst Valids
        if(len(pipeIns.getSrcInt()) != 0):
            srcList = pipeIns.getSrcInt()
            if(pipeIns.mnemonic in skipValidCheckOps):    mode = 2    # Skip Valid , set inUse to stall following instructions post stallwait, cleardvalid [Special Case]
            else:                                       mode = 1    # Check Valid , Don't set inUse as its a source
            assert pipeIns.getOp() != "SETDVALID" , "SETDVALID not supported"
            for i in range(len(srcList)):
                if srcList[i] not in skipValidCheckforRegs:
                    if (self.ttThreadReg.condCheckValid[srcList[i]][pipeIns.getContext()] == -1) or ((self.ttThreadReg.regIndex.dstIndex.value == srcList[i]) and (not self.tensixFunc.tensixSplRegs.isDstRegProgrammed())):
                        print(f"WARNING: Source register {srcList[i]} condCheckValid in context {pipeIns.getContext()} is not programmed. This may lead to deadlock.")
                    else:
                        validCondCheck.append(self.env.process(
                            self.ttThreadReg.checkValid(srcList[i], pipeIns.getContext(), self.ttThreadReg.condCheckValid[srcList[i]][pipeIns.getContext()], mode, self.debug))
                            )

        if(len(pipeIns.getDstInt()) != 0):
            dstList = pipeIns.getDstInt()
            if(pipeIns.mnemonic in skipValidCheckOps):    mode = 2    # Skip Valid , set inUse as its a dst
            else:                                       mode = 3    # Check Valid ,set inUse as its a dst
            assert pipeIns.getOp() != "SETDVALID" , "SETDVALID not supported"
            for i in range(len(dstList)):
                if dstList[i] not in skipValidCheckforRegs:
                    if (self.ttThreadReg.condCheckValid[dstList[i]][pipeIns.getContext()] == -1) or ((self.ttThreadReg.regIndex.dstIndex.value == dstList[i]) and (not self.tensixFunc.tensixSplRegs.isDstRegProgrammed())):
                        print(f"WARNING: Destination register {dstList[i]} condCheckValid in context {pipeIns.getContext()} is not programmed. This may lead to deadlock.")
                    else:
                        validCondCheck.append(self.env.process(
                            self.ttThreadReg.checkValid(dstList[i], pipeIns.getContext(), self.ttThreadReg.condCheckValid[dstList[i]][pipeIns.getContext()], mode, self.debug))
                            )

        if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} CheckValids Condition (inprogress)(barrier) registers:{dstList},{srcList}")
        checkTime = self.env.now
        if(len(validCondCheck) >0 ):    yield simpy.events.AllOf(self.env, validCondCheck)
        if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} CheckValids Condition (done)(barrier) registers:{dstList},{srcList}, StallTime:{self.env.now - checkTime}")

    def tArbiter(self, ttFifo, ttBuffer, replayState, insROB, traceEventList):
        while(True):
            ################# Replay ###################
            currRMode = replayState[self.threadId].getRMode()
            #0: PassThrough, #1: Load, #2: Load/Execute, #3: Replay Execute
            match currRMode:
                case 0:  #PassThrough
                    # Get Instruction from Instruction Buffer
                    ins = yield self.instrBuff.get()
                    assert ins.isTT() , "Expected TT instruction. Received" +  ins.mnemonic
                    if(self.debug & DEBUG_TENSIX_MED_LEVEL):        print("Instruction = ", ins.mnemonic, ", RMode=", currRMode)

                    if(ins.isReplay()):
                        replayAttribs         = self.tensixFunc.__execreplay__(ins)
                        replayAttribs.append(ins.getThread())

                        yield self.env.process(replayState[ins.getThread()].updateRMode(replayAttribs, self.debug))
                        if(self.debug & DEBUG_TENSIX_MED_LEVEL):    print("REPLAY [{replayAttribs[0]},{replayAttribs[1]},{replayAttribs[2]},{replayAttribs[3]}]")
                        if(self.debug & DEBUG_TENSIX_MED_LEVEL):    print(f"Cycle:{self.env.now} RMode(Replay - Passthrough) changed from {replayState[ins.getThread()].getPrevRMode()} to {replayState[ins.getThread()].getRMode()}")
                        if(self.debug & DEBUG_TENSIX_MED_LEVEL):    print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{ins.getThread()} Ins:{ins.getOp()}")

                        #Commit Replay Instruction
                        # Removing latency after Replay
                        # yield self.env.timeout(1)
                        continue
                    else:
                        nextAddr  = self.tensixFunc.execTTIns(ins, self.env.now)
                        if ins.getOp() in opHist.keys():                            opHist[ins.getOp()] += 1
                        else:                                                       opHist[ins.getOp()] = 1
                        if(self.debug & DEBUG_TENSIX_MED_LEVEL):    print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} Ins:{ins.mnemonic} (PassThrough) ")

                case 1|2:  #Load
                    # Get Instruction from Instruction Buffer
                    ins = yield self.instrBuff.get()
                    assert ins.isTT() , "Expected TT instruction. Received" +  ins.mnemonic
                    if(self.debug & DEBUG_TENSIX_MED_LEVEL):        print("Instruction(Load/Load+Exec) = ", ins.mnemonic, ", RMode=", currRMode)

                    # Load Instruction to Replay Buffer
                    yield self.env.process(replayState[ins.getThread()].loadReplayList(ins, self.debug))

                    # Exec Ins / Put in ttFifo
                    assert not ins.isReplay() , "Can't have Replay instruction when Replay Mode is Load/Load+Execute"
                    if(currRMode == 2):   # Load and Execute
                        nextAddr   = self.tensixFunc.execTTIns(ins, self.env.now)
                        if ins.getOp() in opHist.keys():                            opHist[ins.getOp()] += 1
                        else:                                                       opHist[ins.getOp()] = 1
                        if(self.debug & DEBUG_TENSIX_MED_LEVEL):    print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} Ins:{ins.mnemonic} (Load+Execute) ")
                    elif(currRMode == 1): #Skip Valids Check and fwd it directly
                        if(self.debug & DEBUG_TENSIX_MED_LEVEL):    print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{self.threadId} Ins:{ins.mnemonic} (Load) ")
                        # Removing latency after Replay
                        # yield self.env.timeout(1)
                        continue

                case 3:
                    # Get Instruction from Replay Buffer
                    ins = copy.deepcopy(replayState[ins.getThread()].execReplayList(self.debug))
                    assert ins.isTT() , "Expected TT instruction. Received" +  ins.mnemonic
                    if(self.debug & DEBUG_TENSIX_MED_LEVEL):        print("Thread[",ins.getThread(),"] Instruction(Exec) = ", ins.mnemonic, ", RMode=", currRMode)

                    assert not ins.isReplay() , "Can't have Replay instruction when Replay Mode is Execute"
                    if(not ins.isReplay()):          nextAddr   = self.tensixFunc.execTTIns(ins, self.env.now)
                    if ins.getOp() in opHist.keys():                            opHist[ins.getOp()] += 1
                    else:                                                       opHist[ins.getOp()] = 1

                    if(self.debug & DEBUG_TENSIX_MED_LEVEL):  print(f"Cycle:{self.env.now} Replay Executing from replayList {ins.printInstr(ins.getThread())}")

            assert currRMode !=1 , "Replay Mode = 1 (Load) should bypass valid / resource checks"
            assert not ins.isReplay() , "Replay instruction should have been committed"
            ################# Replay ###################

            ins.setExPipe(self.targetResource(ins))

            insId = yield self.env.process(insROB[ins.getThread()].appendRob(ins, self.debug))
            ins.setInsId(insId)

            # Removing latency for insertion into ROB
            # yield self.env.timeout(1)

            # Stall Remove from ROB - Barrier instructions
            if(not enableInOrderIssue and
                ( (ins.getExPipe() in ["SYNC", "TDMA", "THCON", "CFG", "INSTISSUE"]))
            ):
                if(self.debug & DEBUG_TENSIX_MED_LEVEL):      print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{ins.getThread()} insId{ins.getInsId()} Instruction:{ins.getOp()} Forcing InOrder (inProgress)")
                yield self.env.process(insROB[ins.getThread()].headOfRob(ins.getInsId(), self.debug, f"cycle: {self.env.now}, call from tArbiter. instruction: {ins}"))
                if(self.debug & DEBUG_TENSIX_MED_LEVEL):       print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{ins.getThread()} insId{ins.getInsId()} Instruction:{ins.getOp()} Forcing InOrder (done)      ")

            # Barrier Instructions or InOrder Issue.
            syncpipesList = ["SYNC", "TDMA", "THCON", "CFG", "INSTISSUE"]
            checkInOrderTrue = enableInOrderIssue or (not enableInOrderIssue and ( ins.getExPipe() in syncpipesList))

            # 1a. Pipe/Resource Stall Logic
            if checkInOrderTrue and enablePipeStall and enableStallWait: yield from self._stall_and_check_pipes(ins)

            # 2a. Valid/InUse Checks
            if('enableSync' in self.args_dict):     enableSync = self.args_dict['enableSync']
            if(checkInOrderTrue and enableSync): yield from self._check_valids(ins)

            self.ttInstructions += 1

            ## Route to Pipe Buffer
            if(ins.getExPipe() != None):
                if(ins.getExPipe() == "NONE"):
                    # TODO: Confirm if remove from ROB is enough
                    beginEvent = SIMTRACE("TENSIX", ins.getExPipe(), self.env.now, 'B', ins.mnemonic, ins.getAttr())
                    if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{ins.getThread()} insId{ins.getInsId()} Instruction:{ins.getOp()} Sending (inprogress)(barrier) to pipe {ins.getExPipe()}")
                    yield self.env.process(insROB[ins.getThread()].removeRob(ins.getInsId(), self.debug))
                    endEvent = SIMTRACE('TENSIX', ins.getExPipe(), self.env.now, 'E', ins.mnemonic, ins.getAttr())
                    traceEventList.append(beginEvent)
                    traceEventList.append(endEvent)
                else:
                    if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{ins.getThread()} insId{ins.getInsId()} Instruction:{ins.getOp()} ObjectId:{hex(id(ins))} Sending (inprogress)(barrier) to pipe {ins.getExPipe()} ttBufferLen[{ins.getExPipe()}]={len(ttBuffer[ins.getExPipe()].items)}")
                    assert ins.getCoreId() == self.coreId , f"Cycle:{self.env.now} CoreId mismatch {ins.getCoreId()} != {self.coreId}, Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{ins.getThread()} Pipe:{self.pipes[ins.getExPipe()]} Instruction={ins.getOp()}"
                    yield ttBuffer[ins.getExPipe()].put(ins)
                    # print(f"TTBuffer={ttBuffer[ins.getExPipe()].printName()}")
                    if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{ins.getThread()} insId{ins.getInsId()} Instruction:{ins.getOp()} ObjectId:{hex(id(ins))} Sending (done)(barrier)       to pipe {ins.getExPipe()} ttBufferLen[{ins.getExPipe()}]={len(ttBuffer[ins.getExPipe()].items)}")
            else:
                assert False , "WARNING. Unmapped instruction {0} reached instruction issue [{1}]".format(ins.mnemonic, ins.getExPipe())


            if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(ins.getRelAddr())} TCore{self.coreId} Thread{ins.getThread()} insId{ins.getInsId()} Instruction:{ins.getOp()} ObjectId:{hex(id(ins))} Targeting (done)(barrier)       pipe {ins.getExPipe()} ttBufferLen[{ins.getExPipe()}]={len(ttBuffer[ins.getExPipe()].items)}")
            yield self.env.timeout(1)

    def targetResource(self, ins):
        targetEngine = []
        pipe = None

        #Check if Kernel sets the pipe.
        if(ins.getExPipe() != None):
            pipe = ins.getExPipe()
            # Check for pipe name in cfg
            assert pipe in self.pipes, "Unknown execution pipe " + pipe + " set in kernel"

        #Static Pipe Assignment. One or many options
        for i in range(len(self.args_dict['engines'])):
            tptDict = self.args_dict['engines'][i]['engineInstructions']
            for j in range(len(tptDict)):
                for k,v in tptDict[j].items():
                    if(tptDict[j][k] == ins.mnemonic):
                        if(self.debug & DEBUG_TENSIX_HIGH_LEVEL):  print ("Match : ", tptDict[j]['name'] , ins.mnemonic)
                        targetEngine.append(self.args_dict['engines'][i]['engineName'])
                        #TODO: Remove hardcoding of format
                        ins.setPipeDelay(tptDict[j]['tpt']['int32'])

        if(self.debug & DEBUG_TENSIX_HIGH_LEVEL):  print("Instruction = ", ins.mnemonic, "Engine = ", targetEngine)

        #Checks applied before return
        if(pipe != None):
            if(len(targetEngine) > 0):
                assert pipe in targetEngine, "Kernel setting pipe as " + str(pipe) + " but not legal as per static assignment" + str(ins.getOp())
                return pipe
        else:
            if(len(targetEngine) == 0):
                ins.printInstr(ins.getThread())
                assert False, "Engine mapping missing for "+ str(ins.mnemonic)
            elif(len(targetEngine) == 1):
                pipe = targetEngine[0]
                return pipe
            else:
                for i in range(len(targetEngine)):
                    print(targetEngine[i])
                assert False, "Too many resources to select from: " + str(len(targetEngine)) + ", " + str(ins.printInstr(ins.getThread()))

class tensixCore:
    UNPACKERFE_DELAY_CYCLES     = 3
    PACKERFE_DELAY_CYCLES       = 3
    DEFAULT_INPUTFIFO_CAPACITY  = 2
    DEFAULT_L1_PORT_WIDTH       = 128
    DEFAULT_REG_PORT_WIDTH      = 256

    def __init__(self,env, args_dict, coreId, l1IBuffer, l1OBuffer):
        self.threadData     = []
        self.triscRegs      = []
        self.thread         = []
        self.ttBuffer       = {}
        self.insByEngine    = {}
        self.numTotalL1Req  = 0
        self.numTotalRegReq = 0
        self.numTotalL1Bytes  = 0
        self.numTotalRegBytes = 0
        self.ttAutoloopBuffer = {}
        self.ttAutoloopPipes = ["PACKER1", "UNPACKER2"]

        self.name       = "tensixCore_" + str(coreId)
        self.env        = env
        self.debug      = args_dict['debug']
        self.args_dict  = args_dict
        self.coreId     = coreId
        self.numThreads = args_dict['input']['tc' + str(self.coreId)]['numThreads']

        self.memData    = triscFunc.triscMemFunc(self.args_dict)
        self.ttReg      = ttReg(self.env, args_dict, self.coreId, self.numThreads)
        self.riscReg    = riscRegState(self.env, args_dict, self.coreId, self.numThreads)
        self.insROB     = []

        self.tensixReplayState = []
        for i in range(self.numThreads):
            self.tensixReplayState.append(replayState(self.env, self.coreId, i))
            self.insROB.append(rob(self.env,self.coreId, i))

        # To track synchronization commits usually via cleardvalid
        self.commWriVld       = []
        for i in range(self.numThreads):
            self.commWriVld.append(0)

        self.rState = pipeResource(self.env, self.args_dict, self.coreId)

        self.pipes  =   []
        self.pipeGrps   = defaultdict(list)
        self.delay  =   []

        self.ttFifo = simpy.Store(self.env, capacity=self.DEFAULT_INPUTFIFO_CAPACITY)

        self.trace_event_list = []

        for i in range(len(self.args_dict['engines'])):
            pipeGrp = self.args_dict['engines'][i]['engineGrp']
            pipe    = self.args_dict['engines'][i]['engineName']
            self.pipes.append(pipe)
            self.pipeGrps[pipeGrp].append(pipe)

            # self.ttBuffer[pipe] = simpy.Store(self.env, capacity=10)
            self.ttBuffer[pipe] = namedStore("TCore" + str(self.coreId) + "_Pipe" + str(pipe), self.env, capacity=10)

            self.delay.append(self.args_dict['engines'][i]['delay'])
            self.insByEngine[pipe]  = 0

        print("PipeGrps=", self.pipeGrps)
        self.tensixSplRegs = tensixFunc.ttSplRegs(self.coreId, self.args_dict)
        self.tensixData = tensixFunc.tensixFunc(self.coreId, self.memData, self.args_dict, self.pipeGrps, self.pipes, self.tensixSplRegs, self.triscRegs)
        [print(idx, pipe) for idx, pipe in enumerate(self.pipes)]

        self.ttAutoloopBuffer = {pipe : namedStore(
            "ttAutoloopBuffer_TCore" + str(self.coreId) + "_Pipe" + str(pipe),
            self.env,
            capacity = max(self.tensixSplRegs.getMaxNumUnpacker2Packer1AutoloopIterations(), 1000)) for pipe in self.pipes}

        #Threads - Process
        for i in range(self.args_dict['input']['tc' + str(self.coreId)]['numThreads']):
            if self.args_dict['input']['tc' + str(self.coreId)]['th' + str(i) + 'Elf'] == '':
                if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"Skipping Thread[{str(i)}]")
            else:
                # thread --> threadData[len(self.threadData)-1] , thread is mapped to the most recently created functional object to handle skipping of threads
                self.triscRegs.append(triscFunc.triscRegs(self.coreId, i, self.args_dict))
                self.threadData.append(triscFunc.triscFunc(self.coreId, i, self.memData, self.args_dict, self.tensixSplRegs, self.triscRegs[len(self.triscRegs)-1]))
                self.thread.append(thread(self.args_dict, self.env,  self.args_dict['input']['tc' + str(self.coreId)]['startFunction'], self.coreId, i , self.threadData[len(self.threadData)-1], self.tensixData, self.ttFifo, self.ttBuffer, self.rState, self.ttReg, self.riscReg, self.pipes, self.tensixReplayState, self.insROB, self.trace_event_list))

        # Pipes - Process
        self.tensixPipeTrk = []
        self.sourceRdReqTrk  = []
        self.sourceWrReqTrk  = []
        if self.debug & DEBUG_TENSIX_MED_LEVEL: print("len(l1IBuffer)=", len(l1IBuffer), ",len(self.pipes)=", len(self.pipes))
        if self.debug & DEBUG_TENSIX_MED_LEVEL: print("len(l1OBuffer)=", len(l1IBuffer), ",len(self.pipes)=", len(self.pipes))
        assert len(l1IBuffer) == len(self.pipes) , "There isn't as many number of L1IBuffers as tensixPipes"
        assert len(l1OBuffer) == len(self.pipes) , "There isn't as many number of L1OBuffers as tensixPipes"
        for i in range(len(self.pipes)):
            if (self.pipes[i] in self.pipeGrps['UNPACK']):
                self.env.process(self.unpacker(self.args_dict, coreId, i, l1IBuffer[i], l1OBuffer[i]))
            elif (self.pipes[i] in self.pipeGrps['PACK']):
                self.env.process(self.packer(self.args_dict, coreId, i, l1IBuffer[i], l1OBuffer[i]))
            else:
                self.env.process(self.tensixPipe(self.args_dict, coreId, i, l1IBuffer[i], l1OBuffer[i]))
            self.tensixPipeTrk.append({})
            self.sourceRdReqTrk.append({})
            self.sourceWrReqTrk.append({})

        #Register Buffers Per Pipe - Specifically perf unpacker / packer
        #TODO: Support multi-port, multi-reg writes.
        self.regIBuffer = []
        self.regOBuffer = []
        for i in range(args_dict['numTCores']):
            self.regIPerPipeBuffer    = []
            self.regOPerPipeBuffer    = []
            for j in range(len(args_dict['engines'])):
                self.regIPerPipeBuffer.append(simpy.Store(env, capacity=1))
                self.regOPerPipeBuffer.append(simpy.Store(env, capacity=1))
            self.regIBuffer.append(self.regIPerPipeBuffer)
            self.regOBuffer.append(self.regOPerPipeBuffer)

        self._insTrk = {}
        self.reqTrk1 = {}

        if self.debug & DEBUG_TENSIX_MED_LEVEL:
            print("Starting autoloop buffer population processes for all pipes")

        if (self.args_dict['enableAutoLoop']):
            for i in range(len(self.pipes)):
                self.env.process(self._add_instructions_to_ttAutoloopBuffer(i))

        print("Construction Completed")

    def _stall_dst_pipes(self, pipeIns):
        # Stall destination pipes
        pipeCondCheck0 = []
        if pipeIns.getOp() in freeDstPipesforInstrs:
            # for SEMGET/POST/INIT instructions the destination pipes are already stalled. We only check that they are indeed stalled. Here we wait until they are stalled, instead of one time check.
            # check that all dst pipes are stalled.
            for i in range(len(pipeIns.getDstPipes())):
                pipeCondCheck0.append(self.env.process(
                self.rState.checkRsrcState(pipeIns.getDstPipes()[i], pipeIns.getPipesThreadId(), 1, self.debug, 2, instr_info = f"@tensixPipe. cycle: {self.env.now}. check if dst pipes (i = {i}, pipe id = {pipeIns.getDstPipes()[i]}) are busy for instruction: {pipeIns}"))
                )
        else:
            # for other instructions we stall the dst pipes
            # Stall Dst Pipes
            for i in range(len(pipeIns.getDstPipes())):
                pipeCondCheck0.append(self.env.process(
                    self.rState.setRsrcState(pipeIns.getDstPipes()[i], pipeIns.getPipesThreadId(), 1, self.debug, instr_info = f"@tensiPipe. cycle: {self.env.now}. set dst pipe (i = {i}, pipe id = {pipeIns.getDstPipes()[i]}) busy for instruction: {pipeIns}"))
                    )
                if(self.debug & DEBUG_RISC_HIGH_LEVEL):
                    print(f"Cycle:{self.env.now}, Addr: {hex(pipeIns.getRelAddr())}, Thread{pipeIns.getThread()}: Trying to stall pipe[{pipeIns.getDstPipes()[i]}] on pipe thread ID = {pipeIns.getPipesThreadId()} as part of {pipeIns.mnemonic}, Len of pipeCondCheck0 = {len(pipeCondCheck0)}, Len of dstPipes = {len(pipeIns.getDstPipes())}. pipeInstr_info: {pipeIns}")

        yield simpy.events.AllOf(self.env, pipeCondCheck0)
        if (self.debug & DEBUG_TENSIX_MED_LEVEL) and (len(pipeIns.getDstPipes()) > 0):
            print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} Thread{pipeIns.getThread()}: StallPipe completed for {pipeIns.mnemonic}")

    def _check_src_pipes(self, pipeIns):
        # Check source pipes are free
        pipeCondCheck1      = []
        for i in range(len(pipeIns.getSrcPipes())):
            pipeCondCheck1.append(self.env.process(
                self.rState.checkRsrcState(pipeIns.getSrcPipes()[i], pipeIns.getPipesThreadId(), 0, self.debug, 2, instr_info = f"@tensiPipe. cycle: {self.env.now}. check if src pipes (i = {i}, pipe id = {pipeIns.getSrcPipes()[i]}) are free for instruction: {pipeIns}"))
            )
            if(self.debug & DEBUG_TENSIX_MED_LEVEL):                      print(f"Cycle:{self.env.now} Addr:{4},Thread[{0}]: Waiting for pipe[{1}] thread [{6}] as part of {2}, Len of pipeCondCheck1={4}, Len of srcPipes={5}".format(pipeIns.getThread(), pipeIns.getSrcPipes()[i],  pipeIns.mnemonic, self.env.now, len(pipeCondCheck1), len(pipeIns.getSrcPipes()), hex(pipeIns.getRelAddr()), pipeIns.getPipesThreadId()))

        yield simpy.events.AllOf(self.env, pipeCondCheck1)
        if (self.debug & DEBUG_TENSIX_MED_LEVEL) and (len(pipeIns.getSrcPipes()) > 0):
            print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} Thread{pipeIns.getThread()}: WaitRes completed for {3}".format(self.env.now, hex(pipeIns.getRelAddr()), pipeIns.getThread(), pipeIns.mnemonic))

    def _wait_on_exe_pipe(self, pipeIns):
        # Wait on execution pipe to be free
        exe_pipe_id = self.pipes.index(self.targetResource(pipeIns))
        if (exe_pipe_id in pipeIns.getDstPipes()) and (pipeIns.getThread() == pipeIns.getPipesThreadId()):
            if (self.debug & DEBUG_TENSIX_MED_LEVEL):
                msg  = f"Cycle:{self.env.now}. @tensixPipe. instr_info: {pipeIns}\n"
                msg += f"- execution pipe id for instruction {pipeIns.getOp()} is {exe_pipe_id}\n"
                msg += f"- stalled dst pipe ids are: {pipeIns.getDstPipes()}\n"
                msg += f"- pipe thread id for instruction {pipeIns.getOp()} is {pipeIns.getPipesThreadId()}\n"
                msg += f"- as execution pipe and thread is included in dst pipes and corresponding thread, we skip the following:\n"
                msg += f"- skip checking if execution pipe if free for the insutruction\n"
                msg += f"- do not set execution pipe busy (as it is already busy)"
                msg += f"- dst resource state:\n"
                for i in range(len(pipeIns.getDstPipes())):
                    msg += f"  - pipe id: {pipeIns.getDstPipes()[i]}, current status: {self.rState.readRsrcState(pipeIns.getDstPipes()[i], pipeIns.getThread())}\n"

                print(msg.rstrip())
        else:
            # Wait on Exe Pipe Free
            yield(self.env.process(self.rState.checkRsrcState(exe_pipe_id, pipeIns.getThread(), 0, self.debug, instr_info = f"@tensixPipe. cycle: {self.env.now}. check if exe pipe (id = {exe_pipe_id}) are free for instruction: {pipeIns}")))
            if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} pipe:{5} Available.".format(pipeIns.mnemonic, pipeIns.getInsId(), pipeIns.getThread(), hex(pipeIns.getRelAddr()), self.env.now, pipeIns.getExPipe()))

            # Exe Pipe Busy
            yield self.env.process(self.rState.setRsrcState(exe_pipe_id, pipeIns.getThread(), 1, self.debug, instr_info = f"@tensiPipe. cycle: {self.env.now}. set exe pipe (id = {exe_pipe_id}) busy for instruction: {pipeIns}"))
            if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} pipe:{5} Set Busy.".format(pipeIns.mnemonic, pipeIns.getInsId(), pipeIns.getThread(), hex(pipeIns.getRelAddr()), self.env.now, pipeIns.getExPipe()))

    def _stall_and_check_pipes(self, pipeIns):
    # Handles pipe stall and resource checks
    # Reuse for both barrier and non-barrier instructions
        # Stall Dst Pipes
        yield from self._stall_dst_pipes(pipeIns)
        # Check Src Pipes
        yield from self._check_src_pipes(pipeIns)
        # Wait on Exe Pipe Free
        yield from self._wait_on_exe_pipe(pipeIns)

    def _check_valids(self, pipeIns):
        # Handles valid/inUse checks for src/dst registers
        validCondCheck          = []

        srcList = []
        dstList = []
        mode    =  0
        # Wait on Src/Dst Valids
        if(len(pipeIns.getSrcInt()) != 0):
            srcList = pipeIns.getSrcInt()
            if(pipeIns.mnemonic in skipValidCheckOps):  mode = 2    # Skip Valid , set inUse to stall following instructions post stallwait, cleardvalid [Special Case]
            else:                                       mode = 1    # Check Valid , Don't set inUse as its a source
            assert pipeIns.getOp() != "SETDVALID" , "SETDVALID not supported"
            for i in range(len(srcList)):
                if srcList[i] not in skipValidCheckforRegs:
                    if (self.ttReg.condCheckValid[srcList[i]][pipeIns.getContext()] == -1) or ((ttReg.regIndex.dstIndex.value == srcList[i]) and (not self.tensixSplRegs.isDstRegProgrammed())):
                        print(f"WARNING: Source register {srcList[i]} condCheckValid in context {pipeIns.getContext()} is not programmed. This may lead to deadlock.")
                    else:
                        validCondCheck.append(self.env.process(
                            self.ttReg.checkValid(srcList[i], pipeIns.getContext(), self.ttReg.condCheckValid[srcList[i]][pipeIns.getContext()] , mode, self.debug))
                            )

        if(len(pipeIns.getDstInt()) != 0):
            dstList = pipeIns.getDstInt()
            if(pipeIns.mnemonic in skipValidCheckOps):    mode = 2    # Skip Valid , set inUse as its a dst
            else:                                       mode = 3    # Check Valid ,set inUse as its a dst
            assert pipeIns.getOp() != "SETDVALID" , "SETDVALID not supported"
            for i in range(len(dstList)):
                if dstList[i] not in skipValidCheckforRegs:
                    if (self.ttReg.condCheckValid[dstList[i]][pipeIns.getContext()] == -1) or ((ttReg.regIndex.dstIndex.value == dstList[i]) and (not self.tensixSplRegs.isDstRegProgrammed())):
                        print(f"WARNING: Destination register {dstList[i]} condCheckValid in context {pipeIns.getContext()} is not programmed. This may lead to deadlock.")
                    else:
                        validCondCheck.append(self.env.process(
                            self.ttReg.checkValid(dstList[i], pipeIns.getContext(), self.ttReg.condCheckValid[dstList[i]][pipeIns.getContext()], mode, self.debug))
                            )

        if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} CheckValids Condition (inprogress)(nonBarrier) registers:{dstList},{srcList}")
        checkTime = self.env.now
        if(len(validCondCheck) >0 ):    yield simpy.events.AllOf(self.env, validCondCheck)
        if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} CheckValids Condition (done) registers:{dstList},{srcList}, StallTime:{self.env.now - checkTime}")

    """
    def _calcNumReq(self, sI, sO, op):
        match op:
            case "RD":  return math.ceil(sI / sO) # Split
            case "WR":
                if(sO >= sI):   return math.ceil(sO / sI) # Coalesce
                elif(sI > sO):  return math.ceil(sI / sO) # Split
    """

    def _calculate_memory_requests(self, source_bytes, port_width, operation):
        """
        Calculate number of memory requests needed based on operation type.

        Args:
            source_bytes: Total bytes to transfer
            port_width: Memory port width in bytes
            operation: "RD" for read, "WR" for write

        Returns:
            Number of requests needed

        Raises:
            ValueError: For invalid inputs
        """
        if source_bytes <= 0 or port_width <= 0:
            raise ValueError(f"Invalid bytes ({source_bytes}) or port width ({port_width})")

        if operation not in ["RD", "WR"]:
            raise ValueError(f"Unknown operation: {operation}")

        if operation == "RD":
            # For reads, split large requests into port-sized chunks
            return math.ceil(source_bytes / port_width)
        else:  # "WR"
            # For writes, calculate based on coalescing or splitting
            return math.ceil(max(source_bytes, port_width) / min(source_bytes, port_width))

    def _calcNumData(self,s0, s1):         return math.ceil(s1 / s0) # Join

    def _accumulate_request_bytes(self, sReqTrk, target):
        if len(sReqTrk) == 0:
            return [[], 0]
        assert target > 0, f"Cycle:{self.env.now} TCore{self.coreId} Target bytes must be greater than zero"

        accumSrcBytes = 0
        joinList = []
        for req in sReqTrk.values():
            assert req.__getBytes__() > 0, f"Cycle:{self.env.now} TCore{self.coreId} req{req.__getReqId__()} Request has zero bytes"
            accumSrcBytes += req.__getBytes__()
            joinList.append(req.__getReqId__())
            if accumSrcBytes >= target:
                if self.debug & DEBUG_RISC_MED_LEVEL:
                    print(f"Cycle:{self.env.now} TCore{self.coreId} Accumulated bytes {accumSrcBytes} exceeded target {target}, stopping accumulation. Current joinList: {joinList}")
                break
        # accumSrcBytes = sum(req.__getBytes__() for req in sReqTrk.values())

        assert accumSrcBytes > 0, f"Cycle:{self.env.now} TCore{self.coreId} No valid bytes accumulated for request"
        return [joinList, accumSrcBytes]
    

    # reqBuff ----> iBuff/insTrk
    # reqBuff ----> oBuff/insTrk , Short-circuit iBuff --> oBuff path
    def _handle_memory_request(self, reqBuff, iBuff, oBuff,
                        reqTrk, pipeId, overrideOp = None, overrideTarget=None):

        """
        Processes memory/register requests by splitting/coalescing based on port widths.

        Args:
            request_buffer: Source buffer containing memory requests
            input_buffer: Input buffer for L1 requests (shared L1 path) or register requests after split/coalesce
            output_buffer: Output buffer for register requests or direct L1
            request_tracker: tracking in-flight requests
            override_operation: Optional operation override ("RD"/"WR")
            override_target: Optional target override ("L1"/"REG")

        Returns:
            None (runs as SimPy process)
        """
        # sourceReqTrk = {}
        while(True):
            maxTimer    = maxTimerValue = MAXTIMER_LOW
            sourceReq = yield reqBuff.get()
            if self.debug & DEBUG_RISC_HIGH_LEVEL: sourceReq.__printReq__()

            # Update Target in source if needed
            if(overrideTarget != None):
                if self.debug & DEBUG_RISC_HIGH_LEVEL:
                    print(f"Cycle:{self.env.now} TCore{self.coreId} Thread{sourceReq.__getThreadId__()} insId{sourceReq.__getInsId__()} req{sourceReq.__getReqId__()} reqOp:{sourceReq.__getOp__()} Changing Target of Req from {sourceReq.__getTarget__()} to {overrideTarget}")
                sourceReq.__setTarget__(overrideTarget)

            if(overrideOp != None):
                if self.debug & DEBUG_RISC_HIGH_LEVEL:
                    print(f"Cycle:{self.env.now} TCore{self.coreId} Thread{sourceReq.__getThreadId__()} insId{sourceReq.__getInsId__()} req{sourceReq.__getReqId__()} reqOp:{sourceReq.__getOp__()} Target:{sourceReq.__getTarget__()} Changing Op of Req from {sourceReq.__getOp__()} to {overrideOp}")
                sourceReq.__setOp__(overrideOp)

            # Pre-split / pre-coalesce request tracker
            if(sourceReq.__getOp__() == "RD"):
                if self.debug & DEBUG_RISC_MED_LEVEL:
                    print(f"Cycle:{self.env.now} TCore{self.coreId} Thread{sourceReq.__getThreadId__()} insId{sourceReq.__getInsId__()} req{sourceReq.__getReqId__()} reqOp:{sourceReq.__getOp__()} Target:{sourceReq.__getTarget__()} Adding to sourceRdReqTrk[{self.pipes[pipeId]}]")
                self.sourceRdReqTrk[pipeId][sourceReq.__getReqId__()] = sourceReq
            else:
                if self.debug & DEBUG_RISC_MED_LEVEL:
                    print(f"Cycle:{self.env.now} TCore{self.coreId} Thread{sourceReq.__getThreadId__()} insId{sourceReq.__getInsId__()} req{sourceReq.__getReqId__()} reqOp:{sourceReq.__getOp__()} Target:{sourceReq.__getTarget__()} Adding to sourceWrReqTrk[{self.pipes[pipeId]}]")
                self.sourceWrReqTrk[pipeId][sourceReq.__getReqId__()] = sourceReq

            if self.debug & DEBUG_RISC_HIGH_LEVEL:
                print(f"Target: {sourceReq.__getTarget__()} Handling Req{sourceReq.__getReqId__()} for InsId {sourceReq.__getInsId__()} from {sourceReq.__getTarget__()} to {self.pipes[sourceReq.__getSrc__()]} overridden to {overrideTarget}")
            assert sourceReq.__getInsId__() in self.tensixPipeTrk[sourceReq.__getSrc__()], f"Cycle:{self.env.now} TCore{self.coreId} Rsp InsId {sourceReq.__getInsId__()} not found in tensixPipeTrk"

            # Get Source Instruction from tensixPipeTrk based on [source, insId]
            pipeIns = self.tensixPipeTrk[sourceReq.__getSrc__()][sourceReq.__getInsId__()]

            #Set portWidth - TODO: This should be configurable
            l1PortWidth = self.DEFAULT_L1_PORT_WIDTH;   regPortWidth = self.DEFAULT_REG_PORT_WIDTH
            portWidth = -1

            match sourceReq.__getOp__():
                case "RD":
                    #Handle reads to L1 and Register - Extendable to any memory
                    if(sourceReq.__getTarget__() == "L1"):
                        readsSent = "numL1ReadsSent";
                        readsRcvd = "numL1ReadsRcvd";
                        totalReads= "numTotalL1Reads";
                        portWidth = l1PortWidth             #TODO: Make it configurable
                    elif(sourceReq.__getTarget__() == "REG"):
                        readsSent = "numRegReadsSent";
                        readsRcvd = "numRegReadsRcvd";
                        totalReads= "numTotalRegReads";
                        portWidth = regPortWidth            #TODO: Make it configurable
                    else:
                        assert False, f"Unknown target {sourceReq.__getTarget__()}"

                    # Find request(s) meeting targetBytes
                    [reqJoinList, accumSrcRdBytes] = self._accumulate_request_bytes(self.sourceRdReqTrk[pipeId], portWidth)
                    if self.debug & DEBUG_RISC_HIGH_LEVEL:
                        if(len(self.sourceRdReqTrk[pipeId]) > 0):
                            print(f"Cycle:{self.env.now} TCore{self.coreId} Thread{self.sourceRdReqTrk[pipeId][reqJoinList[0]].__getThreadId__()} insId{self.sourceRdReqTrk[pipeId][reqJoinList[0]].__getInsId__()} req{reqJoinList[0]} reqOp:{self.sourceRdReqTrk[pipeId][reqJoinList[0]].__getOp__()} Pipe:{self.pipes[pipeId]} Target:{sourceReq.__getTarget__()} Accumulated Bytes = {accumSrcRdBytes} Target Bytes = {portWidth} Length self.sourceRdReqTrk[pipeId]={len(self.sourceRdReqTrk[pipeId])}, ReqJoinList={reqJoinList}")
                    if(accumSrcRdBytes < portWidth):     continue # Wait for more requests to accumulate
                    if self.debug & DEBUG_RISC_MED_LEVEL:           
                        if(len(self.sourceRdReqTrk[pipeId]) > 0):
                            print(f"Cycle:{self.env.now} TCore{self.coreId} Thread{self.sourceRdReqTrk[pipeId][reqJoinList[0]].__getThreadId__()} Target:{sourceReq.__getTarget__()} Setting number of read requests to {sourceReq.__getTarget__()}={pipeIns.getMemInfo(totalReads)} , NumBytes={accumSrcRdBytes} , PortWidth={portWidth}")

                    accumSentRdBytes = 0
                    while(accumSentRdBytes < accumSrcRdBytes):
                        assert portWidth != -1, "Port Width not set"
                        parents = []

                        #Create Split / Coalesced Request
                        memReq = None
                        memReq = scratchpad.memReq(sourceReq.__getOp__(), 0x0, portWidth)
                        memReq.__setSrc__(sourceReq.__getSrc__())
                        memReq.__setTarget__(sourceReq.__getTarget__())
                        memReq.__setInsId__(sourceReq.__getInsId__())
                        for k in self.sourceRdReqTrk[pipeId].keys():     parents.append(self.sourceRdReqTrk[pipeId][k].__getReqId__())
                        memReq.__setParentReqIds__(parents)
                        memReq.__setThreadId__(sourceReq.__getThreadId__())

                        # L1 - put in iBuff, REG - put in oBuff
                        if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"Target: {memReq.__getTarget__()} Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} pipeInsId:{pipeIns.getInsId()} pipeInstruction:{pipeIns.getOp()} {memReq.__getTarget__()} access (inprogress) at [{self.coreId},{pipeIns.getExPipe()}] from pipe:{pipeIns.getExPipe()}")
                        if(memReq.__getTarget__() == "L1" and self.args_dict['enableSharedL1']):
                            yield iBuff.put(memReq)
                        else:
                            if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} {memReq.__getTarget__()} access initiation (inprogress) from pipe:{pipeIns.getExPipe()} to {memReq.__getTarget__()}")
                            if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} {memReq.__getTarget__()} access initiation (done) from pipe:{pipeIns.getExPipe()} to {memReq.__getTarget__()}")
                            yield oBuff.put(memReq)
                            if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"Cycle:{self.env.now} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} arbitration (done)")

                        reqTrk[memReq.__getReqId__()] = memReq          # Insert into tracker.
                        accumSentRdBytes += portWidth
                        pipeIns.incrMemInfo(readsSent, 1)
                        if(pipeIns.getMemInfo(readsSent) == 1):      pipeIns.setMemInfo("startL1Time", self.env.now)
                        if self.debug & DEBUG_RISC_HIGH_LEVEL:
                            print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} Target: {memReq.__getTarget__()} Inserting Req into tracker. AccumSentRdBytes={accumSentRdBytes}, AccumSrcRdBytes={accumSrcRdBytes}")
                            print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} Target: {memReq.__getTarget__()} Reads Sent {pipeIns.getMemInfo(readsSent)}. Total Reads={pipeIns.getMemInfo(totalReads)}")
                            print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} Target: {memReq.__getTarget__()} Sending request from {pipeIns.getExPipe()}")

                        if accumSentRdBytes == accumSrcRdBytes:
                            if self.debug & DEBUG_RISC_HIGH_LEVEL:
                                print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} Target: {memReq.__getTarget__()} All bytes sent")
                            for i in parents:
                                if self.debug & DEBUG_RISC_HIGH_LEVEL:
                                    print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} Target: {memReq.__getTarget__()} Removing Req{i} from sourceRdReqTrk. AccumSrcRdBytes={accumSrcRdBytes}. AccumSentRdBytes={accumSentRdBytes}")
                                del self.sourceRdReqTrk[pipeId][i]  # Remove from tracker
                            maxTimer = maxTimerValue
                        else:
                            maxTimer = maxTimer - 1
                            if(maxTimer == 0):
                                msg = f"WARNING: Cycle:{self.env.now} TCore{self.coreId} Thread{pipeIns.getThread()} Timeout {maxTimerValue} reached for L1 Req numL1ReadsSent={pipeIns.getMemInfo("numL1ReadsSent")} numTotalL1Reads={pipeIns.getMemInfo("numTotalL1Reads")}"
                                if(self.debug & DEBUG_TENSIX_HIGH_LEVEL):   print(msg); return True
                                assert maxTimer > 0, msg
                        yield self.env.timeout(1)

                    assert accumSentRdBytes == accumSrcRdBytes, f"Cycle:{self.env.now} TCore{self.coreId} Thread{pipeIns.getThread()} Not all bytes sent for L1 Req numL1ReadsSent={pipeIns.getMemInfo("numL1ReadsSent")} numTotalL1Reads={pipeIns.getMemInfo("numTotalL1Reads")}"
                    if self.debug & DEBUG_RISC_HIGH_LEVEL:   print(f"Resetting accumulators")
                    accumSrcRdBytes = 0; accumSentRdBytes = 0
                    if pipeIns.getMemInfo(totalReads) > 0 and (self.debug & DEBUG_RISC_HIGH_LEVEL):
                        print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} Target: {sourceReq.__getTarget__()} All Reads Sent {pipeIns.getMemInfo(readsSent)} == L1 Reads Total {pipeIns.getMemInfo(totalReads)}")

                case "WR":
                    #Handle reads to L1 and Register - Extendable to any memory
                    if(sourceReq.__getTarget__() == "L1"):
                        writesSent = "numL1WritesSent";
                        writesRcvd = "numL1WritesRcvd";
                        totalWrites= "numTotalL1Writes";
                        portWidth = l1PortWidth
                    elif(sourceReq.__getTarget__() == "REG"):
                        writesSent = "numRegWritesSent";
                        writesRcvd = "numRegWritesRcvd";
                        totalWrites= "numTotalRegWrites";
                        portWidth = regPortWidth
                    else:
                        assert False, "Unknown target for L1 Req" + str(sourceReq.__getTarget__())

                    # Find request(s) meeting targetBytes
                    [reqJoinList, accumSrcWrBytes] = self._accumulate_request_bytes(self.sourceWrReqTrk[pipeId], portWidth)
                    if self.debug & DEBUG_RISC_HIGH_LEVEL:
                        if(len(self.sourceWrReqTrk[pipeId]) > 0):
                            print(f"Cycle:{self.env.now} TCore{self.coreId} Thread{self.sourceWrReqTrk[pipeId][reqJoinList[0]].__getThreadId__()} insId{self.sourceWrReqTrk[pipeId][reqJoinList[0]].__getInsId__()} req{reqJoinList[0]} reqOp:{self.sourceWrReqTrk[pipeId][reqJoinList[0]].__getOp__()} Pipe:{self.pipes[pipeId]} Target:{sourceReq.__getTarget__()} Accumulated Bytes = {accumSrcWrBytes} Target Bytes = {portWidth} Length self.sourceWrReqTrk[pipeId]={len(self.sourceWrReqTrk[pipeId])}, ReqJoinList={reqJoinList}")
                    if(accumSrcWrBytes < portWidth):     continue # Wait for more requests to accumulate
                    if self.debug & DEBUG_RISC_MED_LEVEL:           
                        if(len(self.sourceWrReqTrk[pipeId]) > 0):
                            print(f"Cycle:{self.env.now} TCore{self.coreId} Thread{self.sourceWrReqTrk[pipeId][reqJoinList[0]].__getThreadId__()} Target:{sourceReq.__getTarget__()} Setting number of write requests to {sourceReq.__getTarget__()}={pipeIns.getMemInfo(totalWrites)} , NumBytes={accumSrcWrBytes} , PortWidth={portWidth}")

                    assert accumSrcWrBytes > 0, f"Cycle:{self.env.now} TCore{self.coreId} Thread{pipeIns.getThread()} No valid bytes accumulated for {sourceReq.__getTarget__()} request"

                    accumSentWrBytes = 0
                    while(accumSentWrBytes < accumSrcWrBytes):
                        assert portWidth != -1, "Port Width not set"
                        parents = []

                        #Create Split / Coalesced Request
                        memReq = None
                        memReq = scratchpad.memReq(sourceReq.__getOp__(), 0x0, portWidth)
                        memReq.__setSrc__(sourceReq.__getSrc__())
                        memReq.__setTarget__(sourceReq.__getTarget__())
                        memReq.__setInsId__(sourceReq.__getInsId__())
                        for k in self.sourceWrReqTrk[pipeId].keys():     parents.append(self.sourceWrReqTrk[pipeId][k].__getReqId__())
                        memReq.__setParentReqIds__(parents)

                        # L1 - put in iBuff, REG - put in oBuff
                        if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} Target: {sourceReq.__getTarget__()} Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} pipeInsId:{pipeIns.getInsId()} pipeInstruction:{pipeIns.getOp()} Req{memReq.__getReqId__()} {memReq.__getTarget__()} access (inprogress) at [{self.coreId},{pipeIns.getExPipe()}] from pipe:{pipeIns.getExPipe()}")
                        if(memReq.__getTarget__() == "L1" and self.args_dict['enableSharedL1']):
                            yield iBuff.put(memReq)
                        else:
                            if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} {memReq.__getTarget__()} access initiation (inprogress) from pipe:{pipeIns.getExPipe()} to {memReq.__getTarget__()}")
                            if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} {memReq.__getTarget__()} access initiation (done) from pipe:{pipeIns.getExPipe()} to {memReq.__getTarget__()}")
                            yield oBuff.put(memReq)
                            if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"Cycle:{self.env.now} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} arbitration (done)")

                        reqTrk[memReq.__getReqId__()] = memReq          # Insert into tracker.
                        accumSentWrBytes += portWidth
                        pipeIns.incrMemInfo(writesSent, 1)
                        if(pipeIns.getMemInfo(writesSent) == 1):      pipeIns.setMemInfo("startL1Time", self.env.now)
                        if self.debug & DEBUG_RISC_HIGH_LEVEL:
                            print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} Target: {memReq.__getTarget__()} Inserting Req into tracker. AccumSentWrBytes={accumSentWrBytes}, AccumSrcWrBytes={accumSrcWrBytes}")
                            print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} Target: {memReq.__getTarget__()} Writes Sent {pipeIns.getMemInfo(writesSent)}. Total Writes={pipeIns.getMemInfo(totalWrites)}")
                            print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} Target: {memReq.__getTarget__()} Sending request from {pipeIns.getExPipe()}")

                        if accumSentWrBytes == accumSrcWrBytes:
                            if self.debug & DEBUG_RISC_HIGH_LEVEL:
                                print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} Target: {memReq.__getTarget__()} All bytes sent")
                            for i in parents:
                                if self.debug & DEBUG_RISC_HIGH_LEVEL:
                                    print(f"Cycle:{self.env.now} TCore{self.coreId} Req{memReq.__getReqId__()} insId{memReq.__getInsId__()} Target: {memReq.__getTarget__()} Removing Req{i} from sourceWrReqTrk. AccumSrcWrBytes={accumSrcWrBytes}. AccumSentWrBytes={accumSentWrBytes}")
                                del self.sourceWrReqTrk[pipeId][i]  # Remove from tracker
                                maxTimer = maxTimerValue
                        else:
                            maxTimer = maxTimer - 1
                            if(maxTimer == 0):
                                msg = f"WARNING: Cycle:{self.env.now} TCore{self.coreId} Thread{pipeIns.getThread()} Timeout {maxTimerValue} reached for L1 Req numL1ReadsSent={pipeIns.getMemInfo("numL1ReadsSent")} numTotalL1Reads={pipeIns.getMemInfo("numTotalL1Reads")}"
                                if(self.debug & DEBUG_TENSIX_HIGH_LEVEL):   print(msg); return True
                                assert maxTimer > 0, msg
                        yield self.env.timeout(1)

                    assert accumSentWrBytes == accumSrcWrBytes, f"Cycle:{self.env.now} TCore{self.coreId} Thread{pipeIns.getThread()} Not all bytes sent for L1 Req numL1WritesSent={pipeIns.getMemInfo("numL1WritesSent")} numTotalL1Writes={pipeIns.getMemInfo("numTotalL1Writes")}"
                    if self.debug & DEBUG_RISC_HIGH_LEVEL: print(f"Target: {sourceReq.__getTarget__()} Resetting accumulators")
                    accumSrcWrBytes = 0; accumSentWrBytes = 0

                case _:
                    assert False, "L1 Req unknown op " + str(sourceReq.__getOp__())

    # oBuff ----> rspBuff , del insTrk at last req
    def _handle_memory_response(self, rspBuff, oBuff, reqTrk):
        maxTimer    = maxTimerValue = MAXTIMER_LOW
        while(True):
            rsp = yield oBuff.get()
            if self.debug & DEBUG_RISC_HIGH_LEVEL: print(f"Target: {rsp.__getTarget__()} Received Rsp {rsp.__getReqId__()}")

            # Pre-split / coalesce request tracker
            assert rsp.__getInsId__() in self.tensixPipeTrk[rsp.__getSrc__()], f"Cycle:{self.env.now} TCore{self.coreId} Rsp InsId {rsp.__getInsId__()} not found in tensixPipeTrk"

            # Get Source Instruction from tensixPipeTrk based on [source, insId]
            pipeIns = self.tensixPipeTrk[rsp.__getSrc__()][rsp.__getInsId__()]

            yield rspBuff.put(rsp)   # Pass to next stage
            if self.debug & DEBUG_RISC_HIGH_LEVEL: print(f"Target: {rsp.__getTarget__()} Passing Rsp {rsp.__getReqId__()} to next stage")

            match rsp.__getOp__():
                case "RD":
                    if(rsp.__getTarget__() == "L1"):
                        readsSent = "numL1ReadsSent";   readsRcvd = "numL1ReadsRcvd";   totalReads= "numTotalL1Reads"
                    elif(rsp.__getTarget__() == "REG"):
                        readsSent = "numRegReadsSent";   readsRcvd = "numRegReadsRcvd";   totalReads= "numTotalRegReads"
                    else: assert False, "Unknown target for L1 Rsp " + str(rsp.__getTarget__())

                    pipeIns.incrMemInfo(readsRcvd, 1)
                    if self.debug & DEBUG_RISC_HIGH_LEVEL:   print(f"Target: {rsp.__getTarget__()} Received {pipeIns.getMemInfo(readsRcvd)} requests for Req{rsp.__getReqId__()} InsId:{rsp.__getInsId__()}")
                    del reqTrk[rsp.__getReqId__()]
                    if self.debug & DEBUG_RISC_HIGH_LEVEL:
                        print(f"Target: {rsp.__getTarget__()} Removing Req{rsp.__getReqId__()} from tracker")
                        print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} pipeInsId:{pipeIns.getInsId()} pipeInstruction:{pipeIns.getOp()} {rsp.__getTarget__()} access (done) at [{self.coreId},{pipeIns.getExPipe()}] from pipe:{pipeIns.getExPipe()}")
                    if (pipeIns.getMemInfo(readsRcvd) == pipeIns.getMemInfo(totalReads)):
                        if self.debug & DEBUG_RISC_HIGH_LEVEL:   print(f"Target: {rsp.__getTarget__()} All Read Responses Received {pipeIns.getMemInfo(readsRcvd)} == L1 Reads Total {pipeIns.getMemInfo(totalReads)}")
                        if (rsp.__getTarget__() == "L1"):                            pipeIns.setMemInfo("endL1Time", self.env.now)
                        elif (rsp.__getTarget__() == "REG"):                         pipeIns.setMemInfo("endRegTime", self.env.now)
                        else:   assert False, "Unknown target for L1 Rsp " + str(rsp.__getTarget__())
                        if self.debug & DEBUG_RISC_HIGH_LEVEL:   print(f"Target: {rsp.__getTarget__()} Returning InsId {rsp.__getInsId__()}")
                        if self.debug & DEBUG_RISC_HIGH_LEVEL:   print(f"Reset MaxTimer. Current Value={maxTimer}")
                        return rsp.__getInsId__()
                    else:
                        maxTimer = maxTimer - 1
                        if(maxTimer == 0):
                            msg = f"WARNING: Cycle:{self.env.now} TCore{self.coreId} Thread{pipeIns.getThread()} Timeout {maxTimerValue} reached for Rsp numReadsSent={pipeIns.getMemInfo(readsSent)} numTotalReads={pipeIns.getMemInfo(totalReads)}"
                            if(self.debug & DEBUG_TENSIX_HIGH_LEVEL):   print(msg); return True
                            assert maxTimer > 0, msg

                case "WR":
                    if(rsp.__getTarget__() == "L1"):
                        writesSent = "numL1WritesSent";   writesRcvd = "numL1WritesRcvd";   totalWrites= "numTotalL1Writes"
                    elif(rsp.__getTarget__() == "REG"):
                        writesSent = "numRegWritesSent";   writesRcvd = "numRegWritesRcvd";   totalWrites= "numTotalRegWrites"
                    else: assert False, "Unknown target for L1 Rsp " + str(rsp.__getTarget__())
                    pipeIns.incrMemInfo(writesRcvd, 1)
                    if self.debug & DEBUG_RISC_HIGH_LEVEL:   print(f"Target: {rsp.__getTarget__()} Received {pipeIns.getMemInfo(writesRcvd)} requests for Req{rsp.__getReqId__()} InsId:{rsp.__getInsId__()}")
                    del reqTrk[rsp.__getReqId__()]
                    if self.debug & DEBUG_RISC_HIGH_LEVEL:
                        print(f"Target: {rsp.__getTarget__()} Removing Req{rsp.__getReqId__()} from tracker")
                        print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} pipeInsId:{pipeIns.getInsId()} pipeInstruction:{pipeIns.getOp()} {rsp.__getTarget__()} access (done) at [{self.coreId},{pipeIns.getExPipe()}] from pipe:{pipeIns.getExPipe()}")
                    if (pipeIns.getMemInfo(writesRcvd) == pipeIns.getMemInfo(totalWrites)):
                        if self.debug & DEBUG_RISC_HIGH_LEVEL:   print(f"Target: {rsp.__getTarget__()} All Write Responses Received {pipeIns.getMemInfo(writesRcvd)} == Writes Total {pipeIns.getMemInfo(totalWrites)}")
                        if (rsp.__getTarget__() == "L1"):                            pipeIns.setMemInfo("endL1Time", self.env.now)
                        elif (rsp.__getTarget__() == "REG"):                         pipeIns.setMemInfo("endRegTime", self.env.now)
                        else:   assert False, "Unknown target for L1 Rsp " + str(rsp.__getTarget__())
                        if self.debug & DEBUG_RISC_HIGH_LEVEL:   print(f"Target: {rsp.__getTarget__()} Returning InsId {rsp.__getInsId__()}")
                        if self.debug & DEBUG_RISC_HIGH_LEVEL:   print(f"Target: {rsp.__getTarget__()} Reset MaxTimer. Current Value={maxTimer}")
                        return rsp.__getInsId__()
                    else:
                        maxTimer = maxTimer - 1
                        if(maxTimer == 0):
                            msg = f"WARNING: Cycle:{self.env.now} TCore{self.coreId} Thread{pipeIns.getThread()} Timeout {maxTimerValue} reached for Rsp numWritesSent={pipeIns.getMemInfo(writesSent)} numTotalWrites={pipeIns.getMemInfo(totalWrites)}"
                            if(self.debug & DEBUG_TENSIX_HIGH_LEVEL):   print(msg); return True
                            assert maxTimer > 0, msg

                case _:
                    assert False, "L1 Rsp unknown op " + str(rsp.__getOp__())

            yield self.env.timeout(1)

    def _update_valids(self, pipeIns):
        # Handles valid/inUse updates for src/dst registers
        # Update Stalls, Valids
        srcList = []
        dstList = []
        mode    =  0

        if('enableSync' in self.args_dict):     enableSync = self.args_dict['enableSync']

        if(enableSync and self.tensixReplayState[pipeIns.getThread()].getRMode() != 1):
            validCondCheck          = []
            if(len(pipeIns.getSrcInt()) != 0):
                srcList = pipeIns.getSrcInt()
                if(pipeIns.mnemonic in skipValidCheckOps):    mode = 3    # update Valid , reset inUse to allow following instructions post stallwait, cleardvalid [Special Case]
                else:                                           mode = 1    # update Valid , do not reset inUse as its a src
                for i in range(len(srcList)):
                    if srcList[i] not in skipValidCheckforRegs:
                        if (self.ttReg.condWriteValid[srcList[i]][pipeIns.getContext()] == -1) or ((ttReg.regIndex.dstIndex.value == srcList[i]) and (not self.tensixSplRegs.isDstRegProgrammed())):
                            print(f"WARNING: Source register {srcList[i]} condWriteValid in context {pipeIns.getContext()} is not programmed. This may lead to deadlock.")
                        else:
                            validCondCheck.append(self.env.process(
                                self.ttReg.writeValid(srcList[i], pipeIns.getContext(), self.ttReg.condWriteValid[srcList[i]][pipeIns.getContext()],
                                                    pipeIns.getVldUpdMask(srcList[i]), pipeIns.getBankUpdMask(srcList[i]),
                                                    mode, self.args_dict['debug']
                                                    )
                                    )
                                )
            if(len(pipeIns.getDstInt()) != 0):
                dstList = pipeIns.getDstInt()
                if(pipeIns.mnemonic in skipValidCheckOps):    mode = 3    # update Valid , reset inUse to allow following instructions post stallwait, cleardvalid [Special Case]
                else:                                           mode = 3    # update Valid , reset inUse as its a dst
                for i in range(len(dstList)):
                    if dstList[i] not in skipValidCheckforRegs:
                        if (self.ttReg.condWriteValid[dstList[i]][pipeIns.getContext()] == -1) or ((ttReg.regIndex.dstIndex.value == dstList[i]) and (not self.tensixSplRegs.isDstRegProgrammed())):
                            print(f"WARNING: Destination register {dstList[i]} condWriteValid in context {pipeIns.getContext()} is not programmed. This may lead to deadlock.")
                        else:
                            validCondCheck.append(self.env.process(
                                self.ttReg.writeValid(dstList[i],pipeIns.getContext(), self.ttReg.condWriteValid[dstList[i]][pipeIns.getContext()],
                                                    pipeIns.getVldUpdMask(dstList[i]), pipeIns.getBankUpdMask(dstList[i]),
                                                    mode, self.args_dict['debug']
                                                    )
                                )
                            )

            if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} WriteValids Condition (inprogress) registers:{dstList},{srcList}")
            checkTime = self.env.now
            if(len(validCondCheck) >0 ):
                yield simpy.events.AllOf(self.env, validCondCheck)
            if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} WriteValids Condition (done) registers:{dstList},{srcList}, StallTime:{self.env.now - checkTime}")

    def _update_conditional_valids(self, pipeIns):
        # Handles conditional valid updates
        #Cond[Chk/Wri]Vld Update
        condVldUpdCheck = []
        for i in range(self.args_dict['input']['tc' + str(self.coreId)]['numThreads']):
            # Currently only programming for reg 3 (destination register) is supported
            if(pipeIns.hasCondChkVldUpd(3, i)):
                assert pipeIns.hasCondWriVldUpd(3, i) , "Mismatch between condChkVld and condWriVld of destination"
                print("writeCondValid: " , pipeIns.getCondChkVldUpd(3, i), pipeIns.getCondWriVldUpd(3, i))
                condVldUpdCheck.append(self.env.process(
                    self.ttReg.writeCondValid(3, i, pipeIns.getCondChkVldUpd(3, i), pipeIns.getCondWriVldUpd(3, i))
                ))

        checkTime = self.env.now
        if(len(condVldUpdCheck) >0 ):
            if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} CondValids Update Condition (inprogress) ")
            yield simpy.events.AllOf(self.env, condVldUpdCheck)
            if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} CondValids Update Condition (done) StallTime:{(self.env.now - checkTime)}")
            if self.debug & DEBUG_TENSIX_MED_LEVEL:
                print("condCheckValid: ", end='')
                for i in range(4):  #TODO: Replace with number of registers = 4
                    print(self.ttReg.condCheckValid[i],  end =';')
                print()
                print("condWriteValid: ", end='')
                for i in range(4):  #TODO: Replace with number of registers = 4
                    print(self.ttReg.condWriteValid[i],  end =';')
                print()

    def _free_dst_pipes(self, pipeIns):
        # Frees destination pipes after execution
        if(enablePipeStall):
            if(enableStallWait):
                if pipeIns.getOp() in doNotFreeDstPipesforInstrs:
                    if (self.debug & DEBUG_TENSIX_MED_LEVEL):
                        print(f"Cycle:{self.env.now} Dst pipes not to be freed for {pipeIns}")
                else:
                    pipeCondCheck2 = []
                    for i in range(len(pipeIns.getDstPipes())):
                        pipeCondCheck2.append(self.env.process(
                            self.rState.setRsrcState(pipeIns.getDstPipes()[i], pipeIns.getPipesThreadId(), 0, self.args_dict['debug'], instr_info = f"@tensiPipe. cycle: {self.env.now}. set dst pipe (i = {i}, pipe id = {pipeIns.getDstPipes()[i]}) free for instruction: {pipeIns}"))
                            )
                        if(self.args_dict['debug'] & DEBUG_RISC_HIGH_LEVEL):                      print(f"Cycle:{self.env.now} Thread[{0}]: Trying to set free pipe[{1}] as part of {2}".format(pipeIns.getThread(), pipeIns.getDstPipes()[i],  pipeIns.mnemonic, self.env.now))
                    yield simpy.events.AllOf(self.env, pipeCondCheck2)
                    if (self.debug & DEBUG_RISC_HIGH_LEVEL) and (len(pipeIns.getDstPipes()) > 0):
                        print("Cycle:{2} Thread[{0}]: StallPipe reset for {1}, Val={3}".format(pipeIns.getThread(), pipeIns.mnemonic, self.env.now, self.rState.readRsrcState(pipeIns.getDstPipes()[0], pipeIns.getThread())))

    def _free_execution_pipe(self, pipeIns, pipeId):
        # Frees execution pipe if needed
        exe_pipe_id = self.pipes.index(self.targetResource(pipeIns))
        if exe_pipe_id != pipeId:
            raise Exception(f"Cycle:{self.env.now} - error: mismatch between exe_pipe_id and pipeId. instr_info = {pipeIns}")

        if (exe_pipe_id in pipeIns.getDstPipes()) and (pipeIns.getThread() == pipeIns.getPipesThreadId()):
            if (self.debug & DEBUG_TENSIX_MED_LEVEL):
                msg  = f"Cycle:{self.env.now}. @tensixPipe. instr_info: {pipeIns}\n"
                msg += f"- execution pipe id for instruction {pipeIns.getOp()} is {exe_pipe_id}\n"
                msg += f"- stalled dst pipe ids are: {pipeIns.getDstPipes()}\n"
                msg += f"- pipe thread Id is: {pipeIns.getPipesThreadId()}\n"
                msg += f"- as execution pipe and thread is included in dst pipe, we skip the following:\n"
                msg += f"- we do not free exe pipe as dst pipes have already been freed.\n"
                msg += f"- dst resource state:\n"
                for i in range(len(pipeIns.getDstPipes())):
                    msg += f"  - pipe id: {pipeIns.getDstPipes()[i]}, current status: {self.rState.readRsrcState(pipeIns.getDstPipes()[i], pipeIns.getThread())}\n"

                print(msg.rstrip())
        else:
            #Free Exe Pipes
            if(self.debug & DEBUG_TENSIX_MED_LEVEL):   print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Pipe[{pipeId}]: Releasing free for {pipeIns.getOp()}, Val={pipeIns.getThread()}")
            yield self.env.process(self.rState.setRsrcState(pipeId, pipeIns.getThread(), 0, self.args_dict['debug'], instr_info = f"@tensixPipe. cycle: {self.env.now}. set exe pipe (id = {self.pipes.index(self.targetResource(pipeIns))}) free for instruction: {pipeIns}"))
            if(self.debug & DEBUG_TENSIX_MED_LEVEL):   print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Pipe[{pipeId}]: set free after     {pipeIns.getOp()}")

    def _pipe_setup(self, pipeIns, pipeId):
        """Common setup logic for all pipelines."""

        # Check conditions
        syncpipesList = ["SYNC", "TDMA", "THCON", "CFG", "INSTISSUE"]
        checkInOrderFalse = (not enableInOrderIssue) and \
                        (pipeIns.getExPipe() not in syncpipesList)
        if (self.debug & DEBUG_TENSIX_HIGH_LEVEL):
            print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Pipe[{pipeId}] Instruction:{pipeIns.getOp()} checkInOrderFalse={checkInOrderFalse} Pipe Setup")

        return checkInOrderFalse

    def _common_pipe_cleanup(self, pipeIns, pipeId, beginEvent):
        """Common cleanup logic for all pipelines."""
        # Valid/InUse Updates
        yield from self._update_valids(pipeIns)
        yield from self._update_conditional_valids(pipeIns)
        yield from self._free_dst_pipes(pipeIns)

        # Remove from ROB and free execution pipe
        yield self.env.process(self.insROB[pipeIns.getThread()].removeRob(pipeIns.getInsId(), self.debug))
        yield from self._free_execution_pipe(pipeIns, pipeId)

        # Remove from tracker and create end event
        del self.tensixPipeTrk[pipeId][pipeIns.getInsId()]
        endEvent = SIMTRACE('TENSIX', self.pipes[pipeId], self.env.now, 'E', pipeIns.mnemonic, pipeIns.getAttr())

        # Add events to trace
        self.trace_event_list.extend([beginEvent, endEvent])

        #TODO: To reset operands(Src,Dst, )
        # pipeIns.resetOperands()

    def _add_instructions_to_ttAutoloopBuffer(self, pipeId):
        pipe = self.pipes[pipeId]
        while(True):
            pipeIns = yield self.ttBuffer[pipe].get()
            if self.debug & DEBUG_TENSIX_MED_LEVEL:
                print(f"Cycle:{self.env.now}, ttBuffer: pipe = {pipe} Instruction:{pipeIns.getOp()} InsId{pipeIns.getInsId()} Number of items in ttBuffer[{pipe}]={len(self.ttBuffer[pipe].items)}")
            tmp_ins = copy.deepcopy(pipeIns)
            tmp_ins.setInsId(-1)  # Reset insId for the copy
            yield self.ttAutoloopBuffer[pipe].put(pipeIns)
            yield self.env.timeout(1)
            if self.debug & DEBUG_TENSIX_MED_LEVEL:
                print(f"Cycle:{self.env.now}, ttAutoLoopBuffer: pipe = {pipe} Instruction:{pipeIns.getOp()} InsId{pipeIns.getInsId()} Number of items in ttAutoLoopBuffer[{pipe}]={len(self.ttAutoloopBuffer[pipe].items) }")

            if pipe in self.ttAutoloopPipes:
                instr_count = self.tensixSplRegs.readCfgReg(f"THCON_{pipe}_REG0_INSTRN_COUNT") + 1
                instr_loop_count = self.tensixSplRegs.readCfgReg(f"THCON_{pipe}_REG0_INSTRN_LOOP_COUNT") + 1
                num_instr_to_add = instr_count * instr_loop_count - 1 # -1 as one instruction is already added above

                if self.debug & DEBUG_TENSIX_MED_LEVEL:
                    print(f"cycle: {self.env.now}, ttautoloopBuffer: pipe = {pipe} instr_count = {instr_count} instr_loop_count = {instr_loop_count} num_instr_to_add = {num_instr_to_add}")
                for _ in range(num_instr_to_add):
                    instr_copy = copy.deepcopy(tmp_ins)
                    yield self.ttAutoloopBuffer[pipe].put(instr_copy)
                    yield self.env.timeout(1)

    def tensixPipe(self, args, coreId, pipeId, iBuff, oBuff):
        assert pipeId < len(self.pipes) , "Unknown Pipe" + pipeId
        pipeName = f"tensixPipe_{coreId}_{self.pipes[pipeId]}"
        while(True):
            #Wait on Instruction
            if (self.args_dict['enableAutoLoop']):
                if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"{self.name} {self.__class__.__name__} {pipeName}: Cycle:{self.env.now} TCore{self.coreId} Pipe:{self.pipes[pipeId]} Number of items in ttAutoloopBuffer[{self.pipes[pipeId]}]={len(self.ttAutoloopBuffer[self.pipes[pipeId]].items)}")
                # print(f"TTBuffer={self.ttBuffer[self.pipes[pipeId]].printName()}")
                pipeIns     = yield self.ttAutoloopBuffer[self.pipes[pipeId]].get()
                if self.debug & DEBUG_TENSIX_HIGH_LEVEL:
                    print(f"{self.name} {self.__class__.__name__} {pipeName}: Cycle:{self.env.now} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} TCoreFromInstr{pipeIns.getCoreId()} Pipe[{self.pipes[pipeId]}] Instruction:{pipeIns.getOp()} fetched from ttAutoloopBuffer[{self.pipes[pipeId]}], Number of items in ttAutoloopBuffer[{self.pipes[pipeId]}]={len(self.ttAutoloopBuffer[self.pipes[pipeId]].items)}")
            else:
                pipeIns     = yield self.ttBuffer[self.pipes[pipeId]].get()
                if self.debug & DEBUG_TENSIX_HIGH_LEVEL:
                    print(f"{self.name} {self.__class__.__name__} {pipeName}: Cycle:{self.env.now} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} TCoreFromInstr{pipeIns.getCoreId()} Pipe[{self.pipes[pipeId]}] Instruction:{pipeIns.getOp()} fetched from ttBuffer[{self.pipes[pipeId]}], Number of items in ttBuffer[{self.pipes[pipeId]}]={len(self.ttBuffer[self.pipes[pipeId]].items)}")

            # print(f"TTBufferX={self.ttBuffer[self.pipes[pipeId]].printName()} {self.ttBuffer[pipeIns.getExPipe()].printName()}")
            assert pipeIns.getCoreId() == self.coreId , f"Cycle:{self.env.now} CoreId mismatch {pipeIns.getCoreId()} != {self.coreId}, Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} Pipe:{self.pipes[pipeId]} Instruction={pipeIns.getOp()}"
            assert pipeIns.getExPipe() not in self.pipeGrps['UNPACK'] and pipeIns.getExPipe() not in self.pipeGrps['PACK'], f"Compute instantiated for pipe {pipeIns.getExPipe()}"

            # Check In-Order Issue and Resource Availability
            checkInOrderFalse = self._pipe_setup(pipeIns, pipeId)

            # 1b. Pipe/Resource Stall Logic
            if checkInOrderFalse and enablePipeStall and enableStallWait: yield from self._stall_and_check_pipes(pipeIns)

            # 2b. Valid/InUse Checks
            if checkInOrderFalse and enableSync: yield from self._check_valids(pipeIns)
            if(self.debug & DEBUG_TENSIX_MED_LEVEL):   print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} TCoreFromInstr{pipeIns.getCoreId()} Pipe[{pipeId}] Instruction:{pipeIns.getOp()} ready")

            #Exe Start
            execStartTime    = self.env.now
            if self.debug & DEBUG_TENSIX_MED_LEVEL:    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} TCoreFromInstr{pipeIns.getCoreId()} Pipe[{pipeId}] Instruction:{pipeIns.getOp()} Execution (inprogress) in pipe:{pipeIns.getExPipe()}")

            if -1 == pipeIns.getInsId():
                insId = yield self.env.process(self.insROB[pipeIns.getThread()].appendRob(pipeIns, self.debug))
                pipeIns.setInsId(insId)

            beginEvent = SIMTRACE("TENSIX", self.pipes[pipeId], self.env.now, 'B', pipeIns.mnemonic, pipeIns.getAttr())
            self.insByEngine[self.pipes[pipeId]] += 1

            self.tensixPipeTrk[pipeId][pipeIns.getInsId()] = pipeIns

            # 3. L1 access
            # Compute/Sync pipes. No L1 accesses needed.
            # TODO: Model Register Accesses if needed
            pipeIns.setMemInfo("numTotalL1Reads", 0);   pipeIns.setMemInfo("numL1ReadsSent", 0);    pipeIns.setMemInfo("numL1ReadsRcvd", 0)
            pipeIns.setMemInfo("numTotalL1Writes", 0);  pipeIns.setMemInfo("numL1WritesSent", 0);   pipeIns.setMemInfo("numL1WritesRcvd", 0)

            # 4. Execution Delay
            assert pipeIns.getPipeDelay() >=0 , f"Invalid pipeDelay={pipeIns.getPipeDelay()} set for ins={pipeIns.getOp()}"
            yield self.env.timeout(round(pipeIns.getPipeDelay()))
            if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} TCoreFromInstr{pipeIns.getCoreId()} Instruction:{pipeIns.getOp()} Execution (done)      in pipe:{pipeIns.getExPipe()} ExecTime={self.env.now - execStartTime}")

            # TODO: To be removed
            if pipeIns.getOp() == "CLEARDVALID":    #TODO: Handle SetDVALID
                if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"debug: cleardvalid dest_pulse_last={pipeIns.getAttr()['dest_pulse_last']} ")
                match pipeIns.getAttr()['dest_pulse_last']:
                    case CLEARDVALID_dest_pulse_last_MASKS.ZERO: # 0x0
                        if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"debug: Clearing srcA/B/S")
                    case CLEARDVALID_dest_pulse_last_MASKS.UNPACKER: # 0x1
                        if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"debug: UNPACKER CLEARDVALID: {pipeIns.mnemonic}, condCheckValid[3][{pipeIns.getThread()}] = {self.ttReg.condCheckValid[3][pipeIns.getThread()]} condWriteValid[3][{pipeIns.getThread()}] = {self.ttReg.condWriteValid[3][pipeIns.getThread()]} ")
                        self.commWriVld[pipeIns.getThread()] = pipeIns.getAttr()['dest_pulse_last']
                    case CLEARDVALID_dest_pulse_last_MASKS.MATH: # 0x2
                        if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"debug: MATH CLEARDVALID: {pipeIns.mnemonic}, condCheckValid[3][{pipeIns.getThread()}] = {self.ttReg.condCheckValid[3][pipeIns.getThread()]} condWriteValid[3][{pipeIns.getThread()}] = {self.ttReg.condWriteValid[3][pipeIns.getThread()]} ")
                        self.commWriVld[pipeIns.getThread()] = pipeIns.getAttr()['dest_pulse_last']
                    case CLEARDVALID_dest_pulse_last_MASKS.SFPU: # 0x4
                        if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"debug: SFPU CLEARDVALID: {pipeIns.mnemonic}, condCheckValid[3][{pipeIns.getThread()}] = {self.ttReg.condCheckValid[3][pipeIns.getThread()]} condWriteValid[3][{pipeIns.getThread()}] = {self.ttReg.condWriteValid[3][pipeIns.getThread()]} ")
                        self.commWriVld[pipeIns.getThread()] = pipeIns.getAttr()['dest_pulse_last']
                    case CLEARDVALID_dest_pulse_last_MASKS.PACKER: # 0x8
                        if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"debug: PACKER CLEARDVALID: {pipeIns.mnemonic}, condCheckValid[3][{pipeIns.getThread()}] = {self.ttReg.condCheckValid[3][pipeIns.getThread()]} condWriteValid[3][{pipeIns.getThread()}] = {self.ttReg.condWriteValid[3][pipeIns.getThread()]} ")
                        self.commWriVld[pipeIns.getThread()] = pipeIns.getAttr()['dest_pulse_last']

            #5. Valids / Resources Update
            yield from self._common_pipe_cleanup(pipeIns, pipeId, beginEvent)
            if(args['debug'] & DEBUG_TENSIX_MED_LEVEL):    print(f"Cycle:{self.env.now}:{hex(pipeIns.getRelAddr())} Thead{pipeIns.getThread()} insId{pipeIns.getInsId()} Pipe{pipeId} Instruction:{pipeIns.mnemonic} done")

    def convertFormat(self,iBuf, oBuf, scaleFactor):
        while(True):
            req = yield iBuf.get()
            if self.debug & DEBUG_RISC_HIGH_LEVEL:   print(f"Converting format of Req{req.__getReqId__()} by scaleFactor={scaleFactor}. Old ReqWidth={req.__getBytes__()} New ReqWidth={int(req.__getBytes__()*scaleFactor)}    Placing in outBuffer")
            req.__setBytes__(int(req.__getBytes__()*scaleFactor))
            yield oBuf.put(req)
            yield self.env.timeout(1)

    def align(self, rawBytes, alignWidth):
        targetBytes = rawBytes
        if(rawBytes % alignWidth != 0):
            targetBytes += (alignWidth - (rawBytes % alignWidth))
        assert targetBytes % (alignWidth) == 0, f"TargetBytes {targetBytes}B still not aligned to {alignWidth}B"
        return targetBytes

    def unpacker(self, args, coreId, pipeId, iBuff, oBuff):
        assert pipeId < len(self.pipes) , "Unknown Pipe" + pipeId
        pipeName = f"tensixPipe_{coreId}_{self.pipes[pipeId]}"
        reqTrk = {}
        reqTrk1 = {}
        BUFFER_SIZE = 32

        while(True):
            #Wait on Instruction
            if (self.args_dict['enableAutoLoop']):
                if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"{self.name} {self.__class__.__name__} {pipeName}: Cycle:{self.env.now} TCore{self.coreId} Pipe:{self.pipes[pipeId]} Number of items in ttBuffer[{self.pipes[pipeId]}]={len(self.ttBuffer[self.pipes[pipeId]].items)}")
                # print(f"TTBuffer={self.ttBuffer[self.pipes[pipeId]].printName()}")
                if(len(self.ttAutoloopBuffer[self.pipes[pipeId]].items) > 1):
                    if self.debug & DEBUG_RISC_HIGH_LEVEL:   print(f"More than 1 item in unpacker buffer. len={len(self.ttAutoloopBuffer[self.pipes[pipeId]].items)}")
                pipeIns     = yield self.ttAutoloopBuffer[self.pipes[pipeId]].get()
            else:
                pipeIns     = yield self.ttBuffer[self.pipes[pipeId]].get()
            # print(f"TTBufferX={self.ttBuffer[self.pipes[pipeId]].printName()} {self.ttBuffer[pipeIns.getExPipe()].printName()}")
            assert pipeIns.getCoreId() == self.coreId , f"Cycle:{self.env.now} CoreId mismatch {pipeIns.getCoreId()} != {self.coreId}, Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} Pipe:{self.pipes[pipeId]} Instruction={pipeIns.getOp()}"
            assert pipeIns.getExPipe() in self.pipeGrps['UNPACK'], f"Unpacker instantiated for pipe {pipeIns.getExPipe()}"

            # Check In-Order Issue and Resource Availability
            checkInOrderFalse = self._pipe_setup(pipeIns, pipeId)

            # 1b. Pipe/Resource Stall Logic
            if checkInOrderFalse and enablePipeStall and enableStallWait: yield from self._stall_and_check_pipes(pipeIns)

            #Exe Start
            execStartTime    = self.env.now
            if self.debug & DEBUG_TENSIX_MED_LEVEL:    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} TCoreFromInstr{pipeIns.getCoreId()} Pipe[{pipeId}] Instruction:{pipeIns.getOp()} Execution (inprogress) in pipe:{pipeIns.getExPipe()}")

            if -1 == pipeIns.getInsId():
                insId = yield self.env.process(self.insROB[pipeIns.getThread()].appendRob(pipeIns, self.debug))
                pipeIns.setInsId(insId)

            beginEvent = SIMTRACE("TENSIX", self.pipes[pipeId], self.env.now, 'B', pipeIns.mnemonic, pipeIns.getAttr())
            self.insByEngine[self.pipes[pipeId]] += 1

            # 3. Execution Delay
            # assert pipeIns.getPipeDelay() >=0 , f"Invalid pipeDelay={pipeIns.getPipeDelay()} set for ins={pipeIns.getOp()}"
            # yield self.env.timeout(round(pipeIns.getPipeDelay()))
            yield self.env.timeout(round(self.UNPACKERFE_DELAY_CYCLES))
            if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} TCoreFromInstr{pipeIns.getCoreId()} Instruction:{pipeIns.getOp()} Pipeline Stage 1 (done)      in pipe:{pipeIns.getExPipe()} ExecTime={self.env.now - execStartTime}")

            self.tensixPipeTrk[pipeId][pipeIns.getInsId()] = pipeIns

            # 4. L1 read/ REG write
            pipeIns.setMemInfo("numTotalL1Reads", 0);             pipeIns.setMemInfo("numTotalL1Writes", 0);
            pipeIns.setMemInfo("numL1ReadsSent", 0);              pipeIns.setMemInfo("numL1ReadsRcvd", 0);

            pipeIns.setMemInfo("numTotalRegWrites", 0);           pipeIns.setMemInfo("numTotalRegReads", 0);
            pipeIns.setMemInfo("numRegWritesSent", 0);            pipeIns.setMemInfo("numRegWritesRcvd", 0);

            if (pipeIns.getOp() not in "POP_TILES" and pipeIns.getSrcSize() > 0):
                # 4a bytes(L1 Req) = f(Input Format, Tile/Face/Row)
                l1ReqBuff  = simpy.Store(self.env, capacity=BUFFER_SIZE)
                rspBuff    = simpy.Store(self.env, capacity=BUFFER_SIZE)
                regReqBuff = simpy.Store(self.env, capacity=BUFFER_SIZE)
                lastBuff   = simpy.Store(self.env, capacity=BUFFER_SIZE*4)        #TODO: This is really not needed.
                l1PortWidth = self.DEFAULT_L1_PORT_WIDTH; regPortWidth = self.DEFAULT_REG_PORT_WIDTH                       #TODO: This should come from args

                # Src --> convert --> Dst
                if self.debug & DEBUG_TENSIX_MED_LEVEL:
                    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Mem/Reg Request Initiation in pipe:{pipeIns.getExPipe()}")

                # Src Request
                numSrcBytes = self.align(pipeIns.getSrcSize(), max(l1PortWidth, regPortWidth))
                self.numTotalL1Bytes += numSrcBytes
                if self.debug & DEBUG_RISC_HIGH_LEVEL:
                    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Unpacker fetch of {numSrcBytes} bytes from L1")

                # Create source read request
                memReq = None
                memReq = scratchpad.memReq(op="RD", addr=0x0, bytes=numSrcBytes)   #TODO: Generate starting address
                memReq.__setSrc__(pipeId)
                memReq.__setTarget__("L1")
                memReq.__setInsId__(pipeIns.getInsId())
                memReq.__setParentReqIds__([])

                # Dst Request
                numDstBytes = self.align(pipeIns.getDstSize(), max(l1PortWidth, regPortWidth))
                self.numTotalRegBytes += numDstBytes
                if self.debug & DEBUG_RISC_HIGH_LEVEL:
                    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Unpacker write of {numDstBytes} bytes to Reg")

                pipeIns.setMemInfo("numTotalL1Reads", self._calculate_memory_requests(memReq.__getBytes__(), l1PortWidth, memReq.__getOp__()))
                pipeIns.setMemInfo("numTotalRegWrites", self._calculate_memory_requests(numDstBytes, regPortWidth, memReq.__getOp__()))
                self.numTotalL1Req += pipeIns.getMemInfo("numTotalL1Reads")
                self.numTotalRegReq += pipeIns.getMemInfo("numTotalRegWrites")

                if self.debug & DEBUG_RISC_HIGH_LEVEL:
                    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Requests to L1 created. Total Requests to be sent={pipeIns.getMemInfo('numTotalL1Reads')}")
                    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Requests to REG created. Total Requests to be sent={pipeIns.getMemInfo('numTotalRegWrites')}")
                    # memReq.__printReq__()
                if self.debug & DEBUG_TENSIX_MED_LEVEL:
                    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Unpacker Accesses (inprogress)(nonBarrier) Req:{memReq.__getReqId__()}")

                if pipeIns.getSrcSize() > 0:
                    l1ReqBuff.put(memReq)

                if(self.args_dict['enableSharedL1']):
                    if self.debug & DEBUG_TENSIX_MED_LEVEL:
                        print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Unpacker Accesses (inprogress)(nonBarrier) Req:{memReq.__getReqId__()} dstSize={numDstBytes}, srcSize={numSrcBytes}")
                    # L1
                    self.env.process(self._handle_memory_request(l1ReqBuff,iBuff, iBuff, reqTrk, pipeId))
                    self.env.process(self._handle_memory_response(rspBuff, oBuff, reqTrk))

                    # Convert Based on Src/Dst Format
                    self.env.process(self.convertFormat(rspBuff, regReqBuff, numDstBytes/numSrcBytes))

                    # Poll until first response is available
                    while(len(regReqBuff.items) == 0):
                        yield self.env.timeout(1)

                    # 2b. Valid/InUse Checks
                    checkStartTime = self.env.now
                    if(self.debug & DEBUG_TENSIX_MED_LEVEL):   print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} TCoreFromInstr{pipeIns.getCoreId()} Pipe[{self.pipes[pipeId]}] Instruction:{pipeIns.getOp()} Check Valids (inprogress)")
                    if checkInOrderFalse and enableSync: yield from self._check_valids(pipeIns)
                    if(self.debug & DEBUG_TENSIX_MED_LEVEL):   print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} TCoreFromInstr{pipeIns.getCoreId()} Pipe[{self.pipes[pipeId]}] Instruction:{pipeIns.getOp()} Check Valids (done) StallTime={self.env.now - checkStartTime}")

                    # Reg Write
                    self.env.process(self._handle_memory_request(regReqBuff, self.regIBuffer[self.coreId][pipeId], self.regOBuffer[self.coreId][pipeId], reqTrk1, pipeId, overrideOp="WR", overrideTarget="REG"))
                    insId = yield self.env.process(self._handle_memory_response(lastBuff, self.regOBuffer[self.coreId][pipeId], reqTrk1))
                else:
                    # ShortCircuit Mem Path
                    outReq = yield l1ReqBuff.get()

                    # 2b. Valid/InUse Checks
                    if checkInOrderFalse and enableSync: yield from self._check_valids(pipeIns)
                    if(self.debug & DEBUG_TENSIX_MED_LEVEL):   print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} TCoreFromInstr{pipeIns.getCoreId()} Pipe[{pipeId}] Instruction:{pipeIns.getOp()} ready")

                    insId = outReq.__getInsId__()
                    lastBuff.put(outReq)

                assert len(self.tensixPipeTrk[pipeId]) > 0 , "Pipe Tracker is empty"
                pipeIns  = self.tensixPipeTrk[pipeId][insId]
                if self.debug & DEBUG_TENSIX_MED_LEVEL:
                    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Requests to L1 completed. Requests sent={pipeIns.getMemInfo('numL1ReadsSent')} Responses received={pipeIns.getMemInfo('numL1ReadsRcvd')}")
                    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Requests to REG completed. Requests sent={pipeIns.getMemInfo('numRegWritesSent')} Responses received={pipeIns.getMemInfo('numRegWritesRcvd')}")

                if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Unpacker Accesses (done)(nonBarrier) Req:{memReq.__getReqId__()}")

                # Clear LastBuff
                i = 0
                while (i < (len(lastBuff.items))): yield lastBuff.get()
            else:
                # 2b. Valid/InUse Checks
                if checkInOrderFalse and enableSync: yield from self._check_valids(pipeIns)
                if(self.debug & DEBUG_TENSIX_MED_LEVEL):   print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} TCoreFromInstr{pipeIns.getCoreId()} Pipe[{pipeId}] Instruction:{pipeIns.getOp()} ready")

            #5. Valids / Resources Update
            yield from self._common_pipe_cleanup(pipeIns, pipeId, beginEvent)
            if(args['debug'] & DEBUG_TENSIX_MED_LEVEL):    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Pipe{pipeId} Instruction:{pipeIns.mnemonic} done")

    def packer(self, args, coreId, pipeId, iBuff, oBuff):
        assert pipeId < len(self.pipes) , "Unknown Pipe" + pipeId
        pipeName = f"tensixPipe_{coreId}_{self.pipes[pipeId]}"

        reqTrk = {}
        reqTrk1 = {}
        BUFFER_SIZE = 32
        while(True):
            #Wait on Instruction
            if (self.args_dict['enableAutoLoop']):
                if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"{self.name} {self.__class__.__name__} {pipeName}: Cycle:{self.env.now} TCore{self.coreId} Pipe:{self.pipes[pipeId]} Number of items in ttAutoloopBuffer[{self.pipes[pipeId]}]={len(self.ttAutoloopBuffer[self.pipes[pipeId]].items)}")
                # print(f"TTBuffer={self.ttBuffer[self.pipes[pipeId]].printName()}")
                if(len(self.ttAutoloopBuffer[self.pipes[pipeId]].items) > 1):
                    if self.debug & DEBUG_RISC_HIGH_LEVEL:   print(f"More than 1 item in packer buffer. len={len(self.ttAutoloopBuffer[self.pipes[pipeId]].items)}")
                pipeIns     = yield self.ttAutoloopBuffer[self.pipes[pipeId]].get()
            else:
                pipeIns     = yield self.ttBuffer[self.pipes[pipeId]].get()
            # print(f"TTBufferX={self.ttBuffer[self.pipes[pipeId]].printName()} {self.ttBuffer[pipeIns.getExPipe()].printName()}")
            assert pipeIns.getCoreId() == self.coreId , f"Cycle:{self.env.now} CoreId mismatch {pipeIns.getCoreId()} != {self.coreId}, Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} Pipe:{self.pipes[pipeId]} Instruction={pipeIns.getOp()}"
            assert pipeIns.getExPipe() in self.pipeGrps['PACK'], f"Packer instantiated for pipe {pipeIns.getExPipe()}"

            # Check In-Order Issue and Resource Availability
            checkInOrderFalse = self._pipe_setup(pipeIns, pipeId)

            # 1b. Pipe/Resource Stall Logic
            if checkInOrderFalse and enablePipeStall and enableStallWait: yield from self._stall_and_check_pipes(pipeIns)

            #Exe Start
            execStartTime    = self.env.now
            if self.debug & DEBUG_TENSIX_MED_LEVEL:    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} TCoreFromInstr{pipeIns.getCoreId()} Pipe[{pipeId}] Instruction:{pipeIns.getOp()} Execution (inprogress) in pipe:{pipeIns.getExPipe()}")

            if -1 == pipeIns.getInsId():
                insId = yield self.env.process(self.insROB[pipeIns.getThread()].appendRob(pipeIns, self.debug))
                pipeIns.setInsId(insId)

            beginEvent = SIMTRACE("TENSIX", self.pipes[pipeId], self.env.now, 'B', pipeIns.mnemonic, pipeIns.getAttr())
            self.insByEngine[self.pipes[pipeId]] += 1

            # 2. Execution Delay
            assert pipeIns.getPipeDelay() >=0 , f"Invalid pipeDelay={pipeIns.getPipeDelay()} set for ins={pipeIns.getOp()}"
            yield self.env.timeout(round(self.PACKERFE_DELAY_CYCLES))
            # yield self.env.timeout(round(pipeIns.getPipeDelay()))
            if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} TCoreFromInstr{pipeIns.getCoreId()} Instruction:{pipeIns.getOp()} Execution (done)      in pipe:{pipeIns.getExPipe()} ExecTime={self.env.now - execStartTime}")

            # 3. Valid/InUse Checks
            if checkInOrderFalse and enableSync: yield from self._check_valids(pipeIns)
            if(self.debug & DEBUG_TENSIX_MED_LEVEL):   print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} TCoreFromInstr{pipeIns.getCoreId()} Pipe[{pipeId}] Instruction:{pipeIns.getOp()} ready")

            self.tensixPipeTrk[pipeId][pipeIns.getInsId()] = pipeIns

            # 4. L1 write/ REG read
            pipeIns.setMemInfo("numTotalRegReads", 0);             pipeIns.setMemInfo("numTotalRegWrites", 0);
            pipeIns.setMemInfo("numRegReadsSent", 0);             pipeIns.setMemInfo("numRegReadsRcvd", 0);

            pipeIns.setMemInfo("numTotalL1Writes", 0);             pipeIns.setMemInfo("numTotalL1Reads", 0);
            pipeIns.setMemInfo("numL1WritesSent", 0);             pipeIns.setMemInfo("numL1WritesRcvd", 0);

            if (pipeIns.getOp() not in "PUSH_TILES" and pipeIns.getSrcSize() > 0):
                l1ReqBuff  = simpy.Store(self.env, capacity=BUFFER_SIZE)
                regReqBuff  = simpy.Store(self.env, capacity=BUFFER_SIZE)
                rspBuff    = simpy.Store(self.env, capacity=BUFFER_SIZE)
                lastBuff   = simpy.Store(self.env, capacity=BUFFER_SIZE*4)    #TODO: This is really not needed.
                l1PortWidth = self.DEFAULT_L1_PORT_WIDTH; regPortWidth = self.DEFAULT_REG_PORT_WIDTH  #TODO: This should come from args

                # Src --> convert --> Dst
                # Src Request
                numSrcBytes = self.align(pipeIns.getSrcSize(), max(l1PortWidth, regPortWidth))
                if self.debug & DEBUG_TENSIX_MED_LEVEL:
                    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Packer fetch of {numSrcBytes} bytes from REG")
                self.numTotalL1Bytes += numSrcBytes

                # Create source write request
                memReq = None
                memReq = scratchpad.memReq(op="RD", addr=0x0, bytes=numSrcBytes)   #TODO: Generate starting address
                memReq.__setSrc__(pipeId)
                memReq.__setTarget__("REG")
                memReq.__setInsId__(pipeIns.getInsId())
                memReq.__setParentReqIds__([])

                # Dst Request
                numDstBytes = self.align(pipeIns.getDstSize(), max(l1PortWidth, regPortWidth))
                if self.debug & DEBUG_RISC_HIGH_LEVEL:
                    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Packer write of {numDstBytes} bytes to L1")
                self.numTotalRegBytes += numDstBytes

                pipeIns.setMemInfo("numTotalRegReads", self._calculate_memory_requests(memReq.__getBytes__(), regPortWidth, memReq.__getOp__()))
                pipeIns.setMemInfo("numTotalL1Writes", self._calculate_memory_requests(numDstBytes, l1PortWidth, memReq.__getOp__()))
                self.numTotalL1Req += pipeIns.getMemInfo("numTotalL1Writes")
                self.numTotalRegReq += pipeIns.getMemInfo("numTotalRegReads")

                if self.debug & DEBUG_RISC_HIGH_LEVEL:
                    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Requests for insId {pipeIns.getInsId()} to L1 created. Requests to be sent={pipeIns.getMemInfo('numTotalL1Reads')}")
                    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Requests for insId {pipeIns.getInsId()} to REG created. Requests to be sent={pipeIns.getMemInfo('numTotalRegWrites')}")
                    # memReq.__printReq__()
                if pipeIns.getSrcSize() > 0:
                    regReqBuff.put(memReq)

                if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Packer Accesses (inprogress)(nonBarrier) Req:{memReq.__getReqId__()}")
                if(self.args_dict['enableSharedL1']):
                    if self.debug & DEBUG_TENSIX_MED_LEVEL:
                        print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Packer Accesses (inprogress)(nonBarrier) Req:{memReq.__getReqId__()} dstSize={numDstBytes}, srcSize={numSrcBytes}")

                    #Reg
                    self.env.process(self._handle_memory_request(regReqBuff,self.regIBuffer[self.coreId][pipeId], self.regOBuffer[self.coreId][pipeId], reqTrk, pipeId))
                    self.env.process(self._handle_memory_response(rspBuff, self.regOBuffer[self.coreId][pipeId], reqTrk))

                    #Convert
                    self.env.process(self.convertFormat(rspBuff, l1ReqBuff, numDstBytes/numSrcBytes))

                    #L1
                    self.env.process(self._handle_memory_request(l1ReqBuff, iBuff, oBuff, reqTrk1, pipeId, overrideOp="WR", overrideTarget="L1"))
                    insId = yield self.env.process(self._handle_memory_response(lastBuff, oBuff, reqTrk1))
                else:
                    # ShortCircuit Mem Path
                    outReq = yield regReqBuff.get()
                    insId = outReq.__getInsId__()
                    lastBuff.put(outReq)   # Directly put in lastBuff to avoid changing other code

                assert len(self.tensixPipeTrk[pipeId]) > 0 , "Pipe Tracker is empty"
                pipeIns  = self.tensixPipeTrk[pipeId][insId]
                if self.debug & DEBUG_TENSIX_MED_LEVEL:
                    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Requests for insId {insId} to L1 completed. Requests sent={pipeIns.getMemInfo('numL1WritesSent')} Responses received={pipeIns.getMemInfo('numL1WritesRcvd')}")
                    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Requests for insId {insId} to REG completed. Requests sent={pipeIns.getMemInfo('numRegReadsSent')} Responses received={pipeIns.getMemInfo('numRegReadsRcvd')}")

                if self.debug & DEBUG_TENSIX_MED_LEVEL:     print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} TCore{self.coreId} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Instruction:{pipeIns.getOp()} Packer Accesses (done)(nonBarrier) Req:{memReq.__getReqId__()}")

                # Clear LastBuff
                i = 0
                while (i < (len(lastBuff.items))): yield lastBuff.get()

            # 5. Valids / Resources Update
            yield from self._common_pipe_cleanup(pipeIns, pipeId, beginEvent)
            if(args['debug'] & DEBUG_TENSIX_MED_LEVEL):    print(f"Cycle:{self.env.now} Addr:{hex(pipeIns.getRelAddr())} Thread{pipeIns.getThread()} insId{pipeIns.getInsId()} Pipe{pipeId} Instruction:{pipeIns.mnemonic} done")

    def targetResource(self, ins):
        targetEngine = []
        pipe = None

        #Check if Kernel sets the pipe.
        if(ins.getExPipe() != None):
            pipe = ins.getExPipe()
            # Check for pipe name in cfg
            assert pipe in self.pipes, "Unknown execution pipe " + pipe + " set in kernel"

        #Static Pipe Assignment. One or many options
        for i in range(len(self.args_dict['engines'])):
            tptDict = self.args_dict['engines'][i]['engineInstructions']
            for j in range(len(tptDict)):
                for k,v in tptDict[j].items():
                    if(tptDict[j][k] == ins.mnemonic):
                        if(self.debug & DEBUG_TENSIX_HIGH_LEVEL):  print ("Match : ", tptDict[j]['name'] , ins.mnemonic)
                        targetEngine.append(self.args_dict['engines'][i]['engineName'])
                        #TODO: Remove hardcoding of format
                        ins.setPipeDelay(tptDict[j]['tpt']['int32'])

        if(self.debug & DEBUG_TENSIX_HIGH_LEVEL):  print("Instruction = ", ins.mnemonic, "Engine = ", targetEngine)

        #Checks applied before return
        if(pipe != None):
            if(len(targetEngine) > 0):
                assert pipe in targetEngine, "Kernel setting pipe as " + str(pipe) + " but not legal as per static assignment" + str(ins.getOp())
                return pipe
        else:
            if(len(targetEngine) == 0):
                ins.printInstr(ins.getThread())
                assert False, "Engine mapping missing for "+ str(ins.mnemonic)
            elif(len(targetEngine) == 1):
                pipe = targetEngine[0]
                return pipe
            else:
                assert False, "Too many resources to select from: " + str(len(targetEngine)) + ", " + str(ins.printInstr(ins.getThread()))

    def printInstructions(self):
        print("-----------------------------------------------------------------------------------------------")
        print("Summary Start")
        print(f"TensixCore[{self.coreId}] Num Threads = {self.numThreads}")

        tIndex = 0
        for i in range(self.numThreads):
            if self.args_dict['input']['tc' + str(self.coreId)]['th' + str(i) + 'Elf'] == '':
                if self.debug & DEBUG_TENSIX_MED_LEVEL: print(f"Skipping Thread[{str(i)}]")
            else:
                print("TensixCore[",self.coreId, "] Thread[",self.thread[tIndex].threadId, "] Instructions Count(R/T) = ", self.thread[tIndex].rInstructions, " " , self.thread[tIndex].ttInstructions, sep='')
                print(f"TCore{self.coreId} Thread{self.thread[tIndex].threadId} Avg RISC Latency = {self.thread[tIndex].sumRiscLatency/self.thread[tIndex].rInstructions if self.thread[tIndex].rInstructions > 0 else 0}")
                print(f"TCore{self.coreId} Thread{self.thread[tIndex].threadId} Avg RISC Stall Latency = {self.thread[tIndex].sumRiscStallLatency/self.thread[tIndex].rInstructions if self.thread[tIndex].rInstructions > 0 else 0}")
                # print(self.triscRegs[i].__printReg__())
                tIndex += 1
        

        print(f"TensixCore[{self.coreId}] Instructions by Engine")
        for i in range(len(self.pipes)):
            print(self.pipes[i], self.insByEngine[self.pipes[i]])

        print(f"TensixCore[{self.coreId}] CFG Registers Status")
        self.tensixData.__printReg__("cfg")

        print(f"TensixCore[{self.coreId}] TT Instructions Histogram")
        print(opHist)

        print(f"TensixCore[{self.coreId}] L1/REG stats")
        print("TotalL1Accesses =", self.numTotalL1Req)
        print("TotalRegAccesses =", self.numTotalRegReq)
        print("TotalL1Bytes =", self.numTotalL1Bytes)
        print("TotalRegBytes =", self.numTotalRegBytes)
        print("Summary End")

def print_json(jsdata, jsfilename):
    with open(jsfilename, 'w') as jsf:
        json.dump(jsdata, jsf)

def execute_test (args_dict):
    import os

    env = simpy.Environment()

    l1IBuffer = []
    l1OBuffer = []
    assert args_dict['numTCores'] == 1 , "Do not support multi-core configurations"
    for i in range(args_dict['numTCores']):
        l1IPerPipeBuffer    = []
        l1OPerPipeBuffer    = []
        for j in range(len(args_dict['engines'])):
            l1IPerPipeBuffer.append(simpy.Store(env, capacity=2))
            l1OPerPipeBuffer.append(simpy.Store(env, capacity=2))
        l1IBuffer.append(l1IPerPipeBuffer)
        l1OBuffer.append(l1OPerPipeBuffer)
        tCore = tensixCore(env, args_dict, i, l1IBuffer[i], l1OBuffer[i])

    l1 = scratchpad.scratchpadRam(args_dict, env, l1IBuffer, l1OBuffer, args_dict['latency_l1'])

    env.run()
    num_cycles = env.now
    print("Total Cycles = ", env.now)
    tCore.printInstructions()

    # logs_dir = "logs/" + args_dict['odir']
    logs_dir = args_dict['odir']
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    simreport = f"./{logs_dir}/simreport_tc0_" + args_dict['input']['name'] +".json"
    print_json(tCore.trace_event_list, simreport)

    return num_cycles

def main():
    ### ARGS START
    parser = argparse.ArgumentParser(description='Tensix Core Arguments')
    parser.add_argument('--defCfg', help='Configuration File', default='config/tensix_neo/defCfg.json')
    parser.add_argument('--cfg', help='Configuration File', required=True)
    parser.add_argument('--inputcfg', help='Input Configuration File', required=True)
    parser.add_argument('--debug', type=int,
                        help='Debug Mode. 0: No Debug Statement, 1: TRISC Low detail, 4: TRISC Med detail, 16: TRISC High detail, 2: Tensix Low Detail, 8: Tensix Med detail, 32: Tensix High detail, 3: TRISC + Tensix Low detail .....  ',
                        default=0,
                        required=False)
    parser.add_argument('--risc.cpi', type=float, help='RISC IPC. No longer used', default=2, required=False)
    parser.add_argument('--odir', type=str, default ="__llk", help = "Output directory under logs")
    parser.add_argument('--exp', type=str, default ="tcore", help = "Prefix to demarcate different experiment logs")

    args = parser.parse_args()
    print(args)
    args_dict = vars(args)
    with open(args.defCfg, 'r') as file:
        json_dict = json.load(file)
    args_dict.update(json_dict)
    with open(args.cfg, 'r') as file:
        json_dict = json.load(file)
    args_dict.update(json_dict)
    with open(args.inputcfg, 'r') as file:
        json_dict = json.load(file)
    args_dict.update(json_dict)
    ### ARGS END

    return execute_test(args_dict)

if __name__ == '__main__':
    main()
