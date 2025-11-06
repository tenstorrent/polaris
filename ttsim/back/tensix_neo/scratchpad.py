#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import argparse
import json
import simpy

import ttsim.back.tensix_neo.t3sim as t3sim

import itertools

class memReq:
    cnt = itertools.count(start=0, step=1)
    def __init__(self, op, addr, bytes):
        self.reqId  = next(memReq.cnt)
        self.parentReqIds = [] 
        self.childReqIds  = [] 
        self.insId  = None
        self.op     = op
        self.bytes  = bytes
        self.addr   = addr
        self.src    = None
        self.threadId = -1 
        self.target = None

    def __getReqId__(self):
        return self.reqId

    def __getTarget__(self):
        return self.target

    def __setTarget__(self, target):
        self.target = target

    def __getInsId__(self):
        return self.insId
    def __setInsId__(self, id):
        self.insId = id

    def __getParentReqIds__(self):
        return self.parentReqIds
    def __setParentReqIds__(self, idList):
        self.parentReqIds = idList

    def __getChildReqIds__(self):
        return self.childReqIds
    def __setChildReqIds__(self, idList):
        self.childReqIds = idList

    def __getPipeId__(self):
        return self.pipeId
    def __setPipeId__(self, id):
        self.pipeId = id

    def __setBytes__(self, bytes):
        self.bytes = bytes
    def __getBytes__(self):
        return self.bytes

    def __setOp__(self, op):
        self.op = op
    def __getOp__(self):
        return self.op

    def __getAddr__(self):
        return self.addr
    def __setAddr__(self, addr):
        self.addr = addr

    def __getSrc__(self):
        return self.src
    def __setSrc__(self, src):
        self.src = src

    def __getSrcPipe__(self):
        return self.srcPipe
    def __setSrcPipe__(self, pipeName):
        self.srcPipe = pipeName

    def __getCoreId__(self):
        return self.coreId
    def __setCoreId__(self, coreId):
        self.coreId = coreId

    def __getThreadId__(self):
        return self.threadId
    def __setThreadId__(self, threadId):
        self.threadId = threadId

    def __printReq__(self):
        print(f"ReqId: {self.__getReqId__()}, Op: {self.__getOp__()}, Addr: {self.__getAddr__()}, Bytes: {self.__getBytes__()}, Src: {self.__getSrc__()}, Target: {self.__getTarget__()} InsId: {self.__getInsId__()}")

class scratchpadRam:

    numPorts    = 0
    numBanks    = 0
    delayMode   = 0 #0 - Fixed Latency Mode , 1 - Statistical Latency Mode, 2- Cycle Accurate Latency 
    def __init__(self, args, env, iBuffer, oBuffer, latency):
        self.env        = env
        self.args       = args

        print("numEngines =", len(self.args['engines'])) 
        self.numPorts   = self.args['numTCores']*len(self.args['engines'])  #TODO: Num ports should be programmable after port-bank arbitration is implemented
        self.numBanks   = self.numPorts                                     #TODO: Num banks should be programmable after port-bank arbitration is implemented
        self.debug      = args['debug']

        self.iBuffer    = iBuffer
        self.oBuffer    = oBuffer
        # self.latency    = latency
        self.latencyRd    = latency
        self.latencyWr    = 1

        #Processes
        self.procRW = []
        self.reqTrk = {}
        # In reality, there are multiple ports per pipe. TODO: Model
        for core in range(self.args['numTCores']):
            for pipe in range(len(self.args['engines'])):
                self.procRW.append(self.env.process(self.processMemReq(core, pipe)))

    def arbitrate(self, req, oBuffer):
        startTime = self.env.now
        # if self.debug & 0x8: print(f"Cycle:{startTime} Thread{req.__getThreadId__()} req{req.__getReqId__()} insId{req.__getInsId__()} src{req.__getSrc__()} arbitration (in progress)")
        self.reqTrk[req.__getReqId__()] = req
        if req.__getOp__() == "WR":
            yield self.env.timeout(self.latencyWr)
        else:
            yield self.env.timeout(self.latencyRd)
        yield oBuffer.put(req)
        if self.debug & 0x8: print(f"Cycle:{self.env.now} TCore{req.__getCoreId__()} Thread{req.__getThreadId__()} req{req.__getReqId__()} insId{req.__getInsId__()} src:{req.__getSrcPipe__()} Scratchpad response (done)")
        endTime = self.env.now
        assert req.__getReqId__() in self.reqTrk , f"req{req.__getReqId__()} not found in tracker"
        self.reqTrk.pop(req.__getReqId__(), None)
        # if self.debug & 0x8: print(f"Cycle:{endTime} Thread{req.__getThreadId__()} req{req.__getReqId__()} insId{req.__getInsId__()} src{req.__getSrc__()} arbitration (done)")

    def processMemReq(self, tCore, pipe):
        #Hit is guaranteed
        while(True):
            ins = yield self.iBuffer[tCore][pipe].get()
            assert ins.__getCoreId__() == tCore , f"TCore Mismatch: Expected {tCore}, got {ins.__getCoreId__()}"
            if self.debug & 0x8: print(f"Cycle:{self.env.now} TCore{tCore} Thread{ins.__getThreadId__()} req{ins.__getReqId__()} insId{ins.__getInsId__()} src:{ins.__getSrcPipe__()} Scratchpad request (done)")
            self.env.process(self.arbitrate(ins, self.oBuffer[tCore][pipe]))
            yield self.env.timeout(1)