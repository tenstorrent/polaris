#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Test script to verify synthetic RISC-V instruction file works with synTest function
import sys
import os
import json
import copy
import simpy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ttsim'))

from ttsim.back.tensix_neo.isaFunctions import synTest
from ttsim.back.tensix_neo.t3sim import thread

def test_synthetic_with_thread():
    # Test creating a thread object with synthetic RISC-V instructions
    
    print("Testing thread creation with synthetic RISC-V instructions...")
    
    try:
        # Load base configuration
        config_dir = os.getenv('CONFIG_DIR', os.path.join(os.path.dirname(__file__), '../../config/tensix_neo'))
        config_path = os.path.join(config_dir, 'ttqs_neo4_jul1.json')
        with open(config_path, 'r') as f:
            base_config = json.load(f)
        
        # Create a minimal input configuration structure
        input_config = {
            "input": {
                "syn": 1,
                "name": "synthetic_riscv_test",
                "tc0": {
                    "numThreads": 3,
                    "th0Path": "./tests/tensix_neo/",
                    "th0Elf": "synthetic_risc_v_test.thread0.txt",
                    "th1Path": "./tests/tensix_neo/",
                    "th1Elf": "synthetic_risc_v_test.thread1.txt", 
                    "th2Path": "./tests/tensix_neo/",
                    "th2Elf": "synthetic_risc_v_test.thread2.txt", 
                    "th3Path": "",
                    "th3Elf": ""
                }
            }
        }
        
        # Merge configurations
        args_dict = copy.deepcopy(base_config)
        args_dict.update(input_config)
        # args_dict['debug'] = 0x1F  # Enable all debug flags for testing

        # Add synthetic ranges for thread 0
        # We have 34 total instructions (from the test output), so we need to calculate the correct end addresses
        # The assertion is checking if startAddr + (numInstructions * 4) == endAddr
        # For main (13 instructions): startAddr=0x1000, endAddr=0x1000+13*4=0x1034
        # For op1 (10 instructions): startAddr=0x1034, endAddr=0x1034+10*4=0x105C  
        # For op2 (11 instructions): startAddr=0x105C, endAddr=0x105C+11*4=0x1088
        args_dict['input']['syn0Range'] = [
            ['main', 0x1000, 0x1034 ],      # main function: 13 instructions
        ]
        args_dict['input']['syn1Range'] = [
            ['main', 0x1000, 0x1034 ],      # main function: 13 instructions
            ['op1', 0x1034, 0x105C ],       # op1 function: 10 instructions
        ]
        args_dict['input']['syn2Range'] = [
            ['main', 0x1000, 0x1034 ],      # main function: 13 instructions
            ['op1', 0x1034, 0x105C ],       # op1 function: 10 instructions
            ['op2', 0x105C, 0x1088 ]        # op2 function: 11 instructions
        ]
        
        
        # Create SimPy environment
        env = simpy.Environment()
        
        print(f"✓ SimPy environment created")
        print(f"✓ Configuration loaded with synthetic file: {args_dict['input']['tc0']['th0Elf']}")
        print(f"✓ Synthetic mode enabled: {args_dict['input']['syn']}")
        print(f"✓ Thread 0 path: {args_dict['input']['tc0']['th0Path']}")
        print(f"✓ Synthetic ranges defined for functions: main, op1, op2")
        
        # Import additional required modules for thread creation
        from ttsim.back.tensix_neo.t3sim import ttReg, pipeResource, replayState, rob
        import ttsim.back.tensix_neo.triscFunc as triscFunc
        from ttsim.back.tensix_neo.tensixFunc import tensixFunc, ttSplRegs
        
        # Create required components for thread object
        coreId = 0
        numThreads = args_dict['input']['tc0']['numThreads']
        # threadId = 0
        startKernelName = "main"

        # Create memory data
        memData = triscFunc.triscMemFunc(args_dict)

        # Create ttReg - correct constructor: ttReg(env, args_dict, coreId, numThreads)
        ttReg_obj = ttReg(env, args_dict, coreId, numThreads)
            
        # Create pipe resources
        pipes = ["CFG", "MATH", "SFPU", "PACKER", "UNPACKER", "SYNC", "TDMA", "THCON", "INSTISSUE"]
        pipeResource_obj = pipeResource(env, args_dict, coreId)
            
        pipeGrps = {
            'UNPACK': [],
            'SFPU': [],
            'MATH': [],
            'PACK': [],
            'TDMA': [],
            'CFG': [],
            'SYNC': [],
            'THCON': [],
            'XMOV': []
        }

        #Create lists for per thread objects
        replayState_obj = []
        insROB_obj      = []
        threadFunc      = []
        triscRegs_obj   = []

        # Create TRISC registers for each thread
        for threadId in range(numThreads):
            # Create TRISC registers
            triscRegs_obj.append(triscFunc.triscRegs(coreId, threadId, args_dict))

        # Once per thread objects
        # Create tensix special registers
        tensixSplRegs_obj = ttSplRegs(coreId, args_dict)
            
        # Create tensix function - using correct constructor parameters
        # tensixFunc(coreId, mem, args, pipeGrps, pipes, tensixSplRegs, triscRegs)
        pipe_dict = {}
        tensixFunc_obj = tensixFunc(coreId, memData, args_dict, pipeGrps, pipe_dict, tensixSplRegs_obj, triscRegs_obj)
            
        # Create FIFOs and buffers (using simpy stores)
        ttFifo = simpy.Store(env, capacity=100)
        ttBuffer = []
        for i in range(len(pipes)):
            ttBuffer.append(simpy.Store(env, capacity=10))
            

        for threadId in range(numThreads):
            print(f"✓ Creating thread {threadId} for core {coreId}") 
            
            # Create thread function (RISC-V)
            threadFunc.append(triscFunc.triscFunc(coreId, threadId, memData, args_dict, tensixSplRegs_obj, triscRegs_obj[threadId]))
            
            # Create replay state
            replayState_obj.append(replayState(env, coreId, threadId))
            
            # Create instruction ROB
            insROB_obj.append(rob(env, coreId, threadId))
            
            # Create trace event list
            traceEventList = []
            
            print(f"✓ All required components created for thread object")
            
            # Create the thread object
            thread_obj = thread(
                args_dict=args_dict,
                env=env,
                kernelName=startKernelName,
                coreId=coreId,
                threadId=threadId,
                threadFunc=threadFunc[len(threadFunc)-1],
                tensixFunc=tensixFunc_obj,
                ttFifo=ttFifo,
                ttBuffer=ttBuffer,
                rState=pipeResource_obj,
                ttReg=ttReg_obj,
                pipes=pipes,
                replayState=replayState_obj,
                insROB=insROB_obj,
                traceEventList=traceEventList
            )
            
            print(f"✓ Thread object created successfully")
            print(f"  - Core ID: {coreId}")
            print(f"  - Thread ID: {threadId}")
            print(f"  - Kernel Name: {startKernelName}")
            print(f"  - Thread name: {thread_obj.name}")

        # Run the simulation
        print(f"✓ Starting simulation execution...")
        env.run()       # Run until all threads complete
        
        print(f"✓ Simulation completed successfully")
        print(f"  - Final simulation time: {env.now}")

        for threadId in range(numThreads):
            print(f"Thread{threadId} PC={hex(thread_obj.pc)} prevPC={hex(thread_obj.prevPC)} endAddr={hex(thread_obj.endAddr)}")
            # assert thread_obj.pc == 0  # Check if the program counter is at 0 after execution
            # assert thread_obj.prevPC == thread_obj.endAddr - 4 # Check if the previous PC is at the end address minus 4 (last instruction executed) 
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating thread with synthetic instructions: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Running synthetic RISC-V instruction tests")
    print("=" * 60)
    
    # Test 1: Thread creation with synthetic instructions
    print("\n2. Testing thread creation with synthetic instructions:")
    success1 = test_synthetic_with_thread()
    
    # Overall result
    print("\n" + "=" * 60)
    if success1:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)
