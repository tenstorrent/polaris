#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import json
import os
import simpy

import ttsim.back.tensix_neo.scratchpad as scratchpad
import ttsim.back.tensix_neo.t3sim as t3sim
import ttsim.front.llk.read_elfs as read_elfs
import ttsim.front.llk.tensix as binutils_tensix
import ttsim.front.llk.rv32 as binutils_rv32

class neoCore:
    def __init__(self, env, args):
        self.env    = env
        self.args   = args
        self.tCore       = []
        l1IBuffer    = []
        l1OBuffer    = []

        for i in range(args['numTCores']):
            l1IPerPipeBuffer    = []
            l1OPerPipeBuffer    = []
            for j in range(len(args['engines'])):
                l1IPerPipeBuffer.append(simpy.Store(env, capacity=1))
                l1OPerPipeBuffer.append(simpy.Store(env, capacity=1))
            l1IBuffer.append(l1IPerPipeBuffer)
            l1OBuffer.append(l1OPerPipeBuffer)
            self.tCore.append(t3sim.tensixCore(env, args, i, l1IBuffer[i], l1OBuffer[i]))

        print("Tensix Core Details:")
        for i in range(args['numTCores']):
            print(f"{self.tCore[i].coreId} {self.tCore[i].name}")

        l1 = scratchpad.scratchpadRam(args, env, l1IBuffer, l1OBuffer, args['latency_l1'])

        print("Construction Completed")

    def printInstructions(self, args, logs_dir):
        for i in range(self.args['numTCores']):
            self.tCore[i].printInstructions()
            simreport = f"./{logs_dir}/simreport_{self.args['exp']}_tc{i}_{args['input']['name']}.json"
            print("Simreport = ", simreport)
            print_json(self.tCore[i].trace_event_list, simreport)


def print_json(jsdata, jsfilename):
    with open(jsfilename, 'w') as jsf:
        json.dump(jsdata, jsf)

def execute_test (args_dict):
    env = simpy.Environment()
    nCore = neoCore(env, args_dict)

    env.run()
    num_cycles = env.now
    print("Total Cycles = ", env.now)

    logs_dir = args_dict['odir']
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    nCore.printInstructions(args_dict, logs_dir)

    return num_cycles

def execute_test (args_dict):
    env = simpy.Environment()
    nCore = neoCore(env, args_dict)

    env.run()
    num_cycles = env.now
    print("Total Cycles = ", env.now)

    logs_dir = args_dict['odir']
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    nCore.printInstructions(args_dict, logs_dir)

    return num_cycles

def get_accepted_architectures():
    arches = set([
        "ttwh",
        "ttbh",
        "ttqs"])
    return sorted(arches)

def get_accepted_llk_version_tags():
    tags = dict({
        "ttwh": None,
        "ttbh": None,
        "ttqs": ["feb19", "mar18", "jul1", "jul27", "sep23"]
        })

    assert len(tags) == len(get_accepted_architectures()), \
        f"Number of accepted llkVersionTags {len(tags)} does not match number of accepted architectures {len(get_accepted_architectures())}. " \
        f"Please update the get_accepted_llk_version_tags function."

    for key in tags.keys():
        assert key in get_accepted_architectures(), f"Key {key} in accepted llkVersionTags must be an accepted architecture."

    return tags

def needs_llk_version_tag(arch):
    assert arch in get_accepted_architectures(), f"Architecture {arch} is not in the list of accepted architectures."
    assert arch in get_accepted_llk_version_tags().keys(), \
        f"Architecture {arch} is not in the list of accepted llkVersionTags. " \
        f"Please update the get_accepted_llk_version_tags function."
    if get_accepted_llk_version_tags()[arch]:
        return True
    return False

def get_default_cfg_path():
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../../config/tensix_neo"))

def get_default_tt_isa_file_path():
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../config/llk/instruction_sets/ttqs"))

def get_architecture(args_dict):
    accepted_arches = get_accepted_architectures()
    arch_from_elf_files = read_elfs.get_architecture_from_tneo_sim_args_dict(args_dict)
    assert arch_from_elf_files in accepted_arches, f"Arch from ELF files: {arch_from_elf_files} not present in accepted arches. "\
        f"Accepted arches are: {accepted_arches}. "\
        f"Please update accepted_arches."

    if 'arch' in args_dict:
        assert args_dict['arch'] == arch_from_elf_files, \
            f"Mismatch between expected arch and arch from inputcfg. given arch: {args_dict['arch']}, arch from reading of elf files: {arch_from_elf_files}"

    return arch_from_elf_files

def get_llk_version_tag(args_dict):
    arch = get_architecture(args_dict)
    if needs_llk_version_tag(arch):
        accepted_llk_version_tags = get_accepted_llk_version_tags()[arch]
        if 'llkVersionTag' not in args_dict:
            if 1 == len(accepted_llk_version_tags):
                return accepted_llk_version_tags[0]
            else:
                raise ValueError(f"llkVersionTag must be specified for {arch} architecture. "
                                 f"Accepted llkVersionTags are: {accepted_llk_version_tags}. "
                                 f"If this is a new tag, please update the llkVersionTag in args_dict.")
        else:
            assert args_dict['llkVersionTag'] in accepted_llk_version_tags, \
                f"Given llkVersionTag {args_dict['llkVersionTag']} is not in the list of accepted llkVersionTags for {arch} architecture. " \
                f"Accepted llkVersionTags are: {accepted_llk_version_tags}. " \
                f"If this is a new tag, please update the llkVersionTag in args_dict."
            return args_dict['llkVersionTag']
    else:
        print(f"Architecture {arch} does not require a llkVersionTag, returning None.")

    return None

def get_cfg(args, args_dict):
    if hasattr(args, 'cfg') and args.cfg:
        assert os.path.exists(args.cfg), f"Configuration file {args.cfg} does not exist."
        return args.cfg

    if 'cfg' in args_dict and args_dict['cfg']:
        assert os.path.exists(args_dict['cfg']), f"Configuration file {args_dict['cfg']} does not exist."
        return args_dict['cfg']

    assert "arch" in args_dict, "Architecture must be specified in args_dict to determine the default configuration file."
    arch = args_dict['arch']
    assert arch in get_accepted_architectures(), f"Architecture {arch} is not in the list of accepted architectures."

    if "ttqs" != arch:
        raise ValueError(f"No default cfg file is available for architecture {arch}.")

    if needs_llk_version_tag(arch):
        assert "llkVersionTag" in args_dict, "llkVersionTag must be specified in args_dict when using this architecture."

    llk_version_tag = args_dict['llkVersionTag'] if needs_llk_version_tag(arch) else None

    return os.path.normpath(os.path.join(get_default_cfg_path(), f"{arch}_neo4_{llk_version_tag}.json"))

def get_memory_map(args, args_dict):
    if hasattr(args, 'memoryMap') and args.memoryMap:
        assert os.path.exists(args.memoryMap), f"Memory map file {args.memoryMap} does not exist."
        return args.memoryMap

    if 'memoryMap' in args_dict and args_dict['memoryMap']:
        assert os.path.exists(args_dict['memoryMap']), f"Memory map file {args_dict['memoryMap']} does not exist."
        return args_dict['memoryMap']

    assert "arch" in args_dict, "Architecture must be specified in args_dict to determine the default memory map file."
    arch = args_dict['arch']
    assert arch in get_accepted_architectures(), f"Architecture {arch} is not in the list of accepted architectures."

    if "ttqs" != arch:
        raise ValueError(f"No default memory map file is available for architecture {arch}.")

    llk_version_tag = args_dict['llkVersionTag'] if needs_llk_version_tag(arch) else None

    return os.path.normpath(os.path.join(get_default_cfg_path(), f"{arch}_memory_map_{llk_version_tag}.json"))

def get_debug(args, args_dict):
    if hasattr(args, 'debug') and args.debug is not None:
        return args.debug

    if 'debug' in args_dict and args_dict['debug'] is not None:
        return args_dict['debug']

    return 0  # Default debug level

def get_risc_cpi(args, args_dict):
    if hasattr(args, 'risc.cpi') and getattr(args, 'risc.cpi') is not None:
        return getattr(args, 'risc.cpi')

    if 'risc.cpi' in args_dict and args_dict['risc.cpi'] is not None:
        return args_dict['risc.cpi']

    return 2.0  # Default RISC CPI value

def get_tt_isa_file_name(args, args_dict):
    if hasattr(args, 'ttISAFileName') and args.ttISAFileName:
        assert os.path.exists(args.ttISAFileName), f"Tensix ISA file {args.ttISAFileName} does not exist."
        return args.ttISAFileName

    if 'ttISAFileName' in args_dict and args_dict['ttISAFileName']:
        assert os.path.exists(args_dict['ttISAFileName']), f"Tensix ISA file {args_dict['ttISAFileName']} does not exist."
        return args_dict['ttISAFileName']

    assert "arch" in args_dict, "Architecture must be specified in args_dict to determine the default Tensix ISA file."
    arch = args_dict['arch']
    assert arch in get_accepted_architectures(), f"Architecture {arch} is not in the list of accepted architectures."

    if "ttqs" != arch:
        raise NotImplementedError(f"Default Tensix ISA file for architecture {arch} is not implemented.")

    if needs_llk_version_tag(arch):
        assert "llkVersionTag" in args_dict, "llkVersionTag must be specified in args_dict when using this architecture."

        llk_version_tag = str(args_dict['llkVersionTag'])
        isa_file_name = f"assembly.{llk_version_tag}.yaml"
        isa_file_path = os.path.normpath(os.path.join(get_default_tt_isa_file_path(), isa_file_name))
        assert os.path.exists(isa_file_path), f"Default Tensix ISA file {isa_file_path} does not exist."

        return isa_file_path
    else:
        raise NotImplementedError(f"Default Tensix ISA file for architecture {arch} without llkVersionTag is not implemented.")

def update_args_dict_with_inputcfg(args, args_dict):
    accepted_inputcfg_keys = [
        'arch',
        'cfg',
        'debug',
        'description',
        'input',
        'llkVersionTag',
        'memoryMap',
        'ttISAFileName',
        'numTCores']
    with open(args.inputcfg, 'r') as file:
        inputcfg = json.load(file)
        if any(key not in accepted_inputcfg_keys for key in inputcfg.keys()):
            raise ValueError(f"Input configuration file {args.inputcfg} contains key(s) that are not accepted: {inputcfg.keys()}. "
                             f"Accepted keys are: {accepted_inputcfg_keys}. "
                             f"Please update the input configuration file.")
        args_dict.update(inputcfg)

    args_dict['arch']          = get_architecture(args_dict)
    args_dict['llkVersionTag'] = get_llk_version_tag(args_dict)
    args_dict['memoryMap']     = get_memory_map(args, args_dict)
    args_dict['cfg']           = get_cfg(args, args_dict)
    args_dict['debug']         = get_debug(args, args_dict)
    args_dict['ttISAFileName'] = get_tt_isa_file_name(args, args_dict)

def update_args_dict_with_cfg(args, args_dict):
    accepted_cfg_keys = [
        'enableSharedL1',
        'enableSync',
        'engines',
        'globalPointer',
        'latency_l1',
        'maxNumThreadsperNeoCore',
        'numTriscCores',
        'orderScheme',
        'risc.cpi',
        'enableRiscPM',
        'riscPipeDepth',
        'branchMisPredictPenalty',
        'enableScoreboardCheckforRegs',
        'enableForwardingforRegs',
        'stack']

    key_cfg = 'cfg'
    if key_cfg not in args_dict:
        raise ValueError(f"Key '{key_cfg}' not found in args_dict. Please ensure that the configuration file is provided.")

    with open(args_dict[key_cfg], 'r') as file:
        cfg = json.load(file)
        if any(key not in accepted_cfg_keys for key in cfg.keys()):
            raise ValueError(f"Configuration file {args_dict[key_cfg]} contains key(s) that are not accepted: {cfg.keys()}. "
                             f"Accepted keys are: {accepted_cfg_keys}. "
                             f"Please update the configuration file.")

        if any(key in args_dict for key in accepted_cfg_keys if key not in vars(args).keys()):
            raise ValueError(f"Some/all of the keys {accepted_cfg_keys} already present in args_dict. Please check the input script and commandline arguments")

        args_dict.update(cfg)
        args_dict['risc.cpi'] = get_risc_cpi(args, args_dict)

def get_memory_map_from_file(file_name: str):

    accepted_memory_map_keys = ["trisc_map", "n1_cluster_map", "n4_cluster_map"]
    with open(file_name, 'r') as file:
        mem_map = json.load(file)
        if any(key not in accepted_memory_map_keys for key in mem_map.keys()):
            raise ValueError(f"Memory map file {file_name} contains key(s) that are not accepted: {mem_map.keys()}. "
                             f"Accepted keys are: {accepted_memory_map_keys}. "
                             f"Please update the memory map file.")

        for key0, value0 in mem_map.items():
            for key1, value1 in value0.items():
                for key2, value2 in value1.items():
                    if key2 in ['START', 'END']:
                        mem_map[key0][key1][key2] = int(value2, 16)
                    if key2 == "REGISTERS":
                        for key3, value3 in value2.items():
                            mem_map[key0][key1][key2][key3] = int(value3, 16)

        assert 'trisc_map' in mem_map, "'trisc_map' not found in memory map."
        assert 'cfg_regs' in mem_map['trisc_map'], "'cfg_regs' not found in trisc_map."
        assert 'OFFSETS' in mem_map['trisc_map']['cfg_regs'], "'OFFSETS' not found in trisc_map/cfg_regs."

        new_offset = dict()
        for key0, value0 in mem_map['trisc_map']['cfg_regs']['OFFSETS'].items():
            int_key0 = int(key0)
            assert int_key0 not in new_offset, f"Duplicate offset key {int_key0} found when converting from string to int."
            for reg_info in value0.values():
                if 'MASK' in reg_info:
                    reg_info['MASK'] = int(reg_info['MASK'], 16)

            new_offset[int_key0] = value0

        mem_map['trisc_map']['cfg_regs']['OFFSETS'] = new_offset

        return mem_map

    raise Exception("- error: Memory map could not be formed")

def update_args_dict_with_memory_map(args, args_dict):
    key_memory_map = 'memoryMap'
    if key_memory_map not in args_dict:
        raise ValueError(f"Key '{key_memory_map}' not found in args_dict. Please ensure that the memory map file is provided.")

    memory_map = get_memory_map_from_file(args_dict[key_memory_map]) # file_name

    if any(key in args_dict for key in memory_map.keys() if key not in vars(args).keys()):
        raise ValueError(f"Some/all of the keys {memory_map.keys()} already present in args_dict. "
                         f"Please check the input script and commandline arguments")

    args_dict.update(memory_map)

def get_tt_isa_from_file(file_name: str):
    print(f"Reading Tensix ISA from file: {file_name}")
    return binutils_tensix.get_instruction_set_from_file_name(file_name)

def update_args_dict_with_tt_isa(args, args_dict):
    key_tt_isa_file_name = 'ttISAFileName'
    if key_tt_isa_file_name not in args_dict:
        raise ValueError(f"Key '{key_tt_isa_file_name}' not found in args_dict. Please ensure that the Tensix ISA file is provided.")

    assert "arch" in args_dict, "Architecture must be specified in args_dict to determine the default configuration file."
    arch = args_dict['arch']
    assert arch in get_accepted_architectures(), f"Architecture {arch} is not in the list of accepted architectures."

    tt_isa = get_tt_isa_from_file(args_dict[key_tt_isa_file_name]) # file_name

    key = "ttISA"
    assert key not in args_dict, f"Key '{key}' already present in args_dict. Please check the input script and commandline arguments"

    args_dict[key] = {
        binutils_tensix.decoded_instruction.to_instruction_kind(arch) : tt_isa,
        binutils_rv32.instruction_kind() : binutils_rv32.get_default_instruction_set()}

def update_args_dict_with_enableAutoLoop(args, args_dict):
    key_enable_auto_loop = 'enableAutoLoop'

    if key_enable_auto_loop in args:
        args_dict[key_enable_auto_loop] = args.enableAutoLoop
    else:
        args_dict[key_enable_auto_loop] = True
    print(f"AutoLoop in TCore = {args_dict[key_enable_auto_loop]}")

def check_max_num_threads_per_neo_core(args_dict):
    archs_max_num_threads = {
        "ttqs": 4,
        "ttwh": 3,
        "ttbh": 3
    }
    if args_dict['arch'] in archs_max_num_threads:
        assert archs_max_num_threads[args_dict['arch']] == args_dict['maxNumThreadsperNeoCore'], f"Expected {archs_max_num_threads[args_dict['arch']]} threads per Neo core for '{args_dict['arch']}' architecture, but got {args_dict['maxNumThreadsperNeoCore']}."
    else:
        raise ValueError(f"Unknown architecture: {args_dict['arch']}")

def main():
    ### ARGS START
    parser = argparse.ArgumentParser(description='Tensix Core Arguments')
    parser.add_argument('--defCfg', help='Configuration File', required=False)
    parser.add_argument('--cfg', help='Configuration File', required=False)
    parser.add_argument('--memoryMap', help='Memory Map File', required=False)
    parser.add_argument('--ttISAFileName', help='Tensix instruction set File', required=False)
    parser.add_argument('--inputcfg', help='Input Configuration File', required=True)
    parser.add_argument('--debug', type=int,
                        help='Debug Mode. 0: No Debug Statement, 1: TRISC Low detail, 4: TRISC Med detail, 16: TRISC High detail, 2: Tensix Low Detail, 8: Tensix Med detail, 32: Tensix High detail, 3: TRISC + Tensix Low detail .....  ',
                        required=False)
    parser.add_argument('--risc.cpi', type=float, help='RISC IPC', required=False)
    parser.add_argument('--odir', type=str, default ="__llk", help = "Output directory under logs")
    parser.add_argument('--exp', type=str, default ="neo", help = "Prefix to demarcate different experiment logs")

    args = parser.parse_args()
    print("command line arguments: ", args)
    args_dict = copy.deepcopy(vars(args))
    update_args_dict_with_inputcfg(args, args_dict)
    update_args_dict_with_cfg(args, args_dict)
    update_args_dict_with_memory_map(args, args_dict)
    update_args_dict_with_tt_isa(args, args_dict)
    check_max_num_threads_per_neo_core(args_dict)
    update_args_dict_with_enableAutoLoop(args, args_dict)

    ### ARGS END
    return execute_test(args_dict)

if __name__ == '__main__':
    main()
