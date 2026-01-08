#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import collections
import copy
import json
import os
import simpy
import typing

import ttsim.back.tensix_neo.isaFunctions as isaFunctions
import ttsim.back.tensix_neo.scratchpad as scratchpad
import ttsim.back.tensix_neo.t3sim as t3sim
import ttsim.front.llk.read_elfs as read_elfs
import ttsim.front.llk.rv32 as binutils_rv32
import ttsim.front.llk.tensix as binutils_tensix

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
        "ttqs": [str(tag.value) for tag in isaFunctions.TTQSTags]
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

def get_default_tt_isa_file_path(arch: str):
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), f"../../config/llk/instruction_sets/{arch}"))

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
                return isaFunctions.TTQSTags[accepted_llk_version_tags[0]]
            else:
                raise ValueError(f"llkVersionTag must be specified for {arch} architecture. "
                                 f"Accepted llkVersionTags are: {accepted_llk_version_tags}. "
                                 f"If this is a new tag, please update the llkVersionTag in args_dict.")
        else:
            assert args_dict['llkVersionTag'] in accepted_llk_version_tags, \
                f"Given llkVersionTag {args_dict['llkVersionTag']} is not in the list of accepted llkVersionTags for {arch} architecture. " \
                f"Accepted llkVersionTags are: {accepted_llk_version_tags}. " \
                f"If this is a new tag, please update the llkVersionTag in args_dict."
            return isaFunctions.TTQSTags[args_dict['llkVersionTag']]
    else:
        print(f"Architecture {arch} does not require a llkVersionTag, returning None.")

    return None

def get_value_from_args_dicts(key: str,
              args: object | None = None,
              /,
              *,
              dicts: collections.abc.Iterable[collections.abc.Mapping[str, typing.Any]] = (),
              default: typing.Any = None) -> typing.Any:
    # Highest priority: argparse Namespace (or any object with the attribute)
    if args is not None:
        val = getattr(args, key, None)
        if val is not None:
            return val

    # Then check the dicts in the order provided
    for d in dicts:
        if not isinstance(d, collections.abc.Mapping):
            raise ValueError(f"Expected a Mapping, but got {type(d)}, key: {key}, # dicts: {len(dicts)}")
        val = d.get(key, None)
        if val is not None:
            return val

    # Fallback
    return default

def get_file_path_from_args_dicts(
        key: str,
        args: object | None = None,
        /,
        *,
        dicts: collections.abc.Iterable[collections.abc.Mapping[str, typing.Any]] = dict(),
        default: typing.Any | None = None):
    file_incl_path = get_value_from_args_dicts(key, args, dicts = dicts, default = default)
    if file_incl_path is not None:
        assert os.path.isfile(file_incl_path), f"{key} file {file_incl_path} does not exist."
        return file_incl_path

    return default

def get_cfg(args, args_dict):
    cfg_file_name = get_file_path_from_args_dicts('cfg', args, dicts = (args_dict,), default = None)
    if cfg_file_name is not None:
        return cfg_file_name

    assert "arch" in args_dict, "Architecture must be specified in args_dict to determine the default configuration file."
    arch = args_dict['arch']
    assert arch in get_accepted_architectures(), f"Architecture {arch} is not in the list of accepted architectures."

    if "ttqs" != arch:
        raise ValueError(f"No default cfg file is available for architecture {arch}.")

    if needs_llk_version_tag(arch):
        assert "llkVersionTag" in args_dict, "llkVersionTag must be specified in args_dict when using this architecture."

    llk_version_tag = args_dict['llkVersionTag'].value if needs_llk_version_tag(arch) else None

    return os.path.normpath(os.path.join(get_default_cfg_path(), f"{arch}_neo4_{llk_version_tag}.json"))

def get_memory_map(args, args_dict):
    memory_map_file_name = get_file_path_from_args_dicts('memoryMap', args, dicts = (args_dict,), default = None)
    if memory_map_file_name is not None:
        return memory_map_file_name

    assert "arch" in args_dict, "Architecture must be specified in args_dict to determine the default memory map file."
    arch = args_dict['arch']
    assert arch in get_accepted_architectures(), f"Architecture {arch} is not in the list of accepted architectures."

    if "ttqs" != arch:
        raise ValueError(f"No default memory map file is available for architecture {arch}.")

    llk_version_tag = args_dict['llkVersionTag'].value if needs_llk_version_tag(arch) else None

    return os.path.normpath(os.path.join(get_default_cfg_path(), f"{arch}_memory_map_{llk_version_tag}.json"))

def get_tt_isa_file_name(args, args_dict):
    tt_isa_file_name = get_file_path_from_args_dicts('ttISAFileName', args, dicts = (args_dict,), default = None)
    if tt_isa_file_name is not None:
        return tt_isa_file_name

    assert "arch" in args_dict, "Architecture must be specified in args_dict to determine the default Tensix ISA file."
    arch = args_dict['arch']
    assert arch in get_accepted_architectures(), f"Architecture {arch} is not in the list of accepted architectures."

    isa_file_name: str = "assembly.yaml"
    if needs_llk_version_tag(arch):
        assert "llkVersionTag" in args_dict, f"llkVersionTag must be specified in inputcfg for architecture {arch}."
        llk_version_tag = args_dict['llkVersionTag'].value
        isa_file_name = f"assembly.{llk_version_tag}.yaml"

    isa_file_path = os.path.normpath(os.path.join(get_default_tt_isa_file_path(arch), isa_file_name))
    assert os.path.exists(isa_file_path), f"Default Tensix ISA file {isa_file_path} does not exist."
    return isa_file_path

def update_args_dict_with_inputcfg(args, args_dict):
    accepted_inputcfg_keys = [
        'arch',
        'cfg',
        'debug',
        'defCfg',
        'description',
        'exp',
        'input',
        'llkVersionTag',
        'memoryMap',
        'numTCores',
        'odir',
        'ttISAFileName',
        'risc.cpi',
        ]
    with open(args.inputcfg, 'r') as file:
        inputcfg = json.load(file)
        if any(key not in accepted_inputcfg_keys for key in inputcfg.keys()):
            raise ValueError(f"Input configuration file {args.inputcfg} contains key(s) that are not accepted: {inputcfg.keys()}. "
                             f"Accepted keys are: {accepted_inputcfg_keys}. "
                             f"Please update the input configuration file.")
        args_dict.update(inputcfg)

    args_dict['arch']          = get_architecture(args_dict)
    args_dict['llkVersionTag'] = get_llk_version_tag(args_dict)
    args_dict['debug']         = get_value_from_args_dicts('debug', args, dicts = (args_dict,), default = 0)
    args_dict['cfg']           = get_cfg(args, args_dict)
    args_dict['memoryMap']     = get_memory_map(args, args_dict)
    args_dict['ttISAFileName'] = get_tt_isa_file_name(args, args_dict)
    args_dict['defCfg']        = get_file_path_from_args_dicts('defCfg', args, dicts = (args_dict,), default = None)
    args_dict['exp']           = get_value_from_args_dicts('exp', args, dicts = (args_dict,), default = 'neo')
    args_dict['odir']          = get_value_from_args_dicts('odir', args, dicts = (args_dict,), default = '__llk')

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

        cfg['risc.cpi'] = get_value_from_args_dicts('risc.cpi', args, dicts = (args_dict, cfg), default = 1.0)
        args_dict.update(cfg)

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
    parser.add_argument('--odir', type=str, help = "Output directory under logs", required = False)
    parser.add_argument('--exp', type=str, help = "Prefix to demarcate different experiment logs", required = False)

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
