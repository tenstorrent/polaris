#pragma once 

#include <cstdint>

namespace ttdecode {
namespace constants {

static constexpr std::uint32_t NUM_BITS_PER_BYTE = 8;
static constexpr std::uint32_t NUM_BITS_PER_INSTRUCTION = 32;
static constexpr std::uint32_t RISCV_OPCODE_START_BIT = 0;
static constexpr std::uint32_t TENSIX_OPCODE_START_BIT = 24;
static constexpr std::uint32_t NUM_BYTES_PER_INSTRUCTION = NUM_BITS_PER_INSTRUCTION / NUM_BITS_PER_BYTE;

} // namespace constants
} // namespace ttdecode