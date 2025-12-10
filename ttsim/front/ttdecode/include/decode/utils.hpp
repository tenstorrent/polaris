#pragma once

#include "constants/constants.hpp"
#include <cstdint>
#include <vector>
#include <cstring>

namespace ttdecode {
namespace decode {

// Bit rotations within 32-bit words
inline std::uint32_t rotl32(std::uint32_t value, std::uint32_t shift) {
    shift %= ttdecode::constants::NUM_BITS_PER_INSTRUCTION;
    if (shift == 0) return value;
    return static_cast<std::uint32_t>((value << shift) | (value >> (ttdecode::constants::NUM_BITS_PER_INSTRUCTION - shift)));
}

inline std::uint32_t rotr32(std::uint32_t value, std::uint32_t shift) {
    shift %= ttdecode::constants::NUM_BITS_PER_INSTRUCTION;
    if (shift == 0) return value;
    return static_cast<std::uint32_t>((value >> shift) | (value << (ttdecode::constants::NUM_BITS_PER_INSTRUCTION - shift)));
}

inline std::uint32_t swizzle(std::uint32_t instruction) {
    // Left circular shift by 2 bits in 32-bit space
    return rotl32(instruction, 2u);
}

inline std::uint32_t unswizzle(std::uint32_t instruction) {
    // Right circular shift by 2 bits in 32-bit space
    return rotr32(instruction, 2u);
}

// Extract a bit-field of length 'size' starting at 'start_bit' (LSB = 0)
inline std::uint32_t extract_bits(std::uint32_t x, std::uint32_t start_bit, std::uint32_t size) {
    if (size == 0u) return 0u;
    const std::uint32_t mask = (size == 32u ? 0xFFFF'FFFFu : ((1u << size) - 1u));
    return (x >> start_bit) & mask;
}

// Sign-extend 'value' which currently contains a field with 'bit_from_lsb' as MSB (0-based)
inline std::int32_t sign_extend(std::uint32_t value, std::uint32_t bit_from_lsb) {
    const std::uint32_t sign_bit = (1u << bit_from_lsb);
    if (value & sign_bit) {
        const std::uint32_t mask = (1u << (bit_from_lsb + 1u)) - 1u;
        return static_cast<int32_t>(value | ~mask);
    }
    return static_cast<std::int32_t>(value);
}

// Convert bytes to vector of uint32_t
// Handles endianness and padding for incomplete words
std::vector<std::uint32_t> bytes_to_uint32_vector(
    const std::uint8_t* bytes,
    const std::size_t num_bytes,
    const bool little_endian = true,
    const bool ceil = false);

// Overload that takes a vector of bytes
std::vector<std::uint32_t>
bytes_to_uint32_vector(
    const std::vector<std::uint8_t>& bytes,
    const bool little_endian = true,
    const bool ceil = false);

} // namespace decode
} // namespace ttdecode
