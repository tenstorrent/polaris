#include "decode/utils.hpp"

namespace ttdecode {
namespace decode {

std::vector<std::uint32_t> bytes_to_uint32_vector(
    const std::uint8_t* bytes,
    const std::size_t num_bytes,
    const bool little_endian,
    const bool ceil) {
    if (!bytes || num_bytes == 0) {
        return {};
    }

    constexpr std::size_t bytes_per_word = sizeof(std::uint32_t);
    const std::size_t complete_words = num_bytes / bytes_per_word;
    const std::size_t remaining_bytes = num_bytes % bytes_per_word;
    const std::size_t total_words = complete_words + (remaining_bytes ? 1 : 0);

    std::vector<std::uint32_t> result;
    result.reserve(total_words);

    // Process complete words
    for (std::size_t i = 0; i < complete_words; ++i) {
        std::uint32_t word = 0;

        if (little_endian) {
            // Little endian: LSB first
            for (std::size_t j = 0; j < 4; ++j) {
                word |= static_cast<std::uint32_t>(bytes[(i * 4) + j]) << (j * 8);
            }
        } else {
            // Big endian: MSB first
            for (std::size_t j = 0; j < 4; ++j) {
                word |= static_cast<std::uint32_t>(bytes[(i * 4) + j]) << ((3 - j) * 8);
            }
        }

        result.push_back(word);
    }

    if (remaining_bytes and ceil) {
        std::uint32_t word = 0;
        const std::size_t offset = complete_words * bytes_per_word;

        if (little_endian) {
            // Little endian: LSB first
            for (std::size_t j = 0; j < remaining_bytes; ++j) {
                word |= static_cast<std::uint32_t>(bytes[offset + j]) << (j * 8);
            }
        } else {
            // Big endian: MSB first
            for (std::size_t j = 0; j < remaining_bytes; ++j) {
                word |= static_cast<std::uint32_t>(bytes[offset + j]) << ((3 - j) * 8);
            }
        }

        result.push_back(word);
    }

    return result;
}

// Overload that takes a vector of bytes
std::vector<std::uint32_t>
bytes_to_uint32_vector(
    const std::vector<std::uint8_t>& bytes,
    const bool little_endian,
    const bool ceil) {
    return bytes_to_uint32_vector(
        bytes.data(),
        bytes.size(),
        little_endian,
        ceil);
}


} // namespace decode
} // namespace ttdecode