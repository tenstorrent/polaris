#include "decode/decoded_instruction.hpp"
#include <iomanip>

namespace ttdecode {
namespace decode {

static bool is_non_negative(const std::vector<int>& vals) {
    for (const int v : vals) { if (v < 0) return false; }
    return true;
}

bool registers::set_integers(const std::vector<int>& vals) {
    if (!is_non_negative(vals)) return false;
    integers = vals;
    return true;
}

bool registers::set_integers(const int v) {
    if (v < 0) return false;
    integers = {v};
    return true;
}

bool registers::set_floats(const std::vector<int>& vals) {
    if (!is_non_negative(vals)) return false;
    floats = vals;
    return true;
}

bool registers::set_floats(const int v) {
    if (v < 0) return false;
    floats = {v};
    return true;
}

bool registers::empty() const {
    return integers.empty() && floats.empty();
}

void operands::set_all(const std::map<std::string, int>& arg_all, const std::string& mode) {
    if (!arg_all.empty()) {
        all = arg_all;
    } else if (mode == "v") {
        std::printf("- WARNING: attribute all not set, given argument: empty\n");
    }
}

void operands::set_sources(const registers& r, const std::string& mode) {
    if (!r.integers.empty() || !r.floats.empty()) {
        sources = r;
    } else if (mode == "v") {
        std::printf("- WARNING: source not set\n");
    }
}

void operands::set_destinations(const registers& r, const std::string& mode) {
    if (!r.integers.empty() || !r.floats.empty()) {
        destinations = r;
    } else if (mode == "v") {
        std::printf("- WARNING: destination not set\n");
    }
}

void operands::set_integer_sources(const std::vector<int>& vals, const std::string& mode) {
    if (is_non_negative(vals)) {
        sources.set_integers(vals);
    } else if (mode == "v") {
        std::printf("- WARNING: source integer registers not set\n");
    }
}

void operands::set_integer_sources(const int v, const std::string& mode) {
    if (!sources.set_integers(v) && mode == "v") {
        std::printf("- WARNING: source integer registers not set\n");
    }
}

void operands::set_float_sources(const std::vector<int>& vals, const std::string& mode) {
    if (is_non_negative(vals)) {
        sources.set_floats(vals);
    } else if (mode == "v") {
        std::printf("- WARNING: source floating point registers not set\n");
    }
}

void operands::set_float_sources(const int v, const std::string& mode) {
    if (!sources.set_floats(v) && mode == "v") {
        std::printf("- WARNING: source floating point registers not set\n");
    }
}

void operands::set_integer_destinations(const std::vector<int>& vals, const std::string& mode) {
    if (is_non_negative(vals)) {
        destinations.set_integers(vals);
    } else if (mode == "v") {
        std::printf("- WARNING: destination integer registers not set\n");
    }
}

void operands::set_integer_destinations(const int v, const std::string& mode) {
    if (!destinations.set_integers(v) && mode == "v") {
        std::printf("- WARNING: destination integer registers not set\n");
    }
}

void operands::set_float_destinations(const std::vector<int>& vals, const std::string& mode) {
    if (is_non_negative(vals)) {
        destinations.set_floats(vals);
    } else if (mode == "v") {
        std::printf("- WARNING: destination floating point registers not set\n");
    }
}

void operands::set_float_destinations(const int v, const std::string& mode) {
    if (!destinations.set_floats(v) && mode == "v") {
        std::printf("- WARNING: destination floating point registers not set\n");
    }
}

void operands::set_immediates(const std::vector<int>& vals, const std::string& mode) {
    if (!vals.empty()) {
        immediates = vals;
    } else if (mode == "v") {
        std::printf("- WARNING: immediates not set\n");
    }
}

void operands::set_immediates(const int v, const std::string& mode) {
    immediates = {v};
}

void operands::set_attributes(const std::map<std::string, int>& attrs, const std::string& mode) {
    if (!attrs.empty()) {
        attributes = attrs;
    } else if (mode == "v") {
        std::printf("- WARNING: attributes not set\n");
    }
}

bool operands::empty() const {
    return all.empty() && attributes.empty() && sources.empty() &&
           destinations.empty() && immediates.empty() && decoded_values.empty();
}

std::ostream& operator << (std::ostream& os, const operands& op) {
    os <<"- operands: "<<std::endl;
    os <<"  - all: "<<std::endl;
    for (const auto& kv : op.all) {
        os <<"    - "<<kv.first<<" : "<<kv.second<<std::endl;
    }
    if (!op.sources.integers.empty()) {
        os <<"  - integer sources: [";
        for (std::size_t i = 0; i < op.sources.integers.size(); ++i) {
            if (i) os <<", ";
            os <<op.sources.integers[i];
        }
        os <<"]"<<std::endl;
    }
    if (!op.sources.floats.empty()) {
        os <<"  - float sources: [";
        for (std::size_t i = 0; i < op.sources.floats.size(); ++i) {
            if (i) os <<", ";
            os <<op.sources.floats[i];
        }
        os <<"]"<<std::endl;
    }
    if (!op.destinations.integers.empty()) {
        os <<"  - integer destinations: [";
        for (std::size_t i = 0; i < op.destinations.integers.size(); ++i) {
            if (i) os <<", ";
            os <<op.destinations.integers[i];
        }
        os <<"]"<<std::endl;
    }
    if (!op.destinations.floats.empty()) {
        os <<"  - float destinations: [";
        for (std::size_t i = 0; i < op.destinations.floats.size(); ++i) {
            if (i) os <<", ";
            os <<op.destinations.floats[i];
        }
        os <<"]"<<std::endl;
    }
    if (!op.immediates.empty()) {
        os <<"  - immediates: [";
        for (std::size_t i = 0; i < op.immediates.size(); ++i) {
            if (i) os <<", ";
            os <<op.immediates[i];
        }
        os <<"]"<<std::endl;
    }
    os <<"  - attributes: "<<std::endl;
    for (const auto& kv : op.attributes) {
        os <<"    - "<<kv.first<<" : "<<kv.second<<std::endl;
    }
    if (!op.decoded_values.empty()) {
        os <<"  - decoded_values: "<<std::endl;
        for (const auto& kv : op.decoded_values) {
            os <<"    - "<<kv.first<<" : [";
            for (std::size_t i = 0; i < kv.second.size(); ++i) {
                if (i) os <<", ";
                os <<kv.second[i];
            }
            os <<"]"<<std::endl;
        }
    }
    return os;
}

std::ostream& operator << (std::ostream& os, const decoded_instruction& di) {
    os << di.to_string();
    return os;
}

std::string
decoded_instruction::to_string() const
{
    std::stringstream msg;
    if (program_counter.has_value()) {
        msg << "0x" << std::hex << program_counter.value() << std::dec << " ";
    }
    msg << ": " << std::hex << "0x" << std::setw(8) << word << std::dec << " ";
    if (mnemonic.has_value()) {
        msg << mnemonic.value() << " ";
    }
    if (operands.has_value()) {
        const auto& decoded_values = operands.value().decoded_values;
        const auto& all = operands.value().all;
        std::size_t count = 1;
        for (const auto& [name, v] : operands.value().all) {
            msg <<name<<"="<<v;
            if (!decoded_values.empty()) {
                auto it = decoded_values.find(name);
                if (it != decoded_values.end()) {
                    msg << " [";
                    for (std::size_t i = 0; i < it->second.size(); ++i) {
                        if (i) msg <<", ";
                        msg <<it->second[i];
                    }
                    msg << "]";
                }
            }
            if (count != all.size()) {
                msg << ", ";
                count++;
            }
        }
    }

    return msg.str();
}

void decoded_instruction::set_word(const std::uint32_t w, const std::string& mode) {
    (void)mode;
    word = w;
}

void decoded_instruction::set_program_counter(const std::uint32_t pc, const std::string& mode) {
    (void)mode;
    program_counter = pc;
}

void decoded_instruction::set_kind(const ttdecode::isa::instruction_kind k, const std::string& mode) {
    (void)mode;
    kind = k;
}

void decoded_instruction::set_opcode(const std::uint32_t op, const std::string& mode) {
    (void)mode;
    opcode = op;
}

void decoded_instruction::set_mnemonic(const std::string& mnem, const std::string& mode) {
    if (!mnem.empty()) {
        mnemonic = mnem;
    } else if (mode == "v") {
        std::printf("- WARNING: mnemonic not set\n");
    }
}

void decoded_instruction::set_operands(const struct operands& opnds, const std::string& mode) {
    if (!opnds.empty()) {
        operands = opnds;
    } else if (mode == "v") {
        std::printf("- WARNING: operands not set\n");
    }
}

std::optional<std::uint32_t> decoded_instruction::get_program_counter() const {
    return program_counter;
}

// Convenience methods that initialize operands if needed
void decoded_instruction::set_all(const std::map<std::string, int>& arg_all, const std::string& mode) {
    if (arg_all.empty()) return;
    if (!operands.has_value()) {
        operands = ttdecode::decode::operands{};
    }
    operands->set_all(arg_all, mode);
}

void decoded_instruction::set_sources(const registers& r, const std::string& mode) {
    if (r.empty()) return;
    if (!operands.has_value()) {
        operands = ttdecode::decode::operands{};
    }
    operands->set_sources(r, mode);
}

void decoded_instruction::set_destinations(const registers& r, const std::string& mode) {
    if (r.empty()) return;
    if (!operands.has_value()) {
        operands = ttdecode::decode::operands{};
    }
    operands->set_destinations(r, mode);
}

void decoded_instruction::set_integer_sources(const std::vector<int>& vals, const std::string& mode) {
    if (vals.empty()) return;
    if (!operands.has_value()) {
        operands = ttdecode::decode::operands{};
    }
    operands->set_integer_sources(vals, mode);
}

void decoded_instruction::set_integer_sources(const int v, const std::string& mode) {
    if (!operands.has_value()) {
        operands = ttdecode::decode::operands{};
    }
    operands->set_integer_sources(v, mode);
}

void decoded_instruction::set_float_sources(const std::vector<int>& vals, const std::string& mode) {
    if (vals.empty()) return;
    if (!operands.has_value()) {
        operands = ttdecode::decode::operands{};
    }
    operands->set_float_sources(vals, mode);
}

void decoded_instruction::set_float_sources(const int v, const std::string& mode) {
    if (!operands.has_value()) {
        operands = ttdecode::decode::operands{};
    }
    operands->set_float_sources(v, mode);
}

void decoded_instruction::set_integer_destinations(const std::vector<int>& vals, const std::string& mode) {
    if (vals.empty()) return;
    if (!operands.has_value()) {
        operands = ttdecode::decode::operands{};
    }
    operands->set_integer_destinations(vals, mode);
}

void decoded_instruction::set_integer_destinations(const int v, const std::string& mode) {
    if (!operands.has_value()) {
        operands = ttdecode::decode::operands{};
    }
    operands->set_integer_destinations(v, mode);
}

void decoded_instruction::set_float_destinations(const std::vector<int>& vals, const std::string& mode) {
    if (vals.empty()) return;
    if (!operands.has_value()) {
        operands = ttdecode::decode::operands{};
    }
    operands->set_float_destinations(vals, mode);
}

void decoded_instruction::set_float_destinations(const int v, const std::string& mode) {
    if (!operands.has_value()) {
        operands = ttdecode::decode::operands{};
    }
    operands->set_float_destinations(v, mode);
}

void decoded_instruction::set_immediates(const std::vector<int>& vals, const std::string& mode) {
    if (vals.empty()) return;
    if (!operands.has_value()) {
        operands = ttdecode::decode::operands{};
    }
    operands->set_immediates(vals, mode);
}

void decoded_instruction::set_immediates(const int v, const std::string& mode) {
    if (!operands.has_value()) {
        operands = ttdecode::decode::operands{};
    }
    operands->set_immediates(v, mode);
}

void decoded_instruction::set_attributes(const std::map<std::string, int>& attrs, const std::string& mode) {
    if (attrs.empty()) return;
    if (!operands.has_value()) {
        operands = ttdecode::decode::operands{};
    }
    operands->set_attributes(attrs, mode);
}

} // namespace decode
} // namespace ttdecode
