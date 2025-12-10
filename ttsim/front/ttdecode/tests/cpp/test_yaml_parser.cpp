#include "isa/yaml_parser.hpp" // The class we are testing
#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// A helper function to get the path to the test data
std::string get_test_data_path(const std::string& filename) {
    // This assumes tests are run from the build directory
    return std::string(CMAKE_SOURCE_DIR) + "/examples/python/" + filename;
}

std::string get_malformed_data_path(const std::string& filename) {
    return std::string(CMAKE_SOURCE_DIR) + "/tests/cpp/" + filename;
}


// Test fixture for YamlParser tests
class YamlParserTest : public ::testing::Test {
protected:
    ttdecode::isa::parser parser;
};

// Test case for parsing a valid file
TEST_F(YamlParserTest, HandlesGoodFile) {
    std::string good_file = get_test_data_path("config.yaml");
    YAML::Node config = parser.parse(good_file);

    std::cout<<"- here I am!. config['features'] = "<<config["features"]<<std::endl;

    ASSERT_TRUE(config["server"]);
    ASSERT_TRUE(config["database"]);
    ASSERT_TRUE(config["features"]);

    EXPECT_EQ(config["server"]["host"].as<std::string>(), "127.0.0.1");
    EXPECT_EQ(config["database"]["port"].as<int>(), 5432);
    EXPECT_EQ(config["features"].IsSequence(), true);
    EXPECT_EQ(config["features"].size(), 3);
    // EXPECT_EQ(config["features"].as<std::string>(), "authentication");
    EXPECT_THAT(config["features"].as<std::vector<std::string>>(), ::testing::ElementsAre("authentication", "logging", "realtime_updates"));
}

// Test case for handling a file that does not exist
TEST_F(YamlParserTest, ThrowsOnMissingFile) {
    std::string missing_file = "path/to/non_existent_file.yaml";
    EXPECT_THROW(parser.parse(missing_file), ttdecode::isa::YamlParsingException);
}

// Test case for handling a file with malformed YAML
TEST_F(YamlParserTest, ThrowsOnMalformedFile) {
    std::string malformed_file = get_malformed_data_path("malformed.yaml");
    EXPECT_THROW(parser.parse(malformed_file), ttdecode::isa::YamlParsingException);
}
