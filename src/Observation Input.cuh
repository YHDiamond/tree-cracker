#ifndef TREE_CRACKER_OBSERVATION_INPUT_CUH
#define TREE_CRACKER_OBSERVATION_INPUT_CUH

#include <array>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// Each line: version Forest type x z height_min height_max 12_leaf_states.
// Leaf states are 0 (absent), 1 (present), ? (unknown), in Settings order.
// A height range of 0 0 means unknown. No seed is accepted by this format.
struct TreeObservation {
    int version, type;
    int32_t x, z, heightMin, heightMax;
    std::array<int, 12> leaves;
};

inline std::vector<TreeObservation> readTreeObservations(std::istream &input) {
    std::vector<TreeObservation> trees;
    std::set<std::tuple<int, int32_t, int32_t, int>> seen;
    std::map<std::tuple<int, int32_t, int32_t>, std::set<int>> typesAtPosition;
    std::map<std::tuple<int, int32_t, int32_t>, std::set<std::pair<int32_t, int32_t>>> chunks;
    std::string line;
    size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        line = line.substr(0, line.find('#'));
        std::istringstream row(line);
        std::string version, biome, type, leaves, extra;
        if (!(row >> version)) continue;
        TreeObservation tree{};
        auto invalid = [&](const char *message) {
            throw std::invalid_argument("Observation line " + std::to_string(lineNumber) + ": " + message);
        };
        if (!(row >> biome >> type >> tree.x >> tree.z >> tree.heightMin >> tree.heightMax >> leaves) || row >> extra)
            invalid("expected version Forest type x z height_min height_max leaf_states");
        if (version == "1.14.4") tree.version = 3;
        else if (version == "1.16.1") tree.version = 4;
        else if (version == "1.16.4") tree.version = 5;
        else if (version == "1.17.1") tree.version = 6;
        else invalid("unsupported version");
        if (biome != "Forest") invalid("supported biome is Forest");
        int lower = 0, upper = 0;
        if (type == "Oak") { tree.type = 0; lower = 4; upper = 6; }
        else if (type == "Fancy_Oak") { tree.type = 1; lower = 3; upper = 14; }
        else if (type == "Birch") { tree.type = 2; lower = 5; upper = 7; }
        else if (type == "Unknown") { tree.type = 3; lower = 0; upper = 0; }
        else invalid("unsupported tree type");
        if (tree.heightMin == 0 && tree.heightMax == 0) {
            tree.heightMin = INT32_MIN;
            tree.heightMax = INT32_MAX;
        } else if (tree.heightMin < lower || tree.heightMax > upper || tree.heightMin > tree.heightMax)
            invalid("height range is outside this tree type's bounds");
        if (leaves.size() != 12) invalid("expected exactly 12 corner-leaf states");
        for (size_t i = 0; i < leaves.size(); ++i) {
            if (leaves[i] == '0') tree.leaves[i] = 0;
            else if (leaves[i] == '1') tree.leaves[i] = 1;
            else if (leaves[i] == '?') tree.leaves[i] = 2;
            else invalid("leaf states must be 0, 1 or ?");
            if ((tree.type == 1 || tree.type == 3) && tree.leaves[i] != 2)
                invalid("Fancy_Oak and Unknown do not support corner-leaf observations");
        }
        if (!seen.emplace(tree.version, tree.x, tree.z, tree.type).second)
            invalid("duplicate tree type at the same coordinates");
        auto &types = typesAtPosition[{tree.version, tree.x, tree.z}];
        if (!types.empty() && (tree.type == 3 || types.count(3)))
            invalid("Unknown cannot be combined with a known type at the same coordinates");
        types.insert(tree.type);
        auto floor16 = [](int32_t value) { return static_cast<int32_t>((static_cast<int64_t>(value) - (value < 0 ? 15 : 0)) / 16); };
        auto &positions = chunks[{tree.version, floor16(tree.x), floor16(tree.z)}];
        positions.emplace(tree.x, tree.z);
        if (positions.size() > 16) invalid("at most 16 distinct tree positions per chunk are supported");
        trees.push_back(tree);
    }
    if (input.bad()) throw std::runtime_error("Could not read observations");
    if (trees.empty()) throw std::invalid_argument("No tree observations provided");
    return trees;
}

inline std::vector<TreeObservation> readTreeObservations(const char *path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error(std::string("Could not open observations: ") + path);
    return readTreeObservations(input);
}

#endif
