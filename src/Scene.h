// ══════════════════════════════════════════════════════════════
//  Scene — editor scene description + hand-rolled JSON persistence
//
//  A scene is a set of placed object instances (each referencing an
//  asset in the engine's asset registry by index) plus the world
//  generation parameters used to build the terrain. Save/load is a
//  dependency-free JSON reader/writer mirroring TerrainSettings.
// ══════════════════════════════════════════════════════════════
#pragma once

#include <glm/glm.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "TerrainSettings.h"

// One placed instance in the scene. assetIndex refers to the engine's
// sceneAssets registry; assetName is stored too so a scene stays meaningful
// if the asset ordering changes between sessions (resolved by name on load).
struct SceneObject {
    std::string name = "Object";
    int         assetIndex = 0;
    std::string assetName;                 // for name-based re-resolution
    glm::vec3   position{0.0f};
    glm::vec3   eulerDeg{0.0f};
    glm::vec3   scale{1.0f};
};

// The subset of TerrainSettings that defines the generated world. Stored with
// the scene so loading reproduces the same terrain.
struct WorldParams {
    uint32_t worldSeed = 0;
    float    worldSize = 4000.0f;
    uint32_t masterRes = 1024;
    float    seaFraction = 0.42f;
    float    terrainTotalHeight = 90.0f;
    float    mountainRidgeWeight = 0.30f;
    float    sunElevation = 30.0f;
    float    sunAzimuth = 45.0f;

    void captureFrom(const TerrainSettings& s) {
        worldSeed = s.worldSeed;
        worldSize = s.worldSize;
        masterRes = s.masterRes;
        seaFraction = s.seaFraction;
        terrainTotalHeight = s.terrainTotalHeight;
        mountainRidgeWeight = s.mountainRidgeWeight;
        sunElevation = s.sunElevation;
        sunAzimuth = s.sunAzimuth;
    }

    void applyTo(TerrainSettings& s) const {
        s.worldSeed = worldSeed;
        s.worldSize = worldSize;
        s.masterRes = masterRes;
        s.seaFraction = seaFraction;
        s.terrainTotalHeight = terrainTotalHeight;
        s.mountainRidgeWeight = mountainRidgeWeight;
        s.sunElevation = sunElevation;
        s.sunAzimuth = sunAzimuth;
    }
};

struct Scene {
    std::vector<SceneObject> objects;
    WorldParams              world;

    bool saveToJSON(const std::string& path) const {
        std::ofstream f(path);
        if (!f.is_open()) {
            std::cerr << "[scene] cannot open for write: " << path << "\n";
            return false;
        }
        f << std::fixed;
        f << "{\n";
        f << "  \"world\": {\n";
        f << "    \"worldSeed\": " << world.worldSeed << ",\n";
        f << "    \"worldSize\": " << world.worldSize << ",\n";
        f << "    \"masterRes\": " << world.masterRes << ",\n";
        f << "    \"seaFraction\": " << world.seaFraction << ",\n";
        f << "    \"terrainTotalHeight\": " << world.terrainTotalHeight << ",\n";
        f << "    \"mountainRidgeWeight\": " << world.mountainRidgeWeight << ",\n";
        f << "    \"sunElevation\": " << world.sunElevation << ",\n";
        f << "    \"sunAzimuth\": " << world.sunAzimuth << "\n";
        f << "  },\n";
        f << "  \"objects\": [\n";
        for (size_t i = 0; i < objects.size(); ++i) {
            const SceneObject& o = objects[i];
            f << "    {\n";
            f << "      \"name\": \"" << o.name << "\",\n";
            f << "      \"assetIndex\": " << o.assetIndex << ",\n";
            f << "      \"assetName\": \"" << o.assetName << "\",\n";
            f << "      \"position\": [" << o.position.x << ", " << o.position.y << ", " << o.position.z << "],\n";
            f << "      \"eulerDeg\": [" << o.eulerDeg.x << ", " << o.eulerDeg.y << ", " << o.eulerDeg.z << "],\n";
            f << "      \"scale\": [" << o.scale.x << ", " << o.scale.y << ", " << o.scale.z << "]\n";
            f << "    }" << (i + 1 < objects.size() ? "," : "") << "\n";
        }
        f << "  ]\n";
        f << "}\n";
        std::cout << "[scene] saved " << objects.size() << " objects to " << path << "\n";
        return true;
    }

    bool loadFromJSON(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "[scene] not found: " << path << "\n";
            return false;
        }
        std::stringstream buf;
        buf << f.rdbuf();
        const std::string text = buf.str();

        objects.clear();

        // ── Parse the "world" block (flat unique keys) ──
        {
            size_t wpos = text.find("\"world\"");
            std::string sub = (wpos == std::string::npos) ? text : text.substr(wpos);
            auto rdU = [&](const char* k, uint32_t& t) {
                std::string v = scalarValue(sub, k);
                if (!v.empty()) t = (uint32_t)std::stoul(v);
            };
            auto rdF = [&](const char* k, float& t) {
                std::string v = scalarValue(sub, k);
                if (!v.empty()) t = std::stof(v);
            };
            rdU("worldSeed", world.worldSeed);
            rdF("worldSize", world.worldSize);
            rdU("masterRes", world.masterRes);
            rdF("seaFraction", world.seaFraction);
            rdF("terrainTotalHeight", world.terrainTotalHeight);
            rdF("mountainRidgeWeight", world.mountainRidgeWeight);
            rdF("sunElevation", world.sunElevation);
            rdF("sunAzimuth", world.sunAzimuth);
        }

        // ── Parse the "objects" array (one {} block per object) ──
        size_t apos = text.find("\"objects\"");
        if (apos != std::string::npos) {
            size_t lb = text.find('[', apos);
            size_t rb = text.find(']', lb == std::string::npos ? apos : lb);
            if (lb != std::string::npos && rb != std::string::npos && rb > lb) {
                size_t cur = lb;
                while (true) {
                    size_t ob = text.find('{', cur);
                    if (ob == std::string::npos || ob > rb) break;
                    size_t oe = text.find('}', ob);
                    if (oe == std::string::npos || oe > rb) break;
                    std::string block = text.substr(ob, oe - ob + 1);

                    SceneObject o;
                    o.name = stringValue(block, "name");
                    if (o.name.empty()) o.name = "Object";
                    o.assetName = stringValue(block, "assetName");
                    std::string ai = scalarValue(block, "assetIndex");
                    if (!ai.empty()) o.assetIndex = std::stoi(ai);
                    o.position = vec3Value(block, "position", glm::vec3(0.0f));
                    o.eulerDeg = vec3Value(block, "eulerDeg", glm::vec3(0.0f));
                    o.scale    = vec3Value(block, "scale", glm::vec3(1.0f));
                    objects.push_back(o);

                    cur = oe + 1;
                }
            }
        }
        std::cout << "[scene] loaded " << objects.size() << " objects from " << path << "\n";
        return true;
    }

private:
    // Returns the raw scalar token following "key": (number/bool, no quotes).
    static std::string scalarValue(const std::string& text, const std::string& key) {
        size_t pos = text.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = text.find(':', pos);
        if (pos == std::string::npos) return "";
        pos = text.find_first_not_of(" \t\r\n", pos + 1);
        if (pos == std::string::npos) return "";
        size_t end = text.find_first_of(",\t\r\n}]", pos);
        return text.substr(pos, end - pos);
    }

    // Returns the string value following "key": "..." (without quotes).
    static std::string stringValue(const std::string& text, const std::string& key) {
        size_t pos = text.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = text.find(':', pos);
        if (pos == std::string::npos) return "";
        size_t q1 = text.find('"', pos);
        if (q1 == std::string::npos) return "";
        size_t q2 = text.find('"', q1 + 1);
        if (q2 == std::string::npos) return "";
        return text.substr(q1 + 1, q2 - q1 - 1);
    }

    // Reads "key": [a, b, c] into a vec3 (falls back to def on any failure).
    static glm::vec3 vec3Value(const std::string& text, const std::string& key, glm::vec3 def) {
        size_t pos = text.find("\"" + key + "\"");
        if (pos == std::string::npos) return def;
        size_t lb = text.find('[', pos);
        size_t rb = text.find(']', lb == std::string::npos ? pos : lb);
        if (lb == std::string::npos || rb == std::string::npos || rb <= lb) return def;
        std::string inner = text.substr(lb + 1, rb - lb - 1);
        for (char& c : inner) if (c == ',') c = ' ';
        std::istringstream is(inner);
        glm::vec3 v = def;
        is >> v.x >> v.y >> v.z;
        return v;
    }
};
