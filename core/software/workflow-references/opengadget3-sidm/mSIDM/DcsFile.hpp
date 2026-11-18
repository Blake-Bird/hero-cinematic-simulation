
/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Defines the layout of the dcs config file
 *
 * @author OpenGadget3-development team, in particular Marc Wiertel and Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#pragma once
#include "../CodeBase/precision.hpp"
#define JSON_ASSERT(x) // suppress redefinition of assert
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <cstddef>

/**
 * @brief Data structure for configuration and look-up tables.
 */
struct DcsFile
{
    size_t version;
    std::string method; // either "grid" or "zslice"
    size_t len_v, len_xh;
    double reserveFact;
    double v_min, v_max;
    double tfwd, tbwd;
    std::string crossFunction;
    nlohmann::json crossParameters;

    std::vector<double> velocity;
    std::vector<double> sigmaTot;
    std::vector<double> sigmaTrans;
    std::vector<MyFloat> sigmaTotMaxVel;
    std::vector<MyFloat> sigmaTransMaxVel;

    std::vector<double> axis;
    std::vector<std::vector<double>> table;
};

/**
 * @brief Function to write from data structure to json.
 */
inline void to_json(nlohmann::json &j, const DcsFile &f)
{
    j = nlohmann::json{};

    // basic info
    j["version"] = f.version;
    j["method"] = f.method;
    j["len_v"] = f.len_v;
    j["len_xh"] = f.len_xh;
    j["reserveFact"] = f.reserveFact;
    j["v_min"] = f.v_min;
    j["v_max"] = f.v_max;
    j["tfwd"] = f.tfwd;
    j["tbwd"] = f.tbwd;

    // cross section function and parameters
    j["crosssection"] = f.crossFunction;
    j["params"] = f.crossParameters;

    // Table values
    j["velocity"] = f.velocity;
    j["sigmaTot"] = f.sigmaTot;
    j["sigmaTrans"] = f.sigmaTrans;
    j["sigmaTotMaxVel"] = f.sigmaTotMaxVel;
    j["sigmaTransMaxVel"] = f.sigmaTransMaxVel;
    j["axis"] = f.axis;
    j["table"] = f.table;
}

/**
 * @brief Function to write from json to data structure.
 */
inline void from_json(const nlohmann::json &j, DcsFile &f)
{
    // basic info
    j.at("version").get_to(f.version);
    j.at("method").get_to(f.method);
    j.at("len_v").get_to(f.len_v);
    j.at("len_xh").get_to(f.len_xh);
    j.at("reserveFact").get_to(f.reserveFact);
    j.at("v_min").get_to(f.v_min);
    j.at("v_max").get_to(f.v_max);
    j.at("tfwd").get_to(f.tfwd);
    j.at("tbwd").get_to(f.tbwd);

    // cross section function and parameters
    j.at("crosssection").get_to(f.crossFunction);
    j.at("params").get_to(f.crossParameters);

    // Table values
    j.at("velocity").get_to(f.velocity);
    j.at("sigmaTot").get_to(f.sigmaTot);
    j.at("sigmaTrans").get_to(f.sigmaTrans);
    j.at("sigmaTotMaxVel").get_to(f.sigmaTotMaxVel);
    j.at("sigmaTransMaxVel").get_to(f.sigmaTransMaxVel);
    j.at("axis").get_to(f.axis);
    j.at("table").get_to(f.table);
}
