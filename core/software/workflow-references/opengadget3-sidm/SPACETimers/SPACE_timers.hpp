/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the class TimerHierarchy which is used for timing the code.
 *
 * The TimerHierarchy class, which is a stack-based hierarchical timing system.
 * It contains all functions necessary to use the timer class. The macros to use the timer class
 * in OG are defined in allvars.hpp.
 *
 * @author OpenGadget3-development team, in particular Geray S. Karademir.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef TIMERS_HPP
#define TIMERS_HPP

#include <map>
#include <mpi.h>
#include <string>
#include <vector>
#include <cstdio>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <iostream>
#include <algorithm>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

// Stack-based hierarchical timing system
class TimerHierarchy
{
  private:
    int lenOutput = 100;
    double TotalTime = 0;

    // vector of sybols to use for writing of balance file
    std::vector<std::string_view> symbols{
        "!", "#", "@", "$", "%", "^", "&", "*", "(", "-", "=", "+", "{", "[", "|", "/", "<", ";", ".", "'", "]",
        ")", "}", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "Q", "W", "E", "R", "T", "Y", "U", "I", "O",
        "P", "A", "S", "D", "F", "G", "H", "J", "K", "L", "Z", "X", "C", "V", "B", "N", "M", "q", "w", "e", "r",
        "t", "y", "u", "i", "o", "p", "a", "s", "d", "f", "g", "h", "j", "k", "l", "z", "x", "c", "v", "b", "n",
        "m", "ä", "ü", "ö", "ß", "Ü", "Ö", "Ä", "§", "²", "³", "~", "_", "`", "´", "°", "€", "£", "¥", "¢", "₩",
        "₽", "₹", "¬", "∧", "∨", "∀", "∃", "∈", "∉", "⊂", "⊃", "→", "←", "↑", "↓", "↔", "⇒", "⇐", "⇔"};

    using clock = std::chrono::steady_clock;
    std::map<std::string, std::tuple<int, std::string, int, int>>
        timer_write_map; // map with tuples containing: key (name), level, symbol, time, sublevels

    class tstack_node
    {
      private:
        using maptype = std::map<std::string, tstack_node>;
        using Tipair = std::pair<maptype::const_iterator, double>;

      public:
        tstack_node *parent;
        std::string name;
        double accTime;
        maptype child;

      private:
        /**
         * Calculates the total accumulated time
         *
         * This function recursively calculates the total accumulated time of the current node
         * and all its child nodes.
         *
         * @return The total accumulated time
         */
        double full_acc() const
        {
            double t_own = accTime;
            // Iterate through each child node and accumulate their time
            for (const auto &nd : child)
            {
                t_own += nd.second.full_acc();
            }
            return t_own;
        }

        /**
         * @brief Calculates the maximum length of the name among the current node and its children.
         *
         * This function recursively calculates the maximum length of the name among the current node
         * and all its child nodes.
         *
         * @return The maximum length of the name
         */
        size_t max_namelen() const
        {
            auto res = name.length();
            // Iterate through each child node and find the maximum name length among them
            for (const auto &ch : child)
            {
                res = std::max(res, ch.second.max_namelen());
            }
            return res;
        }

        /**
         * @brief Counts the number of colons in a string.
         *
         * This function counts the number of colons in a string to get the level of the hierarchy.
         *
         * @return The number of colons in the string.
         */
        int countColons(const std::string &str) const
        {
            int count = std::count_if(str.begin(), str.end(), [](char c) { return c == ':'; });
            return count;
        }

        /**
         * @brief Formats a floating-point value for output.
         *
         * This function formats a floating-point value for output with a specified number
         * of digits before and after the decimal point.
         *
         * @param val The floating-point value to be formatted.
         * @param pre The number of digits before the decimal point.
         * @param post The number of digits after the decimal point.
         * @param os The output stream to write the formatted value to.
         */
        static void floatformat(double val, size_t pre, size_t post, std::ostream &os)
        {
            size_t fct = 1;
            // Calculate the factor for rounding
            for (size_t i = 0; i < post; ++i, fct *= 10)
                ;
            // Write the formatted value to the output stream
            os << std::setw(pre) << int(val) << "." << std::setw(post) << std::setfill('0')
               << int((val - int(val)) * fct + 0.5) << std::setfill(' ');
        }

        /**
         * Writes a line to the output stream.
         *
         * This function prints a line wiht timing information to the specified output stream, considering indentation,
         * name, value, total, and formatting options.
         *
         * @tparam Output The type of the output stream.
         * @param indent The indentation string.
         * @param twidth The total width for formatting.
         * @param slen The length of the name.
         * @param name The name of the timing line.
         * @param val The value of the timing line.
         * @param total The total value for percentage calculation.
         * @param os The output stream to write the timing line to.
         */
        template <typename Output>
        static void printline(const std::string &indent, int twidth, int slen, const std::string &name, double val,
                              double total, Output &os)
        {
            // only print down to a certain level
            if (int(indent.size() / 2) <= All.TimerReportLevel)
            {
                int width = std::max(0, static_cast<int>(slen) + 1 - static_cast<int>(name.length()) -
                                            static_cast<int>(indent.size()));
                os << indent << Timer_List_Symbol << " " << name << std::setw(width) << " : ";
                // Write the value with specified formatting
                floatformat(val, twidth - 5, 4, os);
                os << " sec, ";
                // Calculate and write the percentage
                floatformat(100 * val / total, 3, 2, os);
                os << "% "
                   << " \n";
            }
        }

        /**
         * @brief Report the timing information.
         *
         * This function is the main function to generate a report with the timing information,
         * either in long or hierarchical format, and writes it to the specified output stream,
         * which can either be a file or the console.
         *
         * @tparam Output The type of the output stream.
         * @param indent The indentation string.
         * @param twidth The total width for formatting.
         * @param slen The length of the name.
         * @param os The output stream to write the report to.
         */
        template <typename Output> void report(const std::string &indent, int twidth, int slen, Output &os) const
        {
            // Get total time
            double total_time = full_acc();

            std::vector<Tipair> tmp;
            // Store child nodes and their accumulated times in a vector
            for (auto it = child.cbegin(); it != child.cend(); ++it)
            {
                tmp.push_back(std::make_pair(it, it->second.full_acc()));
            }

            if (tmp.size() > 0)
            {
                // Sort child nodes based on accumulated times
                sort(tmp.begin(), tmp.end(), [](const Tipair &a, const Tipair &b) { return a.second > b.second; });

                double tsum = 0;
                // Iterate through sorted child nodes and print timing information
                for (unsigned i = 0; i < tmp.size(); ++i)
                {
                    printline(indent, twidth, slen, tmp[i].first->first, tmp[i].second, total_time, os);
                    // Recursively call report for child nodes
                    (tmp[i].first->second).report(indent + Timer_Indent_Symbol + " ", twidth, slen, os);
                    tsum += tmp[i].second;
                }
                // print Unaccounted time if it exists
                if (tsum < 0.999 * total_time)
                {
                    printline(indent, twidth, slen, "Unaccounted", total_time - tsum, total_time, os);
                }
            }
        }

      public:
        /**
         * @brief Constructor for the timer stack node.
         */
        explicit tstack_node(const std::string &name_, tstack_node *parent_ = nullptr)
            : parent(parent_), name(name_), accTime(0.)
        {
        }

        /**
         * @brief Generates a report of timing information.
         *
         * This function generates a report of timing information and writes it to the specified output stream.
         * It calculates the total accumulated time, sets up formatting parameters, and calls the main report function.
         *
         * @tparam Output The type of the output stream.
         * @param os The output stream to write the report to.
         */
        template <typename Output> void report(Output &os, bool generate_balance_stats) const
        {
            // Calculate the length of the longest name plus some padding
            auto slen = std::string("Unaccounted").size();
            slen = std::max(slen, max_namelen()) + 10;

            // Get the total accumulated time
            double total = full_acc();

            // Write header information to the output stream
            os << "\nStep " << All.NumCurrentTiStep << " Time: a=" << All.Time << ", MPI-Tasks: " << NTask
               << " Task:" << ThisTask << "\nTotal wall clock time for " << name << " = " << total << " sec\n";

#ifdef TIMER_DETAILS
            // This block calculates balance statistics across all ranks.
            // The generate_balance_stats variable and separate if block is necessary as GlobalTimers are only called by
            // rank 0 and this creates a deadlock otherwise.
            if (generate_balance_stats)
            {
                // Get min/max values
                double min_total = 0.0, max_total = 0.0;
                MPI_Allreduce(&total, &min_total, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(&total, &max_total, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
       // Do summation of variables
                double total_vals[2] = {total, total * total};
                double total_out[2];
                MPI_Allreduce(total_vals, total_out, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                double sum_total = total_out[0];
                double sum_sq_total = total_out[1];

                // Calc values
                double mean_total = sum_total / NTask;
                double variance = (sum_sq_total / NTask) - (mean_total * mean_total);

                // Numerical guard (as could go negative due to FP rounding)
                variance = std::max(0.0, variance);
                // Standard deviation
                double stddev = std::sqrt(variance);

                os << "Stats across ranks:\n"
                   << "min=" << min_total << " sec, max=" << max_total << " sec, mean=" << mean_total
                   << " sec, stddev=" << stddev << " sec\n";
            }
#endif // TIMER_DETAILS

            // Determine the number of digits in the total time for formatting
            int logtime = std::max(1, int(log10(total) + 1));

            // Call the main report function to generate the report
            report("", logtime + 5, slen, os);
        }

        /**
         * @brief Adds time to the total accumulated time.
         *
         * This function adds the specified time increment to the total accumulated time.
         *
         * @param dt The time to add.
         */
        void addTime(double dt)
        {
            accTime += dt;
        }

        /**
         * @brief Sums all timings recursively and stores them in a map.
         *
         * This function recursively sums all timings of the current node and its children,
         * storing the results in the provided map with the specified prefix.
         *
         * @param prefix The prefix to prepend to the names of the timings.
         * @param res The map to store the summed timings.
         * @return The total accumulated time including its children.
         */
        double add_timings(const std::string &prefix, std::map<std::string, double> &res) const
        {
            // Initialize the total time with the accumulated time of the current node
            double t_own = accTime;
            // Iterate through each child node and sum their timings recursively
            for (const auto &nd : child)
            {
                // Recursively sum timings of the child node with updated prefix
                t_own += nd.second.add_timings(prefix + ":" + nd.first, res);
            }
            // Store the total accumulated time with the provided prefix in the result map
            res[prefix] = t_own;
            return t_own;
        }

        /**
         * Helper function to serialize the timer node and its children to a stream.
         *
         * This function serializes the current timer node and its children to the specified output stream,
         * with indentation for hierarchical representation.
         *
         * @param os The output stream to serialize to. Can be either a file or the console.
         * @param indent The indentation string.
         */
        void serializeHelper(std::ostream &os, const std::string &indent) const
        {
            os << indent << name << " " << accTime << " " << child.size() << "\n";
            // Serialize each child node recursively
            for (const auto &[key, value] : child)
            {
                os << indent << "  " << key << " " << value.accTime << "\n";

                // Recursive call to serialize child's members
                value.serializeHelper(os, indent + "    ");
            }
        }

        /**
         * @brief Serializes the timer node and its children to a stream.
         *
         * This function serializes the entire timer node hierarchy to the specified output stream.
         *
         * @param os The output stream to serialize to.
         */
        void serialize(std::ostream &os) const
        {
            // Call the helper function with an empty indentation string to start serialization
            serializeHelper(os, "");
        }

        /**
         * @brief Deletes a node with the specified name from the current node's children.
         *
         * This function searches for a child node with the specified name and deletes it, including
         * all its descendants.
         *
         * @param node_name The name of the node to delete.
         * @return True if the node was found and deleted, false otherwise.
         */
        bool delete_node(const std::string &node_name)
        {
            // Search for the node in the children map
            auto it = child.find(node_name);
            if (it != child.end())
            {
                // Node found, delete it
                child.erase(it);
                return true;
            }

            // Recursively search in each child's children
            for (auto &ch : child)
            {
                if (ch.second.delete_node(node_name))
                {
                    return true; // Node found and deleted in a descendant
                }
            }
            return false; // Node not found
        }
        /**
         * @brief Retrieve the accumulated execution time for a specified region.
         *
         * This method searches for a region identified by @p region_name within the
         * current node's immediate children. If the region is not found at this level,
         * the search proceeds recursively through all descendant nodes.
         *
         * The function returns the total accumulated time associated with the first
         * matching region encountered during the search.
         *
         * @param region_name Name of the region whose accumulated time is requested.
         * @return Accumulated time for the specified region, or 0 if the region
         *         cannot be found in the hierarchy.
         *
         * @note A return value of 0 indicates either that the region does not exist
         *       or that the region exists but has no accumulated time.
         */
        double get_region_time(const std::string &region_name)
        {
            // Search for the node in the children map
            auto it = child.find(region_name);
            if (it != child.end())
            {
                double region_time = it->second.full_acc();
                return region_time;
            }

            // Recursively search in each child's children
            for (auto &ch : child)
            {
                double region_time = ch.second.get_region_time(region_name);
                if (region_time != 0.0)
                {
                    return region_time;
                }
            }
            return 0; // Return 0 if region not found
        }
    };

    clock::time_point last_time;
    tstack_node root;
    tstack_node *curnode;
    bool intialistaion_is_deleted = false;

    /**
     * @brief Helper function to deserialize the timer from a stream.
     *
     * This function deserializes the timer node hierarchy from the specified input stream.
     * It reads the data for each node and its children recursively.
     *
     * @param is The input stream to deserialize from.
     */
    void deserializeHelper(std::istream &is)
    {
        size_t numChildren;
        std::string name;
        double accTime;
        is >> name >> accTime >> numChildren;
        // If root name is identical to name, then update root
        if (root.name == name)
        {
            // Update root time
            root.accTime = accTime;
        }
        else
        {
            // Insert new node into map with name and value
            auto it = curnode->child.insert(std::make_pair(name, tstack_node(name, curnode))).first;
            it->second.accTime = accTime;

            // Go down to child level
            curnode = &(it->second);
        }

        for (size_t i = 0; i < numChildren; ++i)
        {
            std::string childName;
            double childAccTime;

            // Get input from stream
            is >> childName >> childAccTime;

            // Create a new TimerNode with the deserialized information
            auto it = curnode->child.insert(std::make_pair(childName, tstack_node(childName, curnode))).first;
            it->second.accTime = childAccTime;

            // Recursive call to deserialize child's members
            this->deserializeHelper(is);
        }

        if (curnode->name != root.name)
        {
            // Jump back up to parent
            curnode = curnode->parent;
        }
    }

    /**
     * @brief Deserializes the timer node from a stream.
     *
     * This function deserializes the entire timer node hierarchy from the specified input stream.
     *
     * @param is The input stream to deserialize from.
     */
    void deserialize(std::istream &is)
    {
        // Call the helper function to start deserialization
        deserializeHelper(is);
    }

    /**
     * @brief Adds elapsed time to the current timer node and updates the last time.
     *
     * This function adds the elapsed time since the last call to the current timer node's accumulated time.
     * It also updates the last time to the current time.
     */
    void adjust_time()
    {
        // Get current time
        auto tnow = clock::now();
        // Calculate the elapsed time since the last call and add it to the current node's accumulated time
        curnode->addTime(std::chrono::duration<double>(tnow - last_time).count());
        // Update the last time to the current time
        last_time = tnow;
    }

    /**
     * @brief Pushes a new timer node with the given name onto the stack.
     *
     * This function creates a new timer node with the specified name and adds it as a child of the current node.
     * If a node with the same name already exists, it simply moves the current node to that existing node.
     *
     * @param name The name of the timer node to be pushed onto the stack.
     */
    void push_internal(const std::string &name)
    {
        // Find if a child node with the given name already exists
        auto it = curnode->child.find(name);
        // If the node doesn't exist, insert a new one
        if (it == curnode->child.end())
        {
            // Check for reserved characters in the name
            VERBOSE_IF(name.find(':') != std::string::npos, 0, "Timer: Assertion failure - : - is reserved character!");
            // Insert a new node into the map with the provided name and parent node
            it = curnode->child.insert(std::make_pair(name, tstack_node(name, curnode))).first;
        }
        // Move to the newly pushed or existing node
        curnode = &(it->second);
    }

    TimerHierarchy(const TimerHierarchy &) = delete;
    TimerHierarchy(TimerHierarchy &&) = delete;
    TimerHierarchy &operator=(const TimerHierarchy &) = delete;
    TimerHierarchy &operator=(TimerHierarchy &&) = delete;

    /**
     * @brief Updates the total run-time for the current timestep.
     *
     * This function updates the total run-time for the current timestep by querying the timing information
     * and extracting the total time from the results.
     */
    void update_total_time()
    {
        // Get timing information
        auto timer_results = this->get_timings();
        // Get the total time from the first entry in the timing results
        auto it = timer_results.begin();
        TotalTime = it->second;
    }

    /**
     * @brief Retrieves headers from the timing map.
     *
     * This function retrieves headers from the timing map to later printing to file.
     * It calculates the fractions of time for each sublevel and stores the information in a map.
     */
    void get_header_map()
    {
        // Get timing results
        auto timer_results = this->get_timings();
        // Vector to store cumulative time for each level
        std::vector<double> cummulative_time(All.TimerReportBalance + 2);

        int level_old = 0;
        // Loop to go through map from end but skipping first element of map
        for (auto it = timer_results.rbegin(); it != std::prev(timer_results.rend()); ++it)
        {
            const auto &[key, value] = *it;
            int current_level = std::count(key.begin(), key.end(), ':');

            if (current_level <= All.TimerReportBalance)
            {
                int fraction = 0;
                int subregion = 0;
                // Sum times to get fractions of each sublevel
                cummulative_time[current_level] += value;

                // If previous elements were subregion...
                if (level_old > current_level)
                {
                    subregion = 1;
                    fraction = (int)((value - cummulative_time[current_level + 1]) / TotalTime * lenOutput);
                    cummulative_time[current_level + 1] = 0;
                }
                else
                {
                    fraction = (int)(value / TotalTime * lenOutput);
                }
                // Write information to map
                write_key_to_timer_write_map(current_level, key, fraction, subregion);
                level_old = current_level;
            }
        }
    }

    /**
     * @brief Writes the header file for balance.txt.
     *
     * This function opens the balance header file, writes the headers, and iterates through the timer write map
     * to write the names and symbols to the file. It also adds an entry for "Unaccounted" to the header file.
     */
    void write_header_to_file()
    {
        // Open the balance header file and write the initial header
        std::ofstream FdBalanceHeader;
        std::string filePath = std::string(All.OutputDir) + All.BalanceFileHead;
        FdBalanceHeader.open(filePath);

        // Check if the file is opened successfully
        if (!(FdBalanceHeader.is_open()))
        {
            ENDRUN("Error in opening balance header file.");
        }
        else
        {
            FdBalanceHeader << "Headers for balance file:" << std::endl;
        }

        // Iterate through map and write name and symbols to file
        for (auto it = timer_write_map.begin(); it != timer_write_map.end(); ++it)
        {
            const auto &tupleValue = it->second;
            const int level = std::get<0>(tupleValue);
            // Only print keys with level above report level
            if (level <= All.TimerReportBalance)
            {
                const std::string &key = it->first;
                const std::string &symbol = std::get<1>(tupleValue);
                FdBalanceHeader << "'" << symbol << "' - " << key << std::endl;
            }
        }
        // Add unaccounted to header file
        FdBalanceHeader << "'?' - Unaccounted" << std::endl;
        FdBalanceHeader.flush();
        FdBalanceHeader.close();
    }

    /**
     * @brief This function resets the timer write map.
     *
     * It set all fractions values in the timer_write_map to 0 to have a clean map.
     */
    void reset_timer_write_map()
    {
        // Iterater through all map entries and set the values to 0
        for (auto it = timer_write_map.begin(); it != timer_write_map.end(); ++it)
        {
            std::get<2>(it->second) = 0;
        }
    }

    /**
     * @brief This function matches the headers of the timer write map with the timing results.
     *
     * In addition to matching the headers it also calculates the fractions of time for each sublevel.
     * If a new timer is added, it will write a new balance header file including the updated headers.
     */
    void match_timer_headers()
    {
        auto timer_results = this->get_timings();

        // Set all fraction in map to 0
        reset_timer_write_map();

        // Sized by TimerReportBalance (the bound actually used for indexing below), not
        // TimerReportLevel — see get_header_map() for why.
        std::vector<double> cummulative_time(All.TimerReportBalance + 2);

        int level_old = 0;
        int new_timer = 0;
        // Loop through map from end but skipping first element of map
        for (auto it = timer_results.rbegin(); it != std::prev(timer_results.rend()); ++it)
        {
            const auto &[key, value] = *it;

            // Get current level
            int current_level = std::count(key.begin(), key.end(), ':');

            if (current_level <= All.TimerReportBalance)
            {
                int fraction = 0;
                int subregion = 0;
                // Find name
                size_t firstColonPos = key.find(":");

                // Get matched element between maps
                auto element = timer_write_map.find(key.substr(firstColonPos + 1));

                // Sum times to get fractions of each sublevel
                cummulative_time[current_level] += value;

                // If previous elements were subregions
                if (level_old > current_level)
                {
                    // Calculate fraction based on subregion
                    fraction = (int)((value - cummulative_time[current_level + 1]) / TotalTime * lenOutput);
                    cummulative_time[current_level + 1] = 0;
                    subregion = 1;
                }
                else
                {
                    // Calculate fraction based on total time
                    fraction = (int)(value / TotalTime * lenOutput);
                }

                // Check if the key was found and if levels agree
                if ((element != timer_write_map.end()) && (current_level == std::get<0>(element->second)))
                {
                    // Update the second int value
                    std::get<2>(element->second) = fraction;
                }
                else
                {
                    // If header not found, then add new key to map
                    VERBOSE(3, "TIMERS: region name: %s not used before. -> Adding to header.", key.c_str());
                    write_key_to_timer_write_map(current_level, key, fraction, subregion);
                    // Check if new timer is added
                    if (current_level <= All.TimerReportBalance)
                    {
                        new_timer = 1;
                    }
                }
                level_old = current_level;
            }
        }

        if (new_timer == 1)
        {
            VERBOSE_ALL(2, "Writing new balance header file");
            // Write headers to file
            write_header_to_file();
        }
    }

    /**
     * @brief This function writes the keys to the timer write map.
     *
     * It writes the keys to the timer write map with the level, symbol, fraction, and subregion.
     */
    void write_key_to_timer_write_map(int level, const std::string &key, int fraction, int subregion)
    {
        std::string symbol = " ";
        // Find name
        size_t firstColonPos = key.find(":");
        // Check if there are symbols left
        if (symbols.empty())
        {
            VERBOSE(0, "Running out of symbols... - reduce number of timers!");
        }
        // Get symbol if level is below report level
        if (level <= All.TimerReportBalance)
        {
            symbol = symbols[0];
            symbols.erase(symbols.begin());
        }
        // Write information to map
        timer_write_map[key.substr(firstColonPos + 1)] = std::make_tuple(level, symbol, fraction, subregion);
    }

    /**
     * @brief This function writes the symbols to the output balance file.
     *
     * It writes the symbols to the output balance file based on the fractions of time for each sublevel.
     */
    void write_symbols_to_file(std::ofstream &outputFile)
    {
        // Write header for timestep to output file
        outputFile << "Step=" << std::setw(5) << All.NumCurrentTiStep << " sec=" << std::setw(10) << TotalTime
                   << " Nf= " << std::setw(10) << GlobNumForceUpdate << "    ";
        int sum_fractions = 0;

        std::string SymbolString;
        // Iterate through map and write symbols to string
        for (auto it = timer_write_map.begin(); it != timer_write_map.end(); ++it)
        {
            const auto &tupleValue = it->second;
            const int level = std::get<0>(tupleValue);

            if (level <= All.TimerReportBalance)
            {
                // Access tuple elements
                const std::string &symbol = std::get<1>(tupleValue);
                int fraction = std::get<2>(tupleValue);

                // Sum over all timers to get total time
                sum_fractions += fraction;
                // Add symbols to string based on fraction
                for (int i = 0; i < fraction; i++)
                {
                    SymbolString += symbol;
                }
            }
        }
        // Add unaccounted symbols at the end
        for (int i = 0; i < lenOutput - sum_fractions; i++)
        {
            SymbolString += "?";
        }

        // Write string to file
        outputFile << std::setw(lenOutput) << SymbolString;
        outputFile << std::endl;
        outputFile.flush();
    }

    /**
     * @brief Create indent for output.
     *
     * This function adds blocks for intendation of output
     *
     * @param limit Number of blocks to add.
     * @param output The output stream to write the indentation to.
     */
    template <typename Output> void output_indent(int limit, Output &output)
    {
        // Add N block of spaces for indentation
        for (int i = 1; i <= limit; i++)
        {
            output << "   ";
        }
    }

    /**
     * @brief Pop timer region from stack.
     *
     * This function removes the top category from the stack with no labels for internal use.
     *
     * @param file The file where the function is called.
     * @param line The line where the function is called.
     */
    void pop(const char *file, int line)
    {
        // Adjust time
        adjust_time();
        // Move to parent node
        curnode = curnode->parent;
        // Check if stack is empty
        if (curnode == nullptr)
        {
            {
                ENDRUN("Timer: Assertion failure - tried to pop from empty timer stack! File: %s, Line: %d\n", file,
                       line);
            }
        }
    }

  public:
    /**
     * @brief Constructor for TimerHierarchy with name.
     *
     * @param name The name of the timer.
     */
    explicit TimerHierarchy(const std::string &name = "<root>")
        : last_time(clock::now()), root(name, nullptr), curnode(&root)
    {
    }

    /**
     * @brief Constructor for TimerHierarchy with name and firsttimer.
     *
     * @param name The name of the timer.
     * @param firsttimer The name of the first timer.
     */
    TimerHierarchy(const std::string &name, const std::string &firsttimer) : TimerHierarchy(name)
    {
        push(firsttimer);
    }

    /**
     * @brief Get total time for timer class
     *
     * This function is a helper function to get total time of total timer class
     *
     * @return Total time of timer class
     */
    double get_total_time()
    {
        // Get timing results
        auto timer_results = this->get_timings();
        // Get first element of map
        auto it = timer_results.begin();
        // Get value/time
        double local_TotalTime = it->second;
        return local_TotalTime;
    }

    /**
     * @brief Get total time for a specific region
     *
     * This function is a helper function to get the total time of a specific region
     *
     * @param region_name name of the region
     *
     * @return Total time of this region
     */
    double get_region_time(std::string region_name)
    {
        double region_time = root.get_region_time(region_name);
        return region_time;
    }

    /**
     * @brief Push new timer region to stack.
     *
     * This function pushes a new category with the specified name to the stack.
     *
     * @param name The name of the category to be pushed onto the stack.
     */
    void push(const std::string &name)
    {
        adjust_time();
        // Push new timer region to stack
        push_internal(name);
    }

    /**
     * @brief Pop timer region from stack.
     *
     * This function pops the top category from the stack with the specified label.
     *
     * @param label The label of the category to be popped from the stack.
     * @param file The file where the function is called.
     * @param line The line where the function is called.
     */
    void pop(const std::string &label, const char *file, int line)
    {
        // Check if label is the same as the top category
        if (curnode->name == label)
        {
            adjust_time();
            // Go to parent node
            curnode = curnode->parent;
            // Check if stack is empty
            if (curnode == nullptr)
            {
                {
                    ENDRUN("Timer: Assertion failure - tried to pop from empty timer stack! File: %s, Line: %d\n", file,
                           line);
                }
            }
        }
        else // If label is not the same as the top category - ERROR!
        {
            ENDRUN("Timer: Assertion failure in File: %s, Line: %d\n- tried to pop from %s but %s is active.\n", file,
                   line, label.c_str(), curnode->name.c_str());
        }
    }

    /**
     * @brief Replace the top category on the stack.
     *
     * This function pops the top category from the stack and pushes a new category with the specified name.
     *
     * @param name_pop The name of the category to be popped from the stack.
     * @param name_push The name of the category to be pushed onto the stack.
     * @param file The file where the function is called.
     * @param line The line where the function is called.
     */
    void poppush(const std::string &name_pop, const std::string &name_push, const char *file, int line)
    {
        // Pop the top category from the stack
        pop(name_pop, file, line);
        // Push a new category onto the stack
        push_internal(name_push);
    }

    /**
     * @brief Reset timers.
     *
     * This function resets all timers or only the specified ones.
     *
     * @param name The name of the timer to be reset. Leave empty if all timers should be reset.
     */
    void reset(std::string name = "")
    {
        // If no name is found, reset all timers
        if (name == "")
        {
            last_time = clock::now();
            root = tstack_node(root.name, nullptr);
            curnode = &root;
        }
        else
        {
            auto it = curnode->child.find(name);
            // If found reset
            if (it != curnode->child.end())
            {
                it->second = tstack_node(name, curnode);
                curnode = &(it->second);
            }
            else
            {
                VERBOSE_ALL(0, "Timer: Assertion failure - timer %s not found!", name.c_str());
            }
        }
    }

    /**
     * @brief Deletes a node from the stack.
     *
     * This function calls a function of the root class to delete the region and all its children.
     *
     * @param node_name The name of the node to delete.
     */
    void delete_node(const std::string &node_name)
    {
        if (intialistaion_is_deleted)
        {
            return; // Exit if intialistion was already deleted
        }
        else
        {
            root.delete_node(node_name);
            intialistaion_is_deleted = true;
        }
    }

    /**
     * @brief Write symbols to balance file.
     *
     * This function writes the symbols to the balance file based on the fractions of time for each sublevel.
     * It also creates/updates the header file and writes it to a separate header file.
     *
     * @param outputFile The output file stream to write the symbols to.
     */
    void write_symbols(std::ofstream &outputFile)
    {
        // Update total time
        update_total_time();

        // Check if first timestep
        if (All.NumCurrentTiStep != 0)
        {
            // Check if maps are equal - match headers and update times
            match_timer_headers();
        }
        else
        {
            // Create map with values as its the first time
            get_header_map();

            // Write map to header file
            write_header_to_file();
        }

        // After new/update map is created write map to symbols
        write_symbols_to_file(outputFile);
    }

    /**
     * @brief This class automatically pushes and pops a category.
     */
    class ScopeTimer
    {
      private:
        TimerHierarchy &hierarchy;

      public:
        ScopeTimer() = delete;
        ScopeTimer(const ScopeTimer &) = delete;
        ScopeTimer &operator=(const ScopeTimer &) const = delete;

        /**
         * @brief Constructor for ScopeTimer.
         */
        ScopeTimer(const std::string &name, TimerHierarchy &hierarchy_) : hierarchy(hierarchy_)
        {
            hierarchy.push(name);
        }

        /**
         * @brief Destructor for ScopeTimer.
         */
        ~ScopeTimer()
        {
            hierarchy.pop(__FILE__, __LINE__);
        }
    };

    /**
     * @brief Creates a ScopeTimer object.
     *
     * This function creates a ScopeTimer object with the specified name and TimerHierarchy.
     *
     * @param name The name of the timer.
     * @return The ScopeTimer object.
     */
    ScopeTimer scopeTimer(const std::string &name)
    {
        return ScopeTimer(name, *this);
    }

    /**
     * @brief Get timings.
     *
     * This gets the accumulated time for all regions and returns it as a map.
     *
     * @return The map containing all categories and their times.
     */
    std::map<std::string, double> get_timings()
    {
        adjust_time();
        std::map<std::string, double> res;
        // Get all timings from map
        root.add_timings(root.name, res);
        return res;
    }

    /**
     * @brief Report the timing information.
     *
     * This function is the first to be called by the code to generate a report with the timing information, either in
     * long or hierarchical format. It checks if the output stream is a file stream and flushes it if necessary.
     *
     * @param os The output stream to write the report to.
     */
    template <typename Output> void report(Output &os, bool generate_balance_stats = false) const
    {
        // Call the report function to generate the output
        root.report(os, generate_balance_stats);

        // Check if output is file stream and flush if
        if constexpr (std::is_same<Output, std::ofstream>::value)
        {
            // Check if file is open and flush
            if (os.is_open())
            {
                os.flush();
            }
            else
            {
                std::cerr << "Timer Error: Unable to open file for writing." << std::endl;
            }
        }
    }

    /**
     * @brief Write timer hierarchy to a file.
     *
     * This function serializes the entire timer hierarchy to the specified file
     * for the restart.
     *
     * @param filename The name of the file to serialize the timer hierarchy to.
     */
    void serializeToFile(const std::string &filename) const
    {
        std::ofstream outfile(filename);
        // Check if file is open and serialize
        if (outfile.is_open())
        {
            root.serialize(outfile);
            outfile.close();
            VERBOSE(1, "Timer written to file %s successfully.", filename.c_str());
        }
        else
        {
            ENDRUN("Timer Error: Unable to open file %s for writing.\n", filename.c_str());
        }
    }

    /**
     * @brief Write timer write map to file.
     *
     * This function writes the timer_write_map to the specified file for
     * the proper restart of the balance file.
     *
     * @param filename The name of the file where the map is stored.
     */
    void write_balance_to_file(const std::string &filename) const
    {
        std::ofstream outfile(filename);
        // Check if file is open and serialize
        if (outfile.is_open())
        {
            for (const auto &pair : timer_write_map)
            {
                const std::string &key = pair.first;
                const auto &tuple = pair.second;

                outfile << key << ' ' << std::get<0>(tuple) << ' ' << std::get<1>(tuple) << ' ' << std::get<2>(tuple)
                        << ' ' << std::get<3>(tuple) << '\n';
            }

            outfile.close();
            VERBOSE(1, "Balance written to file %s successfully.", filename.c_str());
        }
        else
        {
            ENDRUN("Timer Error: Unable to open file %s for writing.\n", filename.c_str());
        }
    }

    /**
     * @brief Read timer hierarchy from a file.
     *
     * This function deserializes the entire timer hierarchy from the specified file.
     *
     * @param filename The name of the file to deserialize the timer hierarchy from.
     */
    void deserializeFromFile(const std::string &filename)
    {
        std::ifstream infile(filename);
        // Check if file is open and deserialize
        if (infile.is_open())
        {
            deserialize(infile);
            infile.close();
            VERBOSE(1, "Timer read from file %s successfully.", filename.c_str());
        }
        else
        {
            ENDRUN("Timer Error: Unable to open file %s for readin.\n", filename.c_str());
        }
    }

    /**
     * @brief Read timer write map from a file.
     *
     * This function reads the timer_write_map from the specified file for
     * the proper restart of the balance file.
     *
     * @param filename The name of the file where the map is stored.
     */
    void read_timer_write_map(const std::string &filename)
    {
        std::ifstream infile(filename);
        // Check if file is open and then read content to map
        if (infile.is_open())
        {
            std::string key;
            int int1, int2, int3;
            std::string str;

            // populate map with values from file
            while (infile >> key >> int1 >> str >> int2 >> int3)
            {
                timer_write_map[key] = std::make_tuple(int1, str, int2, int3);
            }

            infile.close();
            VERBOSE(1, "Timer read from file %s successfully.", filename.c_str());
        }
        else
        {
            ENDRUN("Timer Error: Unable to open file %s for readin.\n", filename.c_str());
        }
    }
};

/**
 * @brief Combines two timing maps.
 *
 * This function combines two timing maps by adding the values of the same keys.
 * It returns the combined map.
 *
 * @param m1 The first timing map.
 * @param m2 The second timing map.
 * @return The combined timing map.
 */
inline std::map<std::string, double> combine_timings(const std::map<std::string, double> &m1,
                                                     const std::map<std::string, double> &m2)
{
    std::map<std::string, double> res = m1;
    // Iterate through second map and add values to first map
    for (const auto &[k, v] : m2)
    {
        auto iter = res.find(k);
        if (iter == res.end())
        {
            res[k] = v;
        }
        else
        {
            iter->second += v;
        }
    }
    return res;
}
// this always needs to be defined
#define TIMER_NEW_INSTANCE(TimerClass) extern class TimerHierarchy TimerClass;

// Load macros depending on use case
#ifndef SWITCH_TIMER_OFF

#ifdef TIMER_VTUNE
#include "timer_ITT.hpp"
#elif defined(TIMER_NVTX)
#include "timer_NVTX.hpp"
#elif defined(TIMER_ROCTX)
#include "timer_ROCTX.hpp"
#elif defined(TIMER_OMNITRACE)
#include "timer_OMNITRACE.hpp"
#elif defined(TIMER_MERIC)
#include "timer_CPU.hpp"
#else // Now default cpu timer macros
#include "timer_CPU.hpp"
#endif // TIMER selection

#else // SWITCH_TIMER_OFF
#include "timer_OFF.hpp"
#endif // SWITCH_TIMER_OFF

#include "timer_USER.hpp"

#endif // TIMERS_HPP
