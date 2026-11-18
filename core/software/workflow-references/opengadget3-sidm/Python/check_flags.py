#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to check if activated flags in Config file are availiable in the code
"""
import re
import os
import sys
from typing import List, Union, Set, Tuple, Dict


class ConfigFile:
    """
    Class to parse a configuration file and extract active and inactive flags.

    Attributes
    ----------
    config_filename : str
        Path to the configuration file to be parsed.
    """

    def __init__(self, config_file: str):
        """
        Initialize ConfigFile with the given configuration file path.

        Parameters
        ----------
        config_file : str, optional
            Path to the configuration file (default is None).
        """
        self.config_filename = config_file

    def get_flags_from_configfile(self) -> Tuple[Set[str], Set[str]]:
        """
        Extract active and inactive flags from the configuration file.

        Active flags are detected as capitalized identifiers at the beginning
        of each line. Inactive flags are identified in commented lines.

        Returns
        -------
        Tuple[Set[str], Set[str]]
            A tuple containing:
            - set of active flags
            - set of inactive flags
        """
        found_flags: Set[str] = set()
        found_inactive_flags: Set[str] = set()
        active_flag_counts: Dict[str, int] = {}
        inactive_flag_counts: Dict[str, int] = {}

        with open(self.config_filename, "r") as file_content:
            for line in file_content:
                words = line.split()
                if not words:
                    continue

                # Extract active flags from first word of line
                active_found = re.findall(r"^([a-zA-Z_][a-zA-Z_0-9]*)", words[0])
                for flag in active_found:
                    found_flags.add(flag)
                    active_flag_counts[flag] = active_flag_counts.get(flag, 0) + 1

                # Extract inactive flags from commented lines
                if words[0].startswith('#'):
                    inactive_found = re.findall(
                        r"^([a-zA-Z_][a-zA-Z_0-9]*)", words[0][1:]
                    )
                    for flag in inactive_found: 
                        found_inactive_flags.add(flag)
                        inactive_flag_counts[flag] = (
                            inactive_flag_counts.get(flag, 0) + 1
                        )

        self.check_flag_counts(active_flag_counts, inactive_flag_counts)
        return found_flags, found_inactive_flags
    

    def check_flag_counts(
        self, active_flag_counts: Dict[str, int], inactive_flag_counts: Dict[str, int]
    ) -> None:
        """
        Check for duplicate active or inactive flags and exit if found.

        Parameters
        ----------
        active_flag_counts : Dict[str, int]
            Dictionary with counts of active flags.
        inactive_flag_counts : Dict[str, int]
            Dictionary with counts of inactive flags.
        """
        active_duplicates = [
            (flag, count) for flag, count in active_flag_counts.items() if count > 1
        ]
        inactive_duplicates = [
            (flag, count) for flag, count in inactive_flag_counts.items() if count > 1
        ]

        if active_duplicates:
            output_string = (
                "\nConfig check:\nERROR: The following active flags have duplicates:\n"
            )
            for flag, count in active_duplicates:
                output_string += f"{flag} found {count} times\n"

            if inactive_duplicates:
                output_string += (
                    "\nWarning: The following inactive flags have duplicates:\n"
                )
                for flag, count in inactive_duplicates:
                    output_string += f"{flag} found {count} times\n"

            output_string += "\nPlease remove all duplicates.\n"
            write_str_to_file(output_string)
            sys.exit(1)


class CodeFiles:
    """
    Class to extract compilation flags from source code files.

    Attributes
    ----------
    compilation_filelist : List[str]
        List of files obtained from the Makefile or command-line arguments.
    """

    def __init__(self, filelist: List[str]):
        self.compilation_filelist = filelist

    def get_flags(self) -> Tuple[List[str], List[str]]:
        """
        Get allowed flags from compiled files and all flags from the project.

        Returns
        -------
        Tuple[List[str], List[str]]
            - allowed_flags: flags found in files actually used during compilation
            - all_flags: flags found in all source/header files
        """
        # get files
        file_list_all, file_list = self.get_file_list()

        # get flags which are part of compiled code
        allowed_flags = self.get_flags_from_ch_files(file_list)

        # print(sorted(allowed_flags))
        # check for all flags
        all_flags = self.get_flags_from_ch_files(file_list_all)

        return allowed_flags, all_flags

    def get_file_list(self) -> Tuple[List[str], List[str]]:
        """
        Determine which files to search for flags.

        Returns
        -------
        Tuple[List[str], List[str]]
            - file_list_all: all .c/.cpp/.h/.hpp files in the directory
            - file_list: intersection of Makefile-used files and all available files
        """
        file_list_makefile = self.get_make_used_filenames()
        file_list_all = self.get_all_files()
        file_list = list(set(file_list_makefile) & set(file_list_all))
        
        #exclude all files in libs dir -> dont check external libs
        file_list = [f for f in file_list if "libs" not in os.path.normpath(f).split(os.sep)]
        file_list_all = [f for f in file_list_all if "libs" not in os.path.normpath(f).split(os.sep)]
        
        return file_list_all, file_list

    def get_make_used_filenames(self) -> List[str]:
        """
        Generate a list of source and header file paths based on object files from the Makefile.

        This function converts object files ('.o') in `self.compilation_filelist` into
        possible source file paths ('.c', '.cpp') and keeps existing header files ('.h', '.hpp').
        All paths are prefixed with the current working directory. Duplicates are removed.

        Returns
        -------
        List[str]
            List of absolute paths to source and header files used in compilation.
        """
        # get current dir
        current_dir = os.path.abspath(os.getcwd())

        header_files = [
            os.path.join(current_dir, f)
            for f in self.compilation_filelist
            if f.endswith((".hpp", ".h"))
        ]
        object_files = [
            os.path.join(current_dir, f)
            for f in self.compilation_filelist
            if f.endswith(".o")
        ]

        # Convert object files to source files. Only the ".o" suffix may be replaced.
        stems = [f[: -len(".o")] for f in object_files]
        combined_list_c = [stem + ".c" for stem in stems]
        combined_list_cpp = [stem + ".cpp" for stem in stems]

        # Extra check in case not all header files are added to makefile -> add potential header file with same name as object file
        combined_list_h = [stem + ".h" for stem in stems]
        combined_list_hpp = [stem + ".hpp" for stem in stems]
        
        # remove duplicates
        return list(set(header_files + combined_list_c + combined_list_cpp + combined_list_h + combined_list_hpp))

    def get_all_files(self) -> List[str]:
        """
        Retrieve all source/header files from the current directory recursively.

        Returns
        -------
        List[str]
            List of absolute paths to .c, .cpp, .h, .hpp files.
        """
        current_dir = os.path.abspath(os.getcwd())
        file_list = []
        excluded_dirs = {"libs", "tests", "Python"}

        for dirpath, dirnames, filenames in os.walk(current_dir):
            dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
            for filename in filenames:
                if filename.endswith((".c", ".cpp", ".h", ".hpp")):
                    file_list.append(os.path.join(dirpath, filename))

        return file_list

    def get_flags_from_ch_files(self, file_list: List[str]) -> List[str]:
        """
        Extract all possible flags from a list of C/C++ source/header files.

        Parameters
        ----------
        file_list : List[str]
            List of files to scan.

        Returns
        -------
        List[str]
            List of flags found in the files (duplicates removed).
        """
        found_flags = []

        for file in file_list:
            consider_next_line = 0
            try:
                with open(file, "r") as file_content:
                    for line in file_content:
                        line_string = line.lstrip()
                        if not line_string:
                            continue
                        split_str = line_string.split()

                        # Determine if this line should be analyzed for flags
                        if consider_next_line or line_string.startswith(
                            ("#if", "#elif", "#elseif")
                        ):
                            found_flags = self.return_if(
                                split_str, found_flags, line_string
                            )
                            consider_next_line = self.check_line_break(line_string)
            except OSError:
                print(f"Cannot find/open {file}")
                continue

        # Remove duplicates while preserving order
        return list(dict.fromkeys(found_flags))

    def return_if(
        self, split_str: List[str], found_flags: List[str], line_string: str
    ) -> List[str]:
        """
        Parse a line to extract flags, ignoring comments.

        Parameters
        ----------
        split_str : List[str]
            Line split into words.
        found_flags : List[str]
            Current list of found flags.
        line_string : str
            Original line string.

        Returns
        -------
        List[str]
            Updated list of found flags.
        """
        inside_comment = False

        for item in split_str:
            # Skip line if single-line comment or closing comment without opening
            if item.startswith("//") or (
                "*/" in line_string and "/*" not in line_string
            ):
                break

            # Handle multi-line comment start/end
            if item.startswith("/*"):
                inside_comment = True
                continue
            if item.startswith("*/"):
                inside_comment = False
                continue

            if inside_comment or item.startswith("#"):
                continue

            # Remove leading parentheses
            item = item.lstrip("(")

            # Detect flags by removing common C/C++ macro patterns
            if item.startswith("define"):
                flag_expr = item[7:]
            elif item.startswith(("(!defined", "!(defined")):
                flag_expr = item[9:]
            elif item.startswith(("!defined", "(defined")):
                flag_expr = item[8:]
            else:
                flag_expr = item.split("(")[0]
            
            found_flags.extend(re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", flag_expr))

        return found_flags

    def check_line_break(self, line_string: str) -> int:
        """
        Determine if the next line should be considered due to a trailing backslash.

        Parameters
        ----------
        line_string : str
            Line string to check.

        Returns
        -------
        int
            1 if the line ends with '\', else 0.
        """
        return 1 if line_string.rstrip().endswith("\\") else 0


def get_diff_flags(used_flags: Set[str], available_flags: Set[str]) -> Set[str]:
    """
    Compute the difference between used flags and available flags.

    Parameters
    ----------
    used_flags : Set[str]
        Set of flags that are currently used in the configuration.
    available_flags : Set[str]
        Set of flags that are allowed or present in the code.

    Returns
    -------
    Set[str]
        Set of flags that are used but not available in the code.
    """
    return used_flags.difference(available_flags)


def inactive_flags_output(
    diff_inactive_flags: Union[List[str], int], output_string: str
) -> str:
    """
    Append information about inactive flags to the output string.

    This function checks whether inactive flags are missing, unavailable,
    or all available, and appends an appropriate message to the provided
    output string.

    Parameters
    ----------
    diff_inactive_flags : List[str] or int
        List of inactive flags that are not available in the code, or 0 if no inactive flags exist.
    output_string : str
        The existing output string to which the inactive flag information will be appended.

    Returns
    -------
    str
        Updated output string with inactive flag information appended.
    """
    if diff_inactive_flags == 0:
        output_string += "\nNo inactive flags found.\n"
    elif diff_inactive_flags:  # non-empty list
        output_string += (
            "\nERROR: The following inactive flags are not available in the code:\n"
        )
        output_string += "\n".join(diff_inactive_flags) + "\n"
    else:  # empty list
        output_string += "\nAll inactive flags are available in the code.\n"

    return output_string


def write_str_to_file(output_string: str) -> int:
    """
    Print a string to the terminal and write it to an output file.

    This function writes the provided string to a file named `flagcheck.out`
    and also prints the string to the console.

    Parameters
    ----------
    output_string : str
        The text to print and save to the output file.

    Returns
    -------
    int
        Always returns 0 to indicate success.
    """
    # Use context manager to safely open and close the file
    with open("flagcheck.out", "w") as output_file:
        print(output_string)  # Print to terminal
        output_file.write(output_string)  # Write to file

    return 0


def check_code_flags(used_config_filename: str, files_from_makefile: List[str]) -> None:
    """
    Validate configuration flags against the source code.

    This function compares the active and inactive flags defined in a configuration
    file against the flags found in the source code files. It generates a report
    indicating whether any active flags are missing or if inactive flags are
    outdated/unavailable.

    The results are written to an output file, and the function exits with:
      - 0 if all active flags are valid
      - 1 if any active flags are missing

    Parameters
    ----------
    used_config_filename : str
        Path to the configuration file containing used and inactive flags.
    files_from_makefile : List[str]
        List of source code file paths to scan for allowed flags.

    Returns
    -------
    None
        The function writes output to a file and exits the program.
    """
    # check config file for used flags
    Config_File = ConfigFile(used_config_filename)
    used_flags, inactive_flags = Config_File.get_flags_from_configfile()

    # check for allowed flags
    CodeFlags = CodeFiles(files_from_makefile)
    allowed_flags, all_flags = CodeFlags.get_flags()

    # get diff between used flags and flags in files used during compilation
    diff_active = get_diff_flags(used_flags, allowed_flags)

    # Get diff between inactive flags and all availiable flags
    diff_inactive = get_diff_flags(inactive_flags, all_flags) if inactive_flags else []

    # Build output
    if diff_active:
        output = "\nERROR: The following active flags are not available:\n"
        output += "\n".join(diff_active) + "\n"
    else:
        output = "\nAll active flags are available in the code.\n"

    # Add inactive flags info
    output = inactive_flags_output(diff_inactive, output)

    # Write output and exit
    write_str_to_file(output)
    sys.exit(1 if diff_active else 0)


if __name__ == "__main__":
    """
    Entry point for the script.

    This block parses command-line arguments to retrieve the configuration file
    and the list of source files to analyze. It then calls the
    `check_code_flags` function to validate configuration flags.

    Command-line usage:
        python script.py <config_file> <source_file1> <source_file2> ...

    Parameters
    ----------
    sys.argv[1] : str
        Path to the configuration file.
    sys.argv[2:] : List[str]
        List of source code files to scan for flags.
    """
    # check if flags are in code by getting all flags from code and then comparing it
    check_code_flags(sys.argv[1], sys.argv[2:])
