# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/x86_64/Cmake/3.14.3/Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/x86_64/Cmake/3.14.3/Linux-x86_64/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/draguet/QGdis/GranularDataGatherer/source

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/draguet/QGdis/GranularDataGatherer/build

# Utility rule file for NightlyUpdate.

# Include the progress variables for this target.
include CMakeFiles/NightlyUpdate.dir/progress.make

CMakeFiles/NightlyUpdate:
	/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/x86_64/Cmake/3.14.3/Linux-x86_64/bin/ctest -D NightlyUpdate

NightlyUpdate: CMakeFiles/NightlyUpdate
NightlyUpdate: CMakeFiles/NightlyUpdate.dir/build.make

.PHONY : NightlyUpdate

# Rule to build all files generated by this target.
CMakeFiles/NightlyUpdate.dir/build: NightlyUpdate

.PHONY : CMakeFiles/NightlyUpdate.dir/build

CMakeFiles/NightlyUpdate.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/NightlyUpdate.dir/cmake_clean.cmake
.PHONY : CMakeFiles/NightlyUpdate.dir/clean

CMakeFiles/NightlyUpdate.dir/depend:
	cd /home/draguet/QGdis/GranularDataGatherer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/draguet/QGdis/GranularDataGatherer/source /home/draguet/QGdis/GranularDataGatherer/source /home/draguet/QGdis/GranularDataGatherer/build /home/draguet/QGdis/GranularDataGatherer/build /home/draguet/QGdis/GranularDataGatherer/build/CMakeFiles/NightlyUpdate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/NightlyUpdate.dir/depend

