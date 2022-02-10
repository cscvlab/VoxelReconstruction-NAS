# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree

# Utility rule file for embree-download.

# Include the progress variables for this target.
include CMakeFiles/embree-download.dir/progress.make

CMakeFiles/embree-download: CMakeFiles/embree-download-complete


CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-install
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-mkdir
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-download
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-update
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-patch
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-configure
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-build
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-install
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-test
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'embree-download'"
	/usr/bin/cmake -E make_directory /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/CMakeFiles
	/usr/bin/cmake -E touch /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/CMakeFiles/embree-download-complete
	/usr/bin/cmake -E touch /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-done

embree-download-prefix/src/embree-download-stamp/embree-download-install: embree-download-prefix/src/embree-download-stamp/embree-download-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No install step for 'embree-download'"
	cd /home/thomas/shapeMemory/submodules/libigl/build/embree-build && /usr/bin/cmake -E echo_append
	cd /home/thomas/shapeMemory/submodules/libigl/build/embree-build && /usr/bin/cmake -E touch /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-install

embree-download-prefix/src/embree-download-stamp/embree-download-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Creating directories for 'embree-download'"
	/usr/bin/cmake -E make_directory /home/thomas/shapeMemory/submodules/libigl/cmake/../external/embree
	/usr/bin/cmake -E make_directory /home/thomas/shapeMemory/submodules/libigl/build/embree-build
	/usr/bin/cmake -E make_directory /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/embree-download-prefix
	/usr/bin/cmake -E make_directory /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/embree-download-prefix/tmp
	/usr/bin/cmake -E make_directory /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp
	/usr/bin/cmake -E make_directory /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/embree-download-prefix/src
	/usr/bin/cmake -E touch /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-mkdir

embree-download-prefix/src/embree-download-stamp/embree-download-download: embree-download-prefix/src/embree-download-stamp/embree-download-gitinfo.txt
embree-download-prefix/src/embree-download-stamp/embree-download-download: embree-download-prefix/src/embree-download-stamp/embree-download-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'embree-download'"
	cd /home/thomas/shapeMemory/submodules/libigl/external && /usr/bin/cmake -P /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/embree-download-prefix/tmp/embree-download-gitclone.cmake
	cd /home/thomas/shapeMemory/submodules/libigl/external && /usr/bin/cmake -E touch /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-download

embree-download-prefix/src/embree-download-stamp/embree-download-update: embree-download-prefix/src/embree-download-stamp/embree-download-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Performing update step for 'embree-download'"
	cd /home/thomas/shapeMemory/submodules/libigl/external/embree && /usr/bin/cmake -P /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/embree-download-prefix/tmp/embree-download-gitupdate.cmake

embree-download-prefix/src/embree-download-stamp/embree-download-patch: embree-download-prefix/src/embree-download-stamp/embree-download-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "No patch step for 'embree-download'"
	/usr/bin/cmake -E echo_append
	/usr/bin/cmake -E touch /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-patch

embree-download-prefix/src/embree-download-stamp/embree-download-configure: embree-download-prefix/tmp/embree-download-cfgcmd.txt
embree-download-prefix/src/embree-download-stamp/embree-download-configure: embree-download-prefix/src/embree-download-stamp/embree-download-update
embree-download-prefix/src/embree-download-stamp/embree-download-configure: embree-download-prefix/src/embree-download-stamp/embree-download-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No configure step for 'embree-download'"
	cd /home/thomas/shapeMemory/submodules/libigl/build/embree-build && /usr/bin/cmake -E echo_append
	cd /home/thomas/shapeMemory/submodules/libigl/build/embree-build && /usr/bin/cmake -E touch /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-configure

embree-download-prefix/src/embree-download-stamp/embree-download-build: embree-download-prefix/src/embree-download-stamp/embree-download-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No build step for 'embree-download'"
	cd /home/thomas/shapeMemory/submodules/libigl/build/embree-build && /usr/bin/cmake -E echo_append
	cd /home/thomas/shapeMemory/submodules/libigl/build/embree-build && /usr/bin/cmake -E touch /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-build

embree-download-prefix/src/embree-download-stamp/embree-download-test: embree-download-prefix/src/embree-download-stamp/embree-download-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "No test step for 'embree-download'"
	cd /home/thomas/shapeMemory/submodules/libigl/build/embree-build && /usr/bin/cmake -E echo_append
	cd /home/thomas/shapeMemory/submodules/libigl/build/embree-build && /usr/bin/cmake -E touch /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-test

embree-download: CMakeFiles/embree-download
embree-download: CMakeFiles/embree-download-complete
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-install
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-mkdir
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-download
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-update
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-patch
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-configure
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-build
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-test
embree-download: CMakeFiles/embree-download.dir/build.make

.PHONY : embree-download

# Rule to build all files generated by this target.
CMakeFiles/embree-download.dir/build: embree-download

.PHONY : CMakeFiles/embree-download.dir/build

CMakeFiles/embree-download.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/embree-download.dir/cmake_clean.cmake
.PHONY : CMakeFiles/embree-download.dir/clean

CMakeFiles/embree-download.dir/depend:
	cd /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree /home/thomas/shapeMemory/submodules/libigl/external/.cache/embree/CMakeFiles/embree-download.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/embree-download.dir/depend

