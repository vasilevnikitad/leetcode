cmake_minimum_required(VERSION 3.18)

project(leetcode2)

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

add_executable(leetcode main.cpp)

target_link_libraries(leetcode
        PRIVATE
            "Threads::Threads;${GTEST_LIBRARY};${GTEST_MAIN_LIBRARY}"
)
message("${GTEST_LIBRARY};${GTEST_MAIN_LIBRARY}")

set_target_properties(leetcode
        PROPERTIES
            CXX_STANDARD 20
            CXX_EXTENSION OFF
)

target_compile_options(leetcode
        PRIVATE
            "-Wall;-Wextra;-Wpedantic;-Werror"
)

add_test(NAME test-leetcode
        COMMAND
            leetcode
        WORKING_DIRECTORY
            "${CMAKE_SOURCE_DIR}"
)

include(CTest)
