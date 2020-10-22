# Minimal CMakeLists.txt suggested by deal.ii website

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FIND_PACKAGE(deal.II 9.2.0 REQUIRED
  HINTS  ../ ../../ {DEAL_II_DIR}
  )
DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(London-Staged-Solver)

ADD_EXECUTABLE(solver solver.cc)
DEAL_II_SETUP_TARGET(solver)

# Custom targets to simplify commands used to build in debug / release mode

ADD_CUSTOM_TARGET(debug
  COMMAND  -DCMAKE_BUILD_TYPE=Debug 
  COMMAND  --build  --target all
  COMMENT Switch CMAKE_BUILD_TYPE to Debug
  )

ADD_CUSTOM_TARGET(release
  COMMAND  -DCMAKE_BUILD_TYPE=Release 
  COMMAND  --build  --target all
  COMMENT Switch CMAKE_BUILD_TYPE to Release
  )
