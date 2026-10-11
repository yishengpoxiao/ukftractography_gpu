
set(proj pnlNipype)

# Set dependency list
set(${proj}_DEPENDENCIES "")

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj DEPENDS_VAR ${proj}_DEPENDENCIES)

set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})
set(EP_BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
set(EP_INSTALL_DIR ${CMAKE_BINARY_DIR}/${proj}-install)

ExternalProject_SetIfNotDefined(
  ${CMAKE_PROJECT_NAME}_${proj}_GIT_REPOSITORY
  "https://github.com/pnlbwh/pnlNipype.git"
  QUIET
  )

ExternalProject_SetIfNotDefined(
  ${CMAKE_PROJECT_NAME}_${proj}_GIT_TAG
  "7e0e1e1ac0a77d3013bb889ebe95aab901def3d8"
  QUIET
  )

ExternalProject_Add(${proj}
  ${${proj}_EP_ARGS}
  GIT_REPOSITORY "${${CMAKE_PROJECT_NAME}_${proj}_GIT_REPOSITORY}"
  GIT_TAG "${${CMAKE_PROJECT_NAME}_${proj}_GIT_TAG}"
  SOURCE_DIR ${EP_SOURCE_DIR}
  BINARY_DIR ${EP_BINARY_DIR}
  INSTALL_DIR ${EP_INSTALL_DIR}
  DEPENDS
    ${${proj}_DEPENDENCIES}
  )


