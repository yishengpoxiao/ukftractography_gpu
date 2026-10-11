set(MODULE_NAME UKFGPU)
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS UKFGPU.py UKFGPULib/__init__.py
)
# A runtime staged by scripts/package_slicer_extension.py is optional at build
# time; a user may instead select their own backend in the module.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Runtime")
  install(DIRECTORY Runtime DESTINATION ${Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR}
    USE_SOURCE_PERMISSIONS COMPONENT RuntimeLibraries)
endif()
