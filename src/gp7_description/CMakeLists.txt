cmake_minimum_required(VERSION 3.8)
project(gp7_description)
find_package(ament_cmake REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
# Kinematic-only playback: PlaybackQuintic.hh and KinematicState.hh are
# header-only helpers exercised by the two compiled regression tests below.
add_library(gp7_kinematic_playback SHARED src/KinematicPlayback.cc)
target_compile_features(gp7_kinematic_playback PRIVATE cxx_std_17)
target_link_libraries(gp7_kinematic_playback PRIVATE gz-sim8::gz-sim8 gz-plugin2::gz-plugin2)
install(TARGETS gp7_kinematic_playback LIBRARY DESTINATION lib)
install(DIRECTORY meshes urdf config scripts worlds DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})
include(CTest)
if(BUILD_TESTING)
  add_executable(gp7_kinematic_state_test test/kinematic_state_test.cc)
  target_compile_features(gp7_kinematic_state_test PRIVATE cxx_std_17)
  target_link_libraries(gp7_kinematic_state_test PRIVATE gz-sim8::gz-sim8)
  add_test(NAME gp7_kinematic_gui_state COMMAND gp7_kinematic_state_test)
  add_executable(gp7_playback_quintic_test test/playback_quintic_test.cc)
  target_compile_features(gp7_playback_quintic_test PRIVATE cxx_std_17)
  add_test(NAME gp7_playback_quintic COMMAND gp7_playback_quintic_test)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME gp7_static_audit
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/audit_model.py
    --output ${CMAKE_CURRENT_BINARY_DIR}/static_audit_ctest.json)
  find_program(CHECK_URDF_EXECUTABLE check_urdf REQUIRED)
  add_test(NAME gp7_urdf_parse
    COMMAND ${CHECK_URDF_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/urdf/gp7_with_gripper.urdf)
  add_test(NAME gp7_kinematics
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_kinematics.py)
  add_test(NAME gp7_kinematics_demo
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_kinematics_demo.py)
  add_test(NAME gp7_given_duration_trajectories
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_joint_trajectories.py)
  set(PICK_PLACE_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv-pick-place/bin/python")
  if(EXISTS "${PICK_PLACE_PYTHON}")
    add_test(NAME gp7_demo_application
      COMMAND ${PICK_PLACE_PYTHON} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/../../tests)
    set_tests_properties(gp7_demo_application PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..
      ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
    add_test(NAME gp7_kinematic_transport
      COMMAND ${PICK_PLACE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_pick_place_playback.py)
    add_test(NAME gp7_cartesian_rest_boundaries
      COMMAND ${PICK_PLACE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_cartesian_boundaries.py)
    add_test(NAME gp7_pick_place_collision
      COMMAND ${PICK_PLACE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_pick_place_collision.py)
    add_test(NAME gp7_pick_place_execution
      COMMAND ${PICK_PLACE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_pick_place_execution.py)
    add_test(NAME gp7_pick_place_sequence
      COMMAND ${PICK_PLACE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_pick_place_sequence.py)
    add_test(NAME gp7_pick_place_branches
      COMMAND ${PICK_PLACE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_pick_place_branches.py)
    add_test(NAME gp7_sprint_preflight
      COMMAND ${PICK_PLACE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_workspace_preflight.py)
    add_test(NAME gp7_pick_place_profiles
      COMMAND ${PICK_PLACE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_pick_place_profiles.py)
    add_test(NAME gp7_pick_place_attachment
      COMMAND ${PICK_PLACE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_pick_place_attachment.py)
    add_test(NAME gp7_pick_place_collision_batch
      COMMAND ${PICK_PLACE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_pick_place_collision_batch.py)
  else()
    message(STATUS "FCL task tests require .venv-pick-place with python-fcl and ROS Python dependencies")
  endif()
  add_test(NAME gp7_pick_place_planner
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_pick_place_planner.py)
  add_test(NAME gp7_pick_place_world
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_pick_place_world.py)
  add_test(NAME gp7_pick_place_pickup_area
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_pick_place_pickup_area.py)
  add_test(NAME gp7_trajectory_preview
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_trajectory_preview.py)
endif()
ament_package()
