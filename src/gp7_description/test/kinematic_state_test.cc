// Regression: changing server pose data is insufficient for GUI ECM updates.
#include "../src/KinematicState.hh"
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/JointPosition.hh>
#include <iostream>
#include <stdexcept>
int main() {
  gz::sim::EntityComponentManager ecm;
  const auto entity=ecm.CreateEntity();
  using Pose=gz::sim::components::Pose;
  ecm.CreateComponent(entity,Pose(gz::math::Pose3d::Zero));
  ecm.SetChanged(entity,Pose::typeId,gz::sim::ComponentState::NoChange);
  ecm.SetComponentData<Pose>(entity,gz::math::Pose3d(1,0,0,0,0,0));
  if(ecm.ComponentState(entity,Pose::typeId)!=gz::sim::ComponentState::NoChange)
    throw std::runtime_error("Baseline API semantics changed; reconsider regression");
  gp7::setKinematicState<Pose>(ecm,entity,gz::math::Pose3d(2,0,0,0,0,0));
  if(ecm.ComponentState(entity,Pose::typeId)!=gz::sim::ComponentState::PeriodicChange)
    throw std::runtime_error("Pose was not marked for GUI state synchronization");
  ecm.SetChanged(entity,Pose::typeId,gz::sim::ComponentState::NoChange);
  gp7::setKinematicState<Pose>(ecm,entity,gz::math::Pose3d(2,0,0,0,0,0));
  if(ecm.ComponentState(entity,Pose::typeId)!=gz::sim::ComponentState::PeriodicChange)
    throw std::runtime_error("Held links disappeared from the coherent GUI pose stream");
  using Joint=gz::sim::components::JointPosition;
  ecm.CreateComponent(entity,Joint({0.}));
  ecm.SetChanged(entity,Joint::typeId,gz::sim::ComponentState::NoChange);
  gp7::setKinematicState<Joint>(ecm,entity,{.2});
  if(ecm.ComponentState(entity,Joint::typeId)!=gz::sim::ComponentState::PeriodicChange)
    throw std::runtime_error("Joint state was not marked for GUI synchronization");
  std::cout<<"PASS: raw data updates leave GUI state unchanged; kinematic pose/joint updates mark PeriodicChange\n";
}
