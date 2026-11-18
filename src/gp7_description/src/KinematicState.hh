#pragma once
#include <gz/sim/EntityComponentManager.hh>

namespace gp7 {
// Serialized GUI updates must carry the complete pose tree on every publish,
// including held links; otherwise missed endpoint deltas leave pieces stale.
template<typename Component>
void setKinematicState(gz::sim::EntityComponentManager &ecm,
                      gz::sim::Entity entity,const typename Component::Type &value) {
  ecm.SetComponentData<Component>(entity,value);
  ecm.SetChanged(entity,Component::typeId,gz::sim::ComponentState::PeriodicChange);
}
}
