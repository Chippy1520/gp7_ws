// Coursework-only kinematic animation. No physics, effort servo, PID, or grasp force.
// The checked q(t) is reconstructed on simulation ticks and written to native
// joint-state and link-pose components. Cube attachment is an explicit pose relation.
#include "PlaybackQuintic.hh"
#include "KinematicState.hh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/JointPosition.hh>
#include <gz/sim/components/JointVelocity.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/PoseCmd.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/joint_trajectory.pb.h>
#include <gz/msgs/stringmsg.pb.h>

namespace gp7 {
class KinematicPlayback final : public gz::sim::System,
    public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate {
  using Entity=gz::sim::Entity;
  using Pose=gz::math::Pose3d;
  struct Edge {std::string parent,child,kind;gz::math::Vector3d axis;int index;};
  Entity robot{},cube{gz::sim::kNullEntity};
  std::array<Entity,8> joints{};
  std::array<std::string,8> names{};
  State lower{},upper{},speed{};
  std::map<std::string,Pose> home,poses;
  std::map<std::string,Entity> links;
  std::vector<Edge> edges;
  gz::transport::Node node;
  gz::transport::Node::Publisher statusPub,graspPub;
  std::mutex mutex;
  gz::msgs::JointTrajectory pending;
  bool newMessage=false,holdRequested=false,wantedGrasp=false,attached=false;
  std::vector<double> times;
  std::vector<PlaybackSample> points;
  PlaybackSample current;
  Pose relation;
  double start=0.,lastTime=-1.;
  std::string activeId,statusId,statusState="idle",reason;
  std::chrono::steady_clock::time_point lastPublish{};
  std::chrono::steady_clock::time_point heartbeat=std::chrono::steady_clock::now();

  void Status() {
    gz::msgs::StringMsg msg;
    std::ostringstream out;
    out<<"{\"state\":"<<std::quoted(statusState)<<",\"id\":"<<std::quoted(statusId)
       <<",\"reason\":"<<std::quoted(reason)<<",\"scope\":\"kinematic state; physics disabled\"}";
    msg.set_data(out.str());statusPub.Publish(msg);
  }
  void Reject(const std::string &id,const std::string &why) {
    times.clear();points.clear();current.v.fill(0.);current.a.fill(0.);
    statusId=id;statusState="rejected";reason=why;Status();
  }
  Pose Motion(const Edge &edge) const {
    const double q=edge.index<0?0.:current.q[edge.index];
    if (edge.kind=="prismatic")return Pose(edge.axis*q,gz::math::Quaterniond::Identity);
    if (edge.kind=="revolute")return Pose(gz::math::Vector3d::Zero,gz::math::Quaterniond(edge.axis,q));
    return Pose::Zero;
  }
  void Forward(gz::sim::EntityComponentManager &ecm) {
    poses.clear();std::vector<Edge> remaining=edges;
    while (!remaining.empty()) {
      bool progress=false;
      for (auto it=remaining.begin();it!=remaining.end();) {
        const bool root=home.count(it->parent)==0;
        if (!root && !poses.count(it->parent)) {++it;continue;}
        const Pose parent=root?Pose::Zero:poses.at(it->parent);
        const Pose parentHome=root?Pose::Zero:home.at(it->parent);
        poses[it->child]=parent*parentHome.Inverse()*home.at(it->child)*Motion(*it);
        it=remaining.erase(it);progress=true;
      }
      if (!progress)throw std::runtime_error("Kinematic link graph is disconnected/cyclic");
    }
    for (const auto &[name,entity]:links) {
      if (!poses.count(name))poses[name]=home.at(name);
      setKinematicState<gz::sim::components::Pose>(ecm,entity,poses.at(name));
    }
    for (size_t i=0;i<8;++i) {
      setKinematicState<gz::sim::components::JointPosition>(ecm,joints[i],{current.q[i]});
      setKinematicState<gz::sim::components::JointVelocity>(ecm,joints[i],{current.v[i]});
    }
  }
  void Accept(const gz::msgs::JointTrajectory &msg,double now) {
    std::string id;
    for (const auto &entry:msg.header().data())
      if (entry.key()=="segment_id" && entry.value_size())id=entry.value(0);
    if (id.empty() || id.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-")!=std::string::npos) {
      Reject("","Missing/invalid segment ID");return;
    }
    if (id==activeId) {Status();return;}  // Retransmission never restarts motion.
    if (msg.joint_names_size()!=8 || msg.points_size()<2) {Reject(id,"Expected eight joints and at least two knots");return;}
    for (int i=0;i<8;++i)if(msg.joint_names(i)!=names[i]) {Reject(id,"Joint order mismatch");return;}
    std::vector<double> nextTimes;std::vector<PlaybackSample> nextPoints;
    for (const auto &point:msg.points()) {
      const double t=point.time_from_start().sec()+point.time_from_start().nsec()*1e-9;
      if (!std::isfinite(t) || t<0. || (!nextTimes.empty() && t<=nextTimes.back()) ||
          point.positions_size()!=8 || point.velocities_size()!=8 || point.accelerations_size()!=8) {
        Reject(id,"Invalid knot time or dimensions");return;
      }
      PlaybackSample state;
      for (int i=0;i<8;++i) {
        state.q[i]=point.positions(i);state.v[i]=point.velocities(i);state.a[i]=point.accelerations(i);
        if (!std::isfinite(state.q[i]) || !std::isfinite(state.v[i]) || !std::isfinite(state.a[i]) ||
            state.q[i]<lower[i]-1e-9 || state.q[i]>upper[i]+1e-9 || std::abs(state.v[i])>speed[i]+1e-6) {
          Reject(id,"Nonfinite or out-of-envelope knot");return;
        }
      }
      nextTimes.push_back(t);nextPoints.push_back(state);
    }
    const double nextStart=msg.header().stamp().sec()+msg.header().stamp().nsec()*1e-9;
    if (nextTimes.front()!=0. || nextStart<0. || nextStart>now+2. || nextStart<now-1.) {
      Reject(id,"Invalid/stale simulation start time");return;
    }
    for (int i=0;i<8;++i)if(std::abs(nextPoints.front().q[i]-current.q[i])>(i<6?.003:.0002)) {
      Reject(id,"Trajectory start does not match held kinematic state");return;
    }
    // Phase endpoints must be stationary: a completed or stopped playback cannot
    // silently discard a nonzero endpoint velocity/acceleration.
    for (int i=0;i<8;++i)if(std::abs(nextPoints.front().v[i])>1e-6 ||
        std::abs(nextPoints.front().a[i])>1e-5 ||
        std::abs(nextPoints.back().v[i])>1e-6 || std::abs(nextPoints.back().a[i])>1e-5) {
      Reject(id,"Nonstationary phase boundary");return;
    }
    times=std::move(nextTimes);points=std::move(nextPoints);start=nextStart;
    activeId=id;statusId=id;statusState="accepted";reason.clear();Status();
  }
 public:
  void Configure(const Entity &entity,const std::shared_ptr<const sdf::Element> &input,
      gz::sim::EntityComponentManager &ecm,gz::sim::EventManager &) override {
    robot=entity;gz::sim::Model model(robot);auto sdf=input->Clone();
    for (auto e=sdf->GetElement("home_link");e;e=e->GetNextElement("home_link")) {
      const auto name=e->Get<std::string>("name");home[name]=e->Get<Pose>("pose");
      links[name]=model.LinkByName(ecm,name);
      if(links[name]==gz::sim::kNullEntity)throw std::runtime_error("Missing kinematic link "+name);
    }
    for(auto e=sdf->GetElement("kinematic_joint");e;e=e->GetNextElement("kinematic_joint")) {
      Edge edge{e->Get<std::string>("parent"),e->Get<std::string>("child"),
                e->Get<std::string>("kind"),e->Get<gz::math::Vector3d>("axis"),e->Get<int>("index")};
      edges.push_back(edge);
      if(edge.index>=0) {
        const auto i=edge.index;if(i>=8)throw std::runtime_error("Invalid kinematic joint index");
        names[i]=e->Get<std::string>("name");joints[i]=model.JointByName(ecm,names[i]);
        lower[i]=e->Get<double>("lower");upper[i]=e->Get<double>("upper");speed[i]=e->Get<double>("speed");
        if(joints[i]==gz::sim::kNullEntity)throw std::runtime_error("Missing kinematic joint "+names[i]);
        if(!ecm.Component<gz::sim::components::JointPosition>(joints[i]))
          ecm.CreateComponent(joints[i],gz::sim::components::JointPosition({0.}));
        if(!ecm.Component<gz::sim::components::JointVelocity>(joints[i]))
          ecm.CreateComponent(joints[i],gz::sim::components::JointVelocity({0.}));
      }
    }
    statusPub=node.Advertise<gz::msgs::StringMsg>("/gp7/kinematic/status");
    graspPub=node.Advertise<gz::msgs::Boolean>("/pick_place/grasp/attached");
    node.Subscribe("/gp7/kinematic/trajectory",&KinematicPlayback::Trajectory,this);
    node.Subscribe("/gp7/kinematic/hold",&KinematicPlayback::Hold,this);
    node.Subscribe("/gp7/kinematic/heartbeat",&KinematicPlayback::Heartbeat,this);
    node.Subscribe("/pick_place/grasp/enable",&KinematicPlayback::Grasp,this);
    Forward(ecm);
  }
  void Trajectory(const gz::msgs::JointTrajectory &msg) {
    std::lock_guard<std::mutex> lock(mutex);pending=msg;newMessage=true;
  }
  void Hold(const gz::msgs::Boolean &msg) {
    if(msg.data()){std::lock_guard<std::mutex> lock(mutex);holdRequested=true;}
  }
  void Grasp(const gz::msgs::Boolean &msg) {
    std::lock_guard<std::mutex> lock(mutex);wantedGrasp=msg.data();
  }
  void Heartbeat(const gz::msgs::Boolean &msg) {
    if(msg.data()){std::lock_guard<std::mutex> lock(mutex);heartbeat=std::chrono::steady_clock::now();}
  }
  void PreUpdate(const gz::sim::UpdateInfo &info,gz::sim::EntityComponentManager &ecm) override {
    // UserCommands normally hands pose changes to Physics. This mode has no
    // Physics system, so consume world-model pose commands here (also paused).
    std::vector<std::pair<Entity,Pose>> moves;
    ecm.Each<gz::sim::components::Model,gz::sim::components::WorldPoseCmd>(
      [&](Entity e,const auto *,const auto *cmd){moves.emplace_back(e,cmd->Data());return true;});
    for(const auto &[e,p]:moves) {
      if(!(e==cube && attached))setKinematicState<gz::sim::components::Pose>(ecm,e,p);
      ecm.RemoveComponent<gz::sim::components::WorldPoseCmd>(e);
    }
    if(cube==gz::sim::kNullEntity)cube=ecm.EntityByComponents(
      gz::sim::components::Model(),gz::sim::components::Name("box_b"));
    if(info.paused)return;
    const double now=std::chrono::duration<double>(info.simTime).count();
    if(lastTime>=0. && now<lastTime)Reject(activeId,"Backward simulation time jump; hold");
    lastTime=now;
    bool want,hold;gz::msgs::JointTrajectory msg;bool received;std::chrono::steady_clock::time_point alive;
    {std::lock_guard<std::mutex> lock(mutex);want=wantedGrasp;hold=holdRequested;holdRequested=false;
      alive=heartbeat;received=newMessage;if(received){msg=pending;newMessage=false;}}
    if(hold){times.clear();points.clear();current.v.fill(0.);current.a.fill(0.);statusState="held";Status();}
    // Hold wins over a simultaneous upload. A later explicit upload may resume.
    if(received && !hold) {
      Accept(msg,now);
      if(statusState=="accepted") {
        // A fresh accepted upload itself proves the owner is alive. Otherwise
        // topic callback ordering after an idle planning period can falsely
        // trip the watchdog before the first heartbeat packet is processed.
        std::lock_guard<std::mutex> lock(mutex);heartbeat=alive=std::chrono::steady_clock::now();
      }
    }
    if(!times.empty() && std::chrono::steady_clock::now()-alive>std::chrono::milliseconds(500))
      Reject(activeId,"Lost command heartbeat; kinematic hold");
    if(!times.empty()) {
      const auto previous=current;
      const double t=std::clamp(now-start,0.,times.back());
      if(t>=times.back()){current=points.back();times.clear();points.clear();}
      else {
        const size_t right=std::upper_bound(times.begin(),times.end(),t)-times.begin();
        current=interpolate(points[right-1],points[right],times[right]-times[right-1],t-times[right-1]);
      }
      for(int i=0;i<8;++i)if(current.q[i]<lower[i]-1e-8 || current.q[i]>upper[i]+1e-8 ||
          !std::isfinite(current.q[i]) || std::abs(current.v[i])>speed[i]+1e-5) {
        // Never render an out-of-envelope state: restore the last valid state.
        current=previous;
        Reject(activeId,"Interpolated state outside planned position/speed envelope");break;
      }
    }
    Forward(ecm);
    const Pose palm=gz::sim::worldPose(robot,ecm)*poses.at("gripper_palm");
    if(want && !attached && cube!=gz::sim::kNullEntity) {
      relation=palm.Inverse()*gz::sim::worldPose(cube,ecm);attached=true;
    }
    if(!want)attached=false;
    if(cube!=gz::sim::kNullEntity) {
      const auto *pose=ecm.Component<gz::sim::components::Pose>(cube);
      if(pose)setKinematicState<gz::sim::components::Pose>(ecm,cube,attached?palm*relation:pose->Data());
    }
    const auto wall=std::chrono::steady_clock::now();
    if(wall-lastPublish>std::chrono::milliseconds(25)) {
      gz::msgs::Boolean state;state.set_data(attached);graspPub.Publish(state);Status();lastPublish=wall;
    }
  }
};
}
GZ_ADD_PLUGIN(gp7::KinematicPlayback,gz::sim::System,
  gp7::KinematicPlayback::ISystemConfigure,gp7::KinematicPlayback::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(gp7::KinematicPlayback,"gp7::KinematicPlayback")
