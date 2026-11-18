#!/usr/bin/env python3
"""GP7 exported-coordinate kinematics. Not a manufacturer calibration model."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

PI = math.pi
D1, A1, A2, A3, D4, D6 = .330, .040, .445, .040, .440, .155
NAMES = ['joint_s', 'joint_l', 'joint_u', 'joint_r', 'joint_b', 'joint_t']
URDF = Path(__file__).resolve().parents[1] / 'urdf/gp7_with_gripper.urdf'


def dh(theta, d, a, alpha):
    """Standard DH: Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    c, s, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array([[c,-s*ca,s*sa,a*c], [s,c*ca,-c*sa,a*s],
                     [0,sa,ca,d], [0,0,0,1]], dtype=float)


def dh_rows(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('Expected six finite joint values in radians')
    return [(q[0],D1,A1,PI/2), (q[1]+PI/2,0,A2,0),
            (q[2],0,A3,PI/2), (q[3],D4,0,-PI/2),
            (q[4],0,0,PI/2), (q[5],D6,0,0)]


def fk_dh(q):
    T = np.eye(4)
    for row in dh_rows(q):
        T = T @ dh(*row)
    return T


def rpy_matrix(rpy):
    r,p,y = rpy
    cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def so3_log(R):
    cos_angle = np.clip((np.trace(R)-1)/2,-1,1)
    angle = math.acos(float(cos_angle))
    vector = np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    if angle < 1e-7:
        return vector/2
    if PI-angle < 1e-6:
        vals,vecs = np.linalg.eig(R)
        axis = np.real(vecs[:,np.argmin(abs(vals-1))])
        axis /= np.linalg.norm(axis)
        if np.dot(axis,vector)<0:
            axis = -axis
        return axis*angle
    return vector * angle/(2*math.sin(angle))


class ExportedModel:
    """Exact recorded URDF fixed origins; no geometry rounding."""
    def __init__(self, path=URDF):
        robot = ET.parse(path).getroot()
        self.origins = []
        self.axes = []
        for name in NAMES:
            j = robot.find(f"joint[@name='{name}']")
            o = j.find('origin')
            T = np.eye(4)
            T[:3,3] = np.fromstring(o.get('xyz'),sep=' ')
            T[:3,:3] = rpy_matrix(np.fromstring(o.get('rpy'),sep=' '))
            self.origins.append(T)
            self.axes.append(np.fromstring(j.find('axis').get('xyz'),sep=' '))
        self.tool = np.eye(4)
        for name in ['gripper_mount','gripper_tcp_fixed']:
            o=robot.find(f"joint[@name='{name}']/origin")
            T=np.eye(4)
            T[:3,3]=np.fromstring(o.get('xyz'),sep=' ')
            T[:3,:3]=rpy_matrix(np.fromstring(o.get('rpy'),sep=' '))
            self.tool=self.tool@T

    def _forward(self,q,jacobian):
        dh_rows(q)  # validates input
        T=np.eye(4)
        origins,axes=[],[]
        for fixed,axis,angle in zip(self.origins,self.axes,q):
            T=T@fixed
            if jacobian:
                origins.append(T[:3,3].copy())
                axes.append(T[:3,:3]@axis)
            x,y,z=axis
            K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            motion=np.eye(4)
            motion[:3,:3]=np.eye(3)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)
            T=T@motion
        T=T@self.tool
        if not jacobian:return T
        axes=np.asarray(axes)
        J=np.empty((6,6))
        J[:3,:]=np.cross(axes,T[:3,3]-np.asarray(origins)).T
        J[3:,:]=axes.T
        return T,J

    def evaluate(self,q):
        return self._forward(q,True)

    def fk(self,q):
        return self._forward(q,False)


def validate_pose(T):
    T=np.asarray(T,dtype=float)
    if T.shape!=(4,4) or not np.isfinite(T).all():
        raise ValueError('Expected a finite 4x4 target transform')
    R=T[:3,:3]
    if not np.allclose(T[3],[0,0,0,1],atol=1e-9) or not np.allclose(R.T@R,np.eye(3),atol=1e-8) or abs(np.linalg.det(R)-1)>1e-8:
        raise ValueError('Target must be a rigid pose with proper rotation')
    return T


def near_seed(q,seed):
    return seed + (np.asarray(q)-seed+PI)%(2*PI)-PI


def ik_analytic(target,seed=None):
    """All geometric branches of the ideal DH model, before limit/collision filtering.

    At wrist/base-axis singularities one seed-selected member of the infinite
    family is returned. Returned angles may differ by multiples of 2*pi.
    """
    target=validate_pose(target)
    seed=np.zeros(6) if seed is None else np.asarray(seed,dtype=float)
    dh_rows(seed)
    R=target[:3,:3]
    wrist=target[:3,3]-D6*R[:,2]
    radius=math.hypot(wrist[0],wrist[1])
    heading=math.atan2(wrist[1],wrist[0]) if radius>1e-10 else seed[0]
    length=math.hypot(A3,D4)
    beta=math.atan2(D4,A3)
    candidates=[]
    for base_sign in (1,-1):
        q1=heading+(0 if base_sign==1 else PI)
        radial=base_sign*radius-A1
        vertical=wrist[2]-D1
        cosine=(radial**2+vertical**2-A2**2-length**2)/(2*A2*length)
        if abs(cosine)>1+1e-9:
            continue
        cosine=float(np.clip(cosine,-1,1))
        for elbow_sign in (1,-1):
            phi=elbow_sign*math.acos(cosine)
            theta2=math.atan2(vertical,radial)-math.atan2(length*math.sin(phi),A2+length*math.cos(phi))
            q2,q3=theta2-PI/2,phi+beta
            T03=np.eye(4)
            for row in dh_rows([q1,q2,q3,0,0,0])[:3]:
                T03=T03@dh(*row)
            W=T03[:3,:3].T@R
            sine=math.hypot(W[0,2],W[1,2])
            if sine<1e-9:
                q4=seed[3]
                if W[2,2]>0:
                    q5=0.
                    q6=math.atan2(W[1,0],W[0,0])-q4
                else:
                    q5=PI
                    delta=math.atan2(-W[1,0],-W[0,0])
                    q6=q4-delta
                wrists=[(q4,q5,q6)]
            else:
                q4=math.atan2(W[1,2],W[0,2])
                q5=math.atan2(sine,W[2,2])
                q6=math.atan2(W[2,1],-W[2,0])
                wrists=[(q4,q5,q6),(q4+PI,-q5,q6+PI)]
            for wrist_angles in wrists:
                q=near_seed([q1,q2,q3,*wrist_angles],seed)
                if np.max(abs(fk_dh(q)-target))>1e-7:
                    continue
                if not any(np.max(abs(near_seed(q,old)-old))<1e-7 for old in candidates):
                    candidates.append(q)
    return sorted(candidates,key=lambda q:float(np.linalg.norm(q-seed)))


def refine_ik(model,target,q,max_iterations=40,tolerance=1e-9):
    """Damped Jacobian correction for tiny CAD/export deviations from ideal DH."""
    target=validate_pose(target)
    q=np.array(q,dtype=float,copy=True)
    for _ in range(max_iterations):
        current,J=model.evaluate(q)
        error=np.r_[target[:3,3]-current[:3,3],so3_log(target[:3,:3]@current[:3,:3].T)]
        if np.linalg.norm(error[:3])<tolerance and np.linalg.norm(error[3:])<tolerance:
            return q
        # A 1 m characteristic length keeps translation/orientation residuals
        # equally weighted numerically. This is not a dynamic control law.
        damping=1e-7
        delta=J.T@np.linalg.solve(J@J.T+damping**2*np.eye(6),error)
        norm=np.linalg.norm(delta)
        if norm>.2:
            delta*=.2/norm
        q+=delta
    return None


def ik_exported(target,seed=None,model=None,limits=None):
    """Analytic seeds + exact-URDF refinement; no implicit hardware limits.

    Optional limits must be supplied in the exported q convention. Collision
    filtering is the caller's responsibility. Periodic equivalents inside
    supplied intervals are chosen nearest the seed before rejecting a branch.
    """
    target=validate_pose(target)
    model=model or ExportedModel()
    seed=np.zeros(6) if seed is None else np.asarray(seed,dtype=float)
    candidates=[]
    for guess in ik_analytic(target,seed):
        q=refine_ik(model,target,guess)
        if q is None:
            continue
        q=near_seed(q,seed)
        if limits is not None:
            limits=np.asarray(limits,dtype=float)
            if limits.shape!=(6,2) or not np.isfinite(limits).all() or np.any(limits[:,0]>limits[:,1]):
                raise ValueError('limits must be six finite [lower,upper] pairs')
            valid=True
            for i,(low,high) in enumerate(limits):
                k_low=math.ceil((low-q[i]-1e-10)/(2*PI))
                k_high=math.floor((high-q[i]+1e-10)/(2*PI))
                if k_low>k_high:
                    valid=False
                    break
                k=int(np.clip(round((seed[i]-q[i])/(2*PI)),k_low,k_high))
                q[i]+=2*PI*k
            if not valid:
                continue
        if not any(np.linalg.norm(q-old)<1e-7 for old in candidates):
            candidates.append(q)
    return sorted(candidates,key=lambda q:float(np.linalg.norm(q-seed)))
