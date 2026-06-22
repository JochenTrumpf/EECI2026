import numpy as np

from .quat_utils import quat_inv, quat_mult, quat2angle, quat2Rot, skew2vec, vec2skew

class Pose():
    def __init__(self, quat = np.array([[1.],[0.],[0.],[0.]]), pos = np.zeros((3,1))):
        self.quat = quat
        self.pos = pos

    def from_pose(self, pose):
        self.__init__(pose.quat, pose.pos)

def pose_error(q1, p1, q2, p2):
    """calculates the difference between two poses.
    q1: 4x1 np ndarray, unit norm quaternion    
    p1: 3x1 np ndarray, position vector  
    q2: 4x1 np ndarray, unit norm quaternion    
    p2: 3x1 np ndarray, position vector
    
    -> scalar (error(pose1-pose2))
    """
    return 1 - np.cos(quat2angle(q2, q1)) + np.linalg.norm(p2-p1)

def skew_proj(M):
    """Calculates the skew symmetric projection of matrix M.
    M: nxn np ndarray
    
    -> nxn np ndarray
    """
    return .5 * (M - M.T)

def ang_vel2Rot(w, dt):
    """Calculates the rotation matrix induced by dt times angular velocity w
    frw integ
    w: 3x1 np ndarray of angular velocity
    dt: scalar time step of forward integration
    
    -> 3x3 np ndarray Rotation matrix 
    """
    # the assumed constant angular velocity
    w = w * dt
    
    # The angle of rotation induced by this velocity
    w_s = np.linalg.norm(w)

    if (w_s < 10e-9):
        R_w = np.eye(3)
    else:
        # The axix of rotation induced by this velocity
        w_ax = (1 / w_s) * w
        
        # the quaternion version of the angle and axis of rotation (note that as always quaternion takes half the angle)
        R_w = np.eye(3) + np.sin(w_s) * vec2skew(w_ax) + (1-np.cos(w_s)) * vec2skew(w_ax) @ vec2skew(w_ax)

    return R_w

def Rot2angle (R):
    """Calculates angle of rotation of rotation matrix R
    R : 3x3 np array orthonormal with det=1 (rotation)
    
    -> angle of rotation 
    """
    return np.arccos(0.5 * np.trace(R) - 0.5)

def Rot2axis(R):
    """Calculates (unit norm) axis of rotation of rotation matrix R
    R : 3x3 np ndarray orthonormal with det=1 (rotation)
    
    -> 3x1 np ndarray axis of rotation 
    """
    theta = Rot2angle(R)
    
    if (theta < 10e-9):
        # return a random direction
        s = 0.0
        while(s == 0.0):
            axis = np.random.randn(3,1)
            s = np.linalg.norm(axis)
        axis /= s
        axis = np.array([[0.],[0.],[0.]])
    else:  
        axis = (1 / (np.sin(theta))) * skew2vec(skew_proj(R))

    return axis

def ang_vel2R_p_inv (w, dt):
    """Calculates the rotation multiplier of position p for the log map of SE(3)
    w: 3x1 np ndarray of angular velocity
    dt: 1x1 time step of forward integration
    
    -> 3x3 Rotation matrix np ndarray 
    """
    # the assumed constant angular velocity
    w = w * dt
    
    # The angle of rotation induced by this velocity
    w_s = np.linalg.norm (w)
    
    if (w_s < 10e-9):
        R_p_inv = np.eye(3)
    else:
        # The axix of rotation induced by this velocity
        w_ax = (1 / w_s) * w
    
        # the quaternion version of the angle and axis of rotation (note that as always quaternion takes half the angle)
        R_p_inv = np.eye(3) - w_s * vec2skew(w_ax)/2 + (1- w_s * np.sin(w_s)/(2 - 2*np.cos(w_s))) * vec2skew(w_ax) @ vec2skew(w_ax) 

    return R_p_inv

def pose_log(q, p):
    """Implements the logarithm map of SE(3) explicitly.
    q: 4x1 np ndarray, unit norm quaternion    
    p: 3x1 np ndarray, position vector    
    
    -> 3x1, 3x1 np ndarrays of angular velocity and linear velocity
    """
    R = quat2Rot(q)
    theta = Rot2angle(R)
    axis = Rot2axis(R)
    
    ang_vel = theta * axis
    
    R_p_inv = ang_vel2R_p_inv(ang_vel, 1.)
    
    lin_vel = R_p_inv @ p 
    return ang_vel, lin_vel

def waypoint_nav(wayposes, pose, v_o, a_o, g, dt, prox_lim):
        """ turns a waypoint (first of a list) to a control that will achieve it, if already close enough to current waypoint
        it will remove it from the list"""
        # current target
        target = wayposes[0]

        there = True
        done = False
        change = False # changing waypoint
    
        if (pose_error(pose.quat, pose.pos, target.quat, target.pos) > prox_lim):
            there = False
            
        # update waypoint list
        if there:
            if (len(wayposes)>1):
                wayposes = wayposes[1:]
                change = True
            else:
                done = True
        
        # new target
        target = wayposes[0]
        
        a, ang_vel = waypoint2control(target, pose, v_o, a_o, g, dt, prox_lim)

        return done, change, a, ang_vel, wayposes

def waypoint2control(pose_t, pose_o, v_o, a_o, g, dt, prox_lim, max_ang_vel_rate = np.deg2rad(5.), max_lin_vel = 2., max_lin_acc = 80.):
    """" Inverse dynamics from a pair of old and target poses to lin acc and ang vel commands with clamping"""
    q_e = quat_mult(quat_inv(pose_o.quat),pose_t.quat)
    R_e = quat2Rot(q_e)
    angle_e = Rot2angle(R_e)
    axis_e = Rot2axis(R_e)
    angle_e = np.minimum(angle_e, max_ang_vel_rate)
    
    ang_vel = angle_e * axis_e
    
    R = quat2Rot(pose_o.quat)
    
    # one shot control np.cos(angle_e) *
    a_wp = 2*(pose_t.pos - pose_o.pos - v_o*dt)/dt**2

    # clamp for max speed
    v_pred = v_o + dt*a_wp
    s_pred = np.linalg.norm(v_pred)
    if s_pred > max_lin_vel:
        a_wp = (max_lin_vel * v_pred/s_pred - v_o)/dt
    
    # clamp for max acc
    a_wp_s = np.linalg.norm(a_wp)
    if a_wp_s >= 10e-9:
        a_wp = a_wp * np.minimum(a_wp_s, max_lin_acc)/a_wp_s

    # brake if you reach waypoint
    p_pred = pose_o.pos + v_o*dt + .5 *(a_wp)*dt**2
    if np.linalg.norm(p_pred - pose_t.pos) <=prox_lim:
        a_wp = -v_o/dt

    # clamp for max acc again
    a_wp_s = np.linalg.norm(a_wp)
    if a_wp_s >= 10e-9:
        a_wp = a_wp * np.minimum(a_wp_s, max_lin_acc)/a_wp_s 
    
    # compensate for g
    a_wp = a_wp - g

    # convert to body fixed frame
    a_n = R.T @ a_wp

    # convex slide a
    max_j = 20
    alpha = dt*2*max_j /max_lin_acc
    a =  (1-alpha)*a_o +  alpha*a_n 

    return a, ang_vel
