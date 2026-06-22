import numpy as np

def vec2skew (w):
    """Calculates the skew symmetric version of angular velocity w
    
    w: 3x1 np ndarray, angular velocity vector
    
    -> 3x3 np ndarray
    """
    return np.array([ [0, -w[2,0], w[1,0]], [w[2,0], 0, -w[0,0]], [-w[1,0], w[0,0], 0] ])

def skew2vec (S):
    """Extracts the vector out of a skew symmetric matrix S
    
    S: 3x3 np ndarray
    
    -> 3x1 np ndarray, angular velocity vector
    """
    return np.array([[S[2,1]], [S[0,2]], [S[1,0]]])

def quat_inv(q):
    """calculates the inverse of a unit quaternion as its conjugate.
    q: 4x1 np ndarray unit norm quaternion
    
    -> 4x1 np ndarray unit norm quaternion
    """
    return np.array([[q[0,0]], [-q[1,0]], [-q[2,0]], [-q[3,0]]])

def quat_mult (q1, q2):
    """Multiply q2 from left by q1
    
    q1: 4x1 np ndarray, first unit norm quaternion
    q2: 4x1 np ndarray, second unit norm quaternion
    
    -> 4x1 np ndarray, unit norm quaternion
    """
    return np.array([
        [q1[0,0]*q2[0,0] - q1[1,0]*q2[1,0] - q1[2,0]*q2[2,0] - q1[3,0]*q2[3,0]], 
        [q1[0,0]*q2[1,0] + q2[0,0]*q1[1,0] + q1[2,0]*q2[3,0] - q1[3,0]*q2[2,0]], 
        [q1[0,0]*q2[2,0] + q2[0,0]*q1[2,0] + q1[3,0]*q2[1,0] - q1[1,0]*q2[3,0]],
        [q1[0,0]*q2[3,0] + q2[0,0]*q1[3,0] + q1[1,0]*q2[2,0] - q1[2,0]*q2[1,0]]
    ])

def left_mat(q):
    """ returns a 4x4 matrix q_L such that q mmult q2 = q_L @ q2 
    """
    return q[0,0]* np.eye(4) + np.block([[0, -q[1:4,0:1].T],[q[1:4,0:1], vec2skew(q[1:4,0:1])]])

def right_mat(q):
    """ returns a 4x4 matrix q_R such that q2 mmult q = q_R @ q2 
    """
    return q[0,0]* np.eye(4) + np.block([[0, -q[1:4,0:1].T],[q[1:4,0:1], -vec2skew(q[1:4,0:1])]])

def quat2angle(q1, q2=np.array([[1.],[0.],[0.],[0.]])):
    """Calculates the angle between 2 unit quaternions if one is the 'zero' quaternion,
    np.array([[1.],[0.],[0.],[0.]], it gives the angle of the first quaternion.
    
    q1: 4x1 np ndarray unit norm quaternion
    q2: 4x1 np ndarray unit norm quaternion
    
    -> scalar, angle between the two quaternions
    """
    return 2 * np.arccos(np.abs(q1.T @ q2))[0,0]

def quat2axis(q):
    """Calculates the axis of rotation of unit quaternion q, 
    if q is the 'zero' quaternion np.array([[1.],[0.],[0.],[0.]], it returns a random axis
    
    q: 4x1 np ndarray unit norm quaternion
    
    -> 3x1 np ndarray unit norm axis of rotation
    """
    axis = np.empty((3,1))
    s_q = np.linalg.norm(q[1:4,:])

    # s_q = sin(theta/2) ~= theta/2 for small theta
    # following test is approximately the same as theta < 10e-9
    if (s_q < 5e-9):
        # return random axis
        s = 0.0
        while(s == 0.0):
            axis = np.random.randn(3,1)
            s = np.linalg.norm(axis)
        axis /= s
    else:
        axis = q[1:4,:] / s_q

    return axis

def Rot2quat(R):
    """Calculates unit quaternion equivalent of rotation matrix R
    R : 3x3 np ndarray orthonormal with det=1 (rotation)
    
    -> 4x1 unit quaternion np ndarray 
    """
    theta = np.arccos(.5 * np.trace(R) - .5)
    
    if (theta < 10e-9):
        q = np.array([[1.0], [0.0], [0.0], [0.0]])
    else:
        psi = np.sin(0.5*theta)*(1.0/np.sin(theta))*0.5
        q = np.array([
            [np.cos(0.5*theta)],
            [psi*(R[2,1]-R[1,2])],
            [psi*(R[0,2]-R[2,0])],
            [psi*(R[1,0]-R[0,1])]
        ])

    return q

def quat2Rot(q):
    """Calculates a rotation matrix equivalent of unit quaternion q (nonunique with sign of first element)
    q : 4x1 unit quaternion np ndarray 
    
    -> 3x3 np ndarray orthonormal with det=1 (rotation)
    """
    w = q[0,0]
    v = q[1:4,:]
    return (w**2 - v.T @ v)* np.eye(3) + 2 * v @ v.T + 2 * w * vec2skew(v)

def ang_vel2quat (w, dt):
    """Calculates the quaternion rotation induced by dt times angular
    velocity w frw integ
    w: 3x1 np ndarray of angular velocity
    dt: scalar time step of forward integration
    
    -> 4x1 np ndarray unit norm quaternion
    """
    # the assumed constant angular velocity
    w = w * dt
    
    # The angle of rotation induced by this velocity
    w_s = np.linalg.norm(w)

    if (w_s < 10e-9):
        q_w = np.array([[1.],[0.],[0.],[0.]])
    else:
        # The axix of rotation induced by this velocity
        w_ax = (1 / w_s) * w
        
        # the quaternion version of the angle and axis of rotation 
        # (note that as always quaternion takes half the angle)
        q_w = np.concatenate([np.array([[np.cos(w_s/2)]]), np.sin(w_s/2) * w_ax])

    return q_w
    
def quat_exp(w, q, dt):
    """Implements the exponential map of quaternions explicitly.
    w: 3x1 np ndarray of angular velocity
    quaternion: 4x1 unit quaternion np ndarray (current quaternion) 
    dt: 1x1 time step of forward integration
    
    -> 4x1 unit quaternion np ndarray (next quaternion) 
    """
    return quat_mult(q, ang_vel2quat(w, dt))
