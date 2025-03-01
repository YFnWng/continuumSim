import numpy as np
import time

def hat(u):
    if len(np.shape(u)) == 1:
        return np.array([[0, -u[2], u[1]],
                        [u[2], 0, -u[0]],
                        [-u[1], u[0], 0]])
    else:       
        S = np.zeros((*np.shape(u),3))
        S[...,0,1] = -u[...,2]
        S[...,0,2] = u[...,1]
        S[...,1,2] = -u[...,0]
        S[...,1,0] = u[...,2]
        S[...,2,0] = -u[...,1]
        S[...,2,1] = u[...,0]
        return S

def Hat(u):
    if len(np.shape(u)) == 1:
        return np.array([[0, -u[5], u[4], u[0]],
                        [u[5], 0, -u[3], u[1]],
                        [-u[4], u[3], 0, u[2]],
                        [0,     0,    0,    0]])
    else:       
        S = np.zeros((*np.shape(u)[0:-1],4,4))
        S[...,0,1] = -u[...,5]
        S[...,0,2] = u[...,4]
        S[...,1,2] = -u[...,3]
        S[...,1,0] = u[...,5]
        S[...,2,0] = -u[...,4]
        S[...,2,1] = u[...,3]
        S[...,0,3] = u[...,0]
        S[...,1,3] = -u[...,1]
        S[...,2,3] = u[...,2]
        return S

def ginv(g):
    ginv = np.zeros_like(g)
    ginv[...,0:3,0:3] = np.swapaxes(g[...,0:3,0:3], -1, -2)
    ginv[...,0:3,3] = -ginv[...,0:3,0:3]@g[...,0:3,3]
    return ginv

def expSE3(Psi):
    # exponential map se3 to SE3, refer to Muller's review
    # Psi: N x 6 np array
    N = Psi.shape[0:-1]
    v = Psi[...,0:3] # N x 3
    w = Psi[...,3:6] # N x 3
    # theta = np.sqrt(np.sum(w**2, axis=1))[:,None] # N x 1
    theta = np.linalg.norm(w, ord=2, axis=-1, keepdims=True) # N x 1
    theta2 = theta**2
    nonzeroidx = theta != 0
    # n = np.zeros_like(w)
    # n[nonzeroidx,:] = np.divide(w[nonzeroidx,:], theta[nonzeroidx,:]) # N x 3
    n = np.divide(w, theta, out=np.zeros_like(w), where=nonzeroidx)

    n_hat = hat(n) # N x 3 x 3
    n_hat2 = np.matmul(n_hat, n_hat) # N x 3 x 3
    # alpha = np.zeros_like(theta)
    # alpha[nonzeroidx,:] = np.divide(np.sin(theta[nonzeroidx,:]), theta[nonzeroidx,:]) # N x 1
    alpha = np.divide(np.sin(theta), theta, out=np.zeros_like(theta), where=nonzeroidx)
    # beta = np.zeros_like(theta)
    # beta[nonzeroidx,:] = np.divide(1-np.cos(theta[nonzeroidx,:]), theta[nonzeroidx,:]**2) # N x 1
    beta = np.divide(1-np.cos(theta), theta2, out=np.zeros_like(theta), where=nonzeroidx)
    I = np.tile(np.eye(3),(*N,1,1))
    expw = I + alpha[...,None]*n_hat*theta[...,None] + beta[...,None]*n_hat2*theta2[...,None] # (2.6) N x 3 x 3
    dexpw = I + beta[...,None]*n_hat*theta[...,None] + (1-alpha)[...,None]*n_hat2 # (2.13) N x 3 x 3
    expv = np.matmul(dexpw, v[...,None]) # (2.27) N x 3 x 1
    return np.concatenate((np.concatenate((expw, expv),axis=-1), 
                           np.concatenate((np.zeros((*N,1,3),), np.ones((*N,1,1),)),axis=-1)),axis=-2) # N x 4 x 4

def expTSE3(Psi):
    N = Psi.shape[0:-1]
    # v = Psi[...,0:3] # N x 3
    # w = Psi[...,3:6] # N x 3
    theta = np.linalg.norm(Psi[...,3:6], ord=2, axis=-1, keepdims=True) # N x 1
    nonzeroidx = theta != 0 # theta >= 1e-2
    # zeroidx = theta == 0

    Psihat  = Hat(Psi)
    adjPsi  = ad(Psi)

    Psihatp2 = Psihat@Psihat
    Psihatp3 = Psihatp2@Psihat

    adjPsip2 = adjPsi@adjPsi
    adjPsip3 = adjPsip2@adjPsi
    adjPsip4 = adjPsip3@adjPsi

    limco = np.ones_like(theta)
    I4 = np.tile(np.eye(4),(*N,1,1))
    I6 = np.tile(np.eye(6),(*N,1,1))
        
    tp2        = theta*theta
    tp3        = tp2*theta
    tp4        = tp3*theta
    tp5        = tp4*theta
        
    sintheta   = np.sin(theta)
    costheta   = np.cos(theta)
        
    t1 = theta*sintheta
    t2 = theta*costheta
    
    a2 = np.divide(1-costheta, tp2, out=limco*1/2, where=nonzeroidx)
    a3 = np.divide(theta-sintheta, tp3, out=limco*1/6, where=nonzeroidx)
    g  = I4 + Psihat + a2[...,None]*Psihatp2 + a3[...,None]*Psihatp3

    # b1 = (4-4*costheta-t1)/(2*tp2)
    b1 = np.divide(4-4*costheta-t1, 2*tp2, out=limco*1/2, where=nonzeroidx)
    # b2 = (4*theta-5*sintheta+t2)/(2*tp3)
    b2 = np.divide(4*theta-5*sintheta+t2, 2*tp3, out=limco*1/6, where=nonzeroidx)
    # b3 = (2-2*costheta-t1)/(2*tp4)
    b3 = np.divide(2-2*costheta-t1, 2*tp4, out=limco*1/24, where=nonzeroidx)
    # b4 = (2*theta-3*sintheta+t2)/(2*tp5)
    b4 = np.divide(2*theta-3*sintheta+t2, 2*tp5, out=limco*1/120, where=nonzeroidx)
    Tg = I6 + b1[...,None]*adjPsi + b2[...,None]*adjPsip2 +\
        b3[...,None]*adjPsip3 + b4[...,None]*adjPsip4
        
    return g, Tg

def Ad(g):
    # Adjoint map of SE(3)
    # g: ... x 4 x 4 SE(3) matrix
    R = g[...,0:3,0:3]
    p_hat = hat(g[...,0:3,3])
    return np.concatenate((np.concatenate((R, p_hat@R),axis=-1),
                   np.concatenate((np.zeros_like(R), R),axis=-1)), axis=-2)

def Adinv(g):
    # inverse Adjoint map of SE(3), Adinv() = Ad(ginv())
    # g: ... x 4 x 4 SE(3) matrix
    RT = np.swapaxes(g[...,0:3,0:3], -1,-2)
    p_hat = hat(g[...,0:3,3])
    return np.concatenate((np.concatenate((RT, -RT@p_hat),axis=-1),
                   np.concatenate((np.zeros_like(RT), RT),axis=-1)), axis=-2)

def ad(xi):
    # Adjoint map of se(3)
    # xi: ... x 6 se(3) twist
    v_hat = hat(xi[...,0:3])
    u_hat = hat(xi[...,3:6])
    return np.concatenate((np.concatenate((u_hat, v_hat),axis=-1),
                   np.concatenate((np.zeros_like(u_hat), u_hat),axis=-1)), axis=-2)

def main():
    # print(np.__version__)
    # psi = np.tile(np.array([1.0,0.5,9.0,0.1,0.2,0.3]),(1000,1))
    psi = np.tile(np.array([1.0,0.5,9.0,0.0,0.0,0.0]),(20,1))
    t0 = time.time()
    T = expSE3(psi)
    t1 = time.time()
    Tt, TT = expTSE3(psi)
    t2 = time.time()
    print(t1-t0)
    print(t2-t1)
    # print(T)
    # print(Tt)

if __name__ == "__main__":
    main()