import numpy as np
import time
    
def hat(u):
    S = np.zeros((*np.shape(u),3))
    S[...,0,1] = -u[...,2]
    S[...,0,2] = u[...,1]
    S[...,1,2] = -u[...,0]
    S[...,1,0] = u[...,2]
    S[...,2,0] = -u[...,1]
    S[...,2,1] = u[...,0]
    return S
    
def Hat(u):
    N = np.shape(u)
    if len(N) == 1:
        return np.array([[0, -u[5], u[4], u[0]],
                        [u[5], 0, -u[3], u[1]],
                        [-u[4], u[3], 0, u[2]],
                        [0,     0,    0,    0]])
    S = np.zeros((*(N[0:-1]),4,4))
    S[...,0,1] = -u[...,5]
    S[...,0,2] = u[...,4]
    S[...,1,2] = -u[...,3]
    S[...,1,0] = u[...,5]
    S[...,2,0] = -u[...,4]
    S[...,2,1] = u[...,3]
    S[...,0,3] = u[...,0]
    S[...,1,3] = u[...,1]
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

def expTdSE3(Psi, Psid):
    # N = Psi.shape[0:-1]
    # v = Psi[...,0:3] # N x 3
    w = Psi[...,3:6] # N x 3
    wd = Psid[...,3:6] # N x 3
    theta = np.linalg.norm(w, ord=2, axis=-1, keepdims=True) # N x 1
    nonzeroidx = theta != 0 # theta >= 1e-2
    lim0 = np.zeros_like(theta)
    # thetad = np.divide(np.inner(wd,w), theta, out=np.linalg.norm(wd, ord=2, axis=-1, keepdims=True), where=nonzeroidx)
    thetad = np.divide(np.sum(wd*w, axis=-1, keepdims=True), theta, out=lim0, where=nonzeroidx)

    Psihat  = Hat(Psi)
    adjPsi  = ad(Psi)
    adjPsid  = ad(Psid)

    Psihatp2 = Psihat@Psihat
    Psihatp3 = Psihatp2@Psihat

    adjPsip2 = adjPsi@adjPsi
    adjPsip3 = adjPsip2@adjPsi
    adjPsip4 = adjPsip3@adjPsi

    adjPsid2 = adjPsid@adjPsi + adjPsi@adjPsid
    adjPsid3 = adjPsid2@adjPsi + adjPsip2@adjPsid
    adjPsid4 = adjPsid3@adjPsi + adjPsip3@adjPsid

    limco = np.ones_like(theta)
    # I4 = np.tile(np.eye(4),(*N,1,1))
    # I6 = np.tile(np.eye(6),(*N,1,1))
        
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
    g  = np.eye(4)[None,...] + Psihat + a2[...,None]*Psihatp2 + a3[...,None]*Psihatp3

    b1 = np.divide(4-4*costheta-t1, 2*tp2, out=limco*1/2, where=nonzeroidx)
    b2 = np.divide(4*theta-5*sintheta+t2, 2*tp3, out=limco*1/6, where=nonzeroidx)
    b3 = np.divide(2-2*costheta-t1, 2*tp4, out=limco*1/24, where=nonzeroidx)
    b4 = np.divide(2*theta-3*sintheta+t2, 2*tp5, out=limco*1/120, where=nonzeroidx)
    dexpPsi = np.eye(6)[None,...] + b1[...,None]*adjPsi + b2[...,None]*adjPsip2 +\
        b3[...,None]*adjPsip3 + b4[...,None]*adjPsip4
    
    tp6 = tp5*theta
    t3 = (-8+(8-tp2)*costheta+5*t1)*thetad
    t4 = (-8*theta+(15-tp2)*sintheta-7*t2)*thetad
    
    c1 = np.divide(t3, 2*tp3, out=lim0, where=nonzeroidx)
    c2 = np.divide(t4, 2*tp4, out=lim0, where=nonzeroidx)
    c3 = np.divide(t3, 2*tp5, out=lim0, where=nonzeroidx)
    c4 = np.divide(t4, 2*tp6, out=lim0, where=nonzeroidx)
    ddexpPsidt = c1[...,None]*adjPsi + b1[...,None]*adjPsid +\
          c2[...,None]*adjPsip2 + b2[...,None]*adjPsid2 +\
          c3[...,None]*adjPsip3 + b3[...,None]*adjPsid3 +\
          c4[...,None]*adjPsip4 + b4[...,None]*adjPsid4
        
    return g, dexpPsi, ddexpPsidt

def Ad(g):
    # Adjoint map of SE(3)
    # g: ... x 4 x 4 SE(3) matrix
    R = g[...,0:3,0:3]
    p_hat = hat(g[...,0:3,3])
    Adg = np.zeros((*(g.shape[:-2]),6,6))
    Adg[...,0:3,0:3] = R
    Adg[...,3:6,3:6] = R
    Adg[...,0:3,3:6] = p_hat@R
    return Adg

def Adinv(g):
    # inverse Adjoint map of SE(3), Adinv() = Ad(ginv())
    # g: ... x 4 x 4 SE(3) matrix
    RT = np.swapaxes(g[...,0:3,0:3], -1,-2)
    p_hat = hat(g[...,0:3,3])
    Adinvg = np.zeros((*(g.shape[:-2]),6,6))
    Adinvg[...,0:3,0:3] = RT
    Adinvg[...,3:6,3:6] = RT
    Adinvg[...,0:3,3:6] = -RT@p_hat
    return Adinvg

def ad(xi):
    # Adjoint map of se(3)
    # xi: ... x 6 se(3) twist
    v_hat = hat(xi[...,0:3])
    u_hat = hat(xi[...,3:6])
    ad_xi = np.zeros((*xi.shape,6))
    ad_xi[...,0:3,0:3] = u_hat
    ad_xi[...,0:3,3:6] = v_hat
    ad_xi[...,3:6,3:6] = u_hat
    return ad_xi

def coad(xi):
    # co-adjoint map of se(3)
    # xi: ... x 6 se(3) twist
    v_hat = hat(xi[...,0:3])
    u_hat = hat(xi[...,3:6])
    coad_xi = np.zeros((*xi.shape,6))
    coad_xi[...,0:3,0:3] = u_hat
    coad_xi[...,3:6,0:3] = v_hat
    coad_xi[...,3:6,3:6] = u_hat
    return coad_xi

def hat_for_h(u):
    u1, u2, u3 = u
    return np.array([
            [0, -u1, -u2, -u3],
            [u1, 0, u3, -u2],
            [u2, -u3, 0, u1],
            [u3, u2, -u1, 0]
            ])

def Rh(h):
    h1, h2, h3, h4 = h
    h_squared = np.dot(h, h)
    I = np.eye(3)
    h2_2 = h2**2
    h3_2 = h3**2
    h4_2 = h4**2
    h2h3 = h2*h3
    h4h1 = h4*h1
    h2h4 = h2*h4
    h3h1 = h3*h1
    h3h4 = h3*h4
    h2h1 = h2*h1
    return I + 2 / h_squared * np.array([
        [-h3_2 - h4_2, h2h3 - h4h1, h2h4 + h3h1],
        [h2h3 + h4h1, -h2_2 - h4_2, h3h4 - h2h1],
        [h2h4 - h3h1, h3h4 + h2h1, -h2_2 - h3_2]
    ])

def main():
    # print(np.__version__)

    # psi = np.tile(np.array([1.0,0.5,9.0,0.1,0.2,0.3]),(1000,1))
    psi = np.tile(np.array([1.0,0.5,9.0,0.0,0.0,0.0]),(36,1))
    t0 = time.time()
    T = expSE3(psi)
    t1 = time.time()
    Tt, _, _ = expTdSE3(psi,psi)
    t2 = time.time()
    print(t1-t0)
    print(t2-t1)
    # print(T)
    # print(Tt)

    # xi = np.random.rand(6)
    # t0 = time.time()
    # ad_xi = ad(xi)
    # t1 = time.time()
    # adt_xi = ad_test(xi)
    # t2 = time.time()
    # print(t1-t0)
    # print(t2-t1)
    # print(np.max(np.absolute(ad_xi-adt_xi)))

    # xi = np.random.rand(1000,6)
    # t0 = time.time()
    # h = Hat(xi)
    # t1 = time.time()
    # ht = Hat_test(xi)
    # t2 = time.time()
    # print(t1-t0)
    # print(t2-t1)
    # print(np.max(np.absolute(h-ht)))

if __name__ == "__main__":
    main()