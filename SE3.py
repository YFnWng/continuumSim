import numpy as np

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
        return np.array([[0, -u[2], u[1], u[3]],
                        [u[2], 0, -u[0], u[4]],
                        [-u[1], u[0], 0, u[5]],
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

def expSE3_test(Psi):
    N = Psi.shape[0:-1]
    v = Psi[...,0:3] # N x 3
    w = Psi[...,3:6] # N x 3
    theta = np.linalg.norm(w, ord=2, axis=-1, keepdims=True) # N x 1

    Psihat  = Hat(Psi)
    # adjPsi  = ad(Psi)

    Psihatp2 = Psihat@Psihat
    Psihatp3 = Psihatp2@Psihat

    # adjPsip2 = adjPsi@adjPsi
    # adjPsip3 = adjPsip2@adjPsi
    # adjPsip4 = adjPsip3@adjPsi

    I = np.tile(np.eye(4),(*N,1,1))

    if (theta<=1e-2):
        g  = I+Psihat+Psihatp2/2+Psihatp3/6
        
        # f1 = 1/2
        # f2 = 1/6
        # f3 = 1/24
        # f4 = 1/120
        
        # Tg  = np.eye(6)+f1*adjPsi+f2*adjPsip2+f3*adjPsip3+f4*adjPsip4
    else:
        
        tp2        = theta*theta
        tp3        = tp2*theta
        # tp4        = tp3*theta
        # tp5        = tp4*theta
        
        sintheta   = np.sin(theta)
        costheta   = np.cos(theta)
        
        # t1 = theta*sintheta
        # t2 = theta*costheta
        
        g   = I+Psihat+\
            (1-costheta)/(tp2)*Psihatp2+\
            ((theta-sintheta)/(tp3))*Psihatp3
        # Tg  = np.eye(6)+((4-4*costheta-t1)/(2*tp2))*adjPsi+\
        #     ((4*theta-5*sintheta+t2)/(2*tp3))*adjPsip2+\
        #     ((2-2*costheta-t1)/(2*tp4))*adjPsip3+\
        #     ((2*theta-3*sintheta+t2)/(2*tp5))*adjPsip4
        
    return g#, Tg

def Ad(g):
    # Adjoint map of SE(3)
    # g: ... x 4 x 4 SE(3) matrix
    R = g[...,0:3,0:3]
    p_hat = hat(g[...,0:3,3])
    return np.concatenate((np.concatenate((R, p_hat@R),axis=-1),
                   np.concatenate((np.zeros_like(R), R),axis=-1)), axis=-2)

def ad(xi):
    # Adjoint map of se(3)
    # xi: ... x 6 se(3) twist
    v_hat = hat(xi[...,0:3])
    u_hat = hat(xi[...,3:6])
    return np.concatenate((np.concatenate((u_hat, v_hat),axis=-1),
                   np.concatenate((np.zeros_like(u_hat), u_hat),axis=-1)), axis=-2)

def main():
    # print(np.__version__)
    psi = np.array([1.0,0.5,9.0,0.1,0.2,0.3])
    T = expSE3(psi)
    Tt = expSE3_test(psi)
    print(T)
    print(Tt)

if __name__ == "__main__":
    main()