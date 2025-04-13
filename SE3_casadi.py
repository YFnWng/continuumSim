# SE(3) operations using casadi
# Yifan Wang, Apr 2025
# import numpy as np
from casadi import *
import time
    
def hat(u):
    S = SX(3,3)
    S[0,1] = -u[2]
    S[0,2] = u[1]
    S[1,2] = -u[0]
    S[1,0] = u[2]
    S[2,0] = -u[1]
    S[2,1] = u[0]
    return S
    
def Hat(u):
    # N = SX.shape(u)
    # if len(N) == 1:
    #     return SX.array([[0, -u[5], u[4], u[0]],
    #                     [u[5], 0, -u[3], u[1]],
    #                     [-u[4], u[3], 0, u[2]],
    #                     [0,     0,    0,    0]])
    S = SX(4,4)
    S[0,1] = -u[5]
    S[0,2] = u[4]
    S[1,2] = -u[3]
    S[1,0] = u[5]
    S[2,0] = -u[4]
    S[2,1] = u[3]
    S[0,3] = u[0]
    S[1,3] = u[1]
    S[2,3] = u[2]
    return S

def ginv(g):
    ginv = SX(4,4)
    ginv[0:3,0:3] = g[0:3,0:3].T
    ginv[0:3,3] = -ginv[0:3,0:3]@g[0:3,3]
    return ginv

def expTdSE3(Psi, Psid, order=1):
    # N = Psi.shape[0:-1]
    # v = Psi[0:3] # N x 3
    w = Psi[3:6] # N x 3
    wd = Psid[3:6] # N x 3
    theta = norm_2(w) # N x 1
    # nonzeroidx = theta != 0 # theta >= 1e-2
    nonzero = theta >= 1e-2 # theta >= 1e-2

    Psihat  = Hat(Psi)
    adjPsi  = ad(Psi)

    Psihatp2 = Psihat@Psihat
    Psihatp3 = Psihatp2@Psihat

    adjPsip2 = adjPsi@adjPsi
    adjPsip3 = adjPsip2@adjPsi
    adjPsip4 = adjPsip3@adjPsi
        
    tp2        = theta*theta
    tp3        = tp2*theta
    tp4        = tp3*theta
    tp5        = tp4*theta
        
    sintheta   = sin(theta)
    costheta   = cos(theta)
        
    t1 = theta*sintheta
    t2 = theta*costheta
    
    a2 = if_else(nonzero, (1.0-costheta)/tp2, 1/2)
    a3 = if_else(nonzero, (theta-sintheta)/tp3, 1/6)
    g  = SX.eye(4) + Psihat + a2*Psihatp2 + a3*Psihatp3

    b1 = if_else(nonzero, (4.0-4.0*costheta-t1)/(2.0*tp2), 1/2)
    b2 = if_else(nonzero, (4.0*theta-5.0*sintheta+t2)/(2.0*tp3), 1/6)
    b3 = if_else(nonzero, (2.0-2.0*costheta-t1)/(2.0*tp4), 1/24)
    b4 = if_else(nonzero, (2.0*theta-3.0*sintheta+t2)/(2.0*tp5), 1/120)
    dexpPsi = SX.eye(6) + b1*adjPsi + b2*adjPsip2 + b3*adjPsip3 + b4*adjPsip4

    return g, dexpPsi
    
    
    # if order == 1:
    #     return g, dexpPsi
    # else:
    #     lim0 = SX.zeros_like(theta)
    #     # thetad = SX.divide(SX.inner(wd,w), theta, out=SX.linalg.norm(wd, ord=2, axis=-1, keepdims=True), where=nonzeroidx)
    #     thetad = if_else(nonzero, dot(w,wd)/theta, 0.0)

    #     adjPsid  = ad(Psid)
    #     adjPsid2 = adjPsid@adjPsi + adjPsi@adjPsid
    #     adjPsid3 = adjPsid2@adjPsi + adjPsip2@adjPsid
    #     adjPsid4 = adjPsid3@adjPsi + adjPsip3@adjPsid

    #     tp6 = tp5*theta
    #     t3 = (-8+(8-tp2)*costheta+5*t1)*thetad
    #     t4 = (-8*theta+(15-tp2)*sintheta-7*t2)*thetad

    #     c1 = if_else(nonzero, t3/(2*tp3), 0.0)
    #     c2 = if_else(nonzero, t4/(2*tp4), 0.0)
    #     c3 = if_else(nonzero, t3/(2*tp5), 0.0)
    #     c4 = if_else(nonzero, t4/(2*tp6), 0.0)
    #     ddexpPsidt = c1*adjPsi + b1*adjPsid +\
    #         c2*adjPsip2 + b2*adjPsid2 +\
    #         c3*adjPsip3 + b3*adjPsid3 +\
    #         c4*adjPsip4 + b4*adjPsid4
        
    #     return g, dexpPsi, ddexpPsidt

def Ad(g):
    # Adjoint map of SE(3)
    # g: 4 x 4 SE(3) matrix
    R = g[0:3,0:3]
    p_hat = hat(g[0:3,3])
    Adg = SX(6,6)
    Adg[0:3,0:3] = R
    Adg[3:6,3:6] = R
    Adg[0:3,3:6] = p_hat@R
    return Adg

def Adinv(g):
    # inverse Adjoint map of SE(3), Adinv() = Ad(ginv())
    # g: 4 x 4 SE(3) matrix
    RT = g[0:3,0:3].T
    p_hat = hat(g[0:3,3])
    Adinvg = SX(6,6)
    Adinvg[0:3,0:3] = RT
    Adinvg[3:6,3:6] = RT
    Adinvg[0:3,3:6] = -RT@p_hat
    return Adinvg

def ad(xi):
    # Adjoint map of se(3)
    # xi: 6 x 1 se(3) twist
    v_hat = hat(xi[0:3])
    u_hat = hat(xi[3:6])
    ad_xi = SX(6,6)
    ad_xi[0:3,0:3] = u_hat
    ad_xi[0:3,3:6] = v_hat
    ad_xi[3:6,3:6] = u_hat
    return ad_xi

def coad(xi):
    # co-adjoint map of se(3)
    # xi: 6 x 1 se(3) twist
    v_hat = hat(xi[0:3])
    u_hat = hat(xi[3:6])
    coad_xi = SX(6,6)
    coad_xi[0:3,0:3] = u_hat
    coad_xi[3:6,0:3] = v_hat
    coad_xi[3:6,3:6] = u_hat
    return coad_xi

# def hat_for_h(u):
#     u1, u2, u3 = u
#     return SX.array([
#             [0, -u1, -u2, -u3],
#             [u1, 0, u3, -u2],
#             [u2, -u3, 0, u1],
#             [u3, u2, -u1, 0]
#             ])

# def Rh(h):
#     h1, h2, h3, h4 = h
#     h_squared = SX.dot(h, h)
#     I = SX.eye(3)
#     h2_2 = h2**2
#     h3_2 = h3**2
#     h4_2 = h4**2
#     h2h3 = h2*h3
#     h4h1 = h4*h1
#     h2h4 = h2*h4
#     h3h1 = h3*h1
#     h3h4 = h3*h4
#     h2h1 = h2*h1
#     return I + 2 / h_squared * SX.array([
#         [-h3_2 - h4_2, h2h3 - h4h1, h2h4 + h3h1],
#         [h2h3 + h4h1, -h2_2 - h4_2, h3h4 - h2h1],
#         [h2h4 - h3h1, h3h4 + h2h1, -h2_2 - h3_2]
#     ])

def main():
    # print(SX.__version__)

    # psi = SX.tile(SX.array([1.0,0.5,9.0,0.1,0.2,0.3]),(1000,1))
    psi = SX.tile(SX.array([1.0,0.5,9.0,0.0,0.0,0.0]),(36,1))
    # t0 = time.time()
    # T = expSE3(psi)
    # t1 = time.time()
    # Tt, _, _ = expTdSE3(psi,psi)
    # t2 = time.time()
    # print(t1-t0)
    # print(t2-t1)
    # print(T)
    # print(Tt)

    # xi = SX.random.rand(6)
    # t0 = time.time()
    # ad_xi = ad(xi)
    # t1 = time.time()
    # adt_xi = ad_test(xi)
    # t2 = time.time()
    # print(t1-t0)
    # print(t2-t1)
    # print(SX.max(SX.absolute(ad_xi-adt_xi)))

    # xi = SX.random.rand(1000,6)
    # t0 = time.time()
    # h = Hat(xi)
    # t1 = time.time()
    # ht = Hat_test(xi)
    # t2 = time.time()
    # print(t1-t0)
    # print(t2-t1)
    # print(SX.max(SX.absolute(h-ht)))

if __name__ == "__main__":
    main()