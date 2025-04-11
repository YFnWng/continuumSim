import numpy as np
# from casadi import *
from scipy.interpolate import BSpline
# from scipy.spatial.transform import Rotation
# from scipy.linalg import solve
from scipy.optimize import root
# from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from SE3 import *
from contact import *

class continuumRobot_GVS:
    def __init__(self, MP, deg=4, p=3, nb=12, batch=1, model="Cosserat"):
        # space integration parameters
        # self.N = 41
        self.L = MP["L"]
        # self.s_step = np.linspace(0,self.L,self.N)
        # self.ds = self.s_step[1] - self.s_step[0]
        xq, wq = np.polynomial.legendre.leggauss(deg) # Legendre zeros and quadrature weights
        self.xq = (xq + 1)/2
        
        xz, _ = np.polynomial.legendre.leggauss(2) # 4th-order Zanna quadrature (4.17)
        self.xz = (xz + 1)/2

        self.nb = nb # number of basis/control points
        self.sk = np.linspace(0, 1, p+nb+1-2*p) # joints of polynomial pieces, uniformly distributed
        knots = np.concatenate((np.zeros(p), self.sk, np.ones(p))) # knot points for B-spline
        self.wq = np.tile(wq,len(self.sk)-1)/(2*(len(self.sk)-1)) # all weights should sum to 1 !!!!
        # self.sk = np.unique(knots) # joints of polynomial pieces
        self.hk = self.sk[1:] - self.sk[:-1] # quadrature intervals
        self.sq = (self.sk[:-1,None] + self.hk[:,None]*self.xq[None,:]).flatten() # quadrature points between knots
        # self.sg = np.sort(np.concatenate([self.sk,self.sq])) # locations of all sig points
        self.sg = np.concatenate((np.array([0]),self.sq,np.array([1]))) # locations of all sig points
        self.hg = self.sg[1:] - self.sg[:-1] # Magnus expansion intervals
        self.sz1 = self.sg[:-1] + self.hg*self.xz[0] # Zanna quadrature points between sig points
        self.sz2 = self.sg[:-1] + self.hg*self.xz[1] # Zanna quadrature points between sig points
        self.p = p # degree of B-spline
        # self.nb = len(knots) - p - 1 # number of B-spline basis
        self.ng = len(self.sg) # if sig points are only quadrature points, then need to add s=0 and s=L
        self.N = 0
        
        # B_Cosserat = np.eye(6)
        # B_Kirchhoff = np.vstack((np.eye(3),np.zeros(3)))
        if model == "Cosserat":
            self.B = np.eye(6)
        elif model == "Kirchhoff":
            self.B = np.vstack((np.zeros((3,3)),np.eye(3)))
        na = np.shape(self.B)[1]
        self.dof = na*self.nb

        # collocation basis
        nz = self.ng-1
        Bg = np.zeros((self.ng,self.nb))
        Bk = np.zeros((len(self.sk),self.nb))
        Bq = np.zeros((len(self.sq),self.nb))
        Bz1 = np.zeros((nz,self.nb))
        Bz2 = np.zeros((nz,self.nb))
        # self.D = np.zeros_like(self.S,device=device)
        for i in range(self.nb):
            b = BSpline.basis_element(knots[i:i+p+2])
            active_sg = (self.sg>=knots[i]) & (self.sg<=knots[i+p+1])
            Bg[active_sg,i] = b(self.sg[active_sg])
            active_sk = (self.sk>=knots[i]) & (self.sk<=knots[i+p+1])
            Bk[active_sk,i] = b(self.sk[active_sk])
            active_sq = (self.sq>=knots[i]) & (self.sq<=knots[i+p+1])
            Bq[active_sq,i] = b(self.sq[active_sq])
            active_sz1 = (self.sz1>=knots[i]) & (self.sz1<=knots[i+p+1])
            Bz1[active_sz1,i] = b(self.sz1[active_sz1])
            active_sz2 = (self.sz2>=knots[i]) & (self.sz2<=knots[i+p+1])
            Bz2[active_sz2,i] = b(self.sz2[active_sz2])
        Bk[-1,-1] = 1
        Bg[-1,-1] = 1
        self.Bg = np.zeros((self.ng,na,self.dof))
        self.Bk = np.zeros((len(self.sk),na,self.dof))
        self.Bq = np.zeros((len(self.sq),na,self.dof))
        self.Bz1 = np.zeros((nz,na,self.dof))
        self.Bz2 = np.zeros((nz,na,self.dof))
        for i in range(na):
            self.Bg[:,i,i*self.nb:(i+1)*self.nb] = Bg
            self.Bk[:,i,i*self.nb:(i+1)*self.nb] = Bk
            self.Bq[:,i,i*self.nb:(i+1)*self.nb] = Bq
            self.Bz1[:,i,i*self.nb:(i+1)*self.nb] = Bz1
            self.Bz2[:,i,i*self.nb:(i+1)*self.nb] = Bz2
        self.Bg = self.B[None,...]@self.Bg
        self.Bk = self.B[None,...]@self.Bk
        self.Bq = self.B[None,...]@self.Bq
        self.Bz1 = self.B[None,...]@self.Bz1
        self.Bz2 = self.B[None,...]@self.Bz2
        self.BqT = np.swapaxes(self.Bq,-1,-2)
        self.BgT = np.swapaxes(self.Bg,-1,-2)

        self.hg = self.hg*self.L
        self.sg = self.sg*self.L
        self.hk = self.hk*self.L
        self.sk = self.sk*self.L
        self.sq = self.sq*self.L

        # time integration parameters
        # self.BDF(0,5e-3)

        # mechanical properties
        self.r = MP["r"]
        self.E = MP["E"]
        self.nu = MP["nu"]
        self.rho = MP["rho"]
        
        A   = np.pi*MP["r"]**2     
        G   = MP["E"]/(2*(1+MP["nu"]))   
        I1 = np.pi*MP["r"]**4/4
        I2 = I1
        I3  = 2*I1

        EI  = MP["E"]*I1
        GI3 = G*I3
        GA  = G*A
        EA  = MP["E"]*A

        self.I   = np.diag([I1,I2,I3])
        self.Ks = np.diag([GA, GA, EA, EI, EI, GI3])
        self.Kbt = np.diag([EI, EI, GI3])
        self.Kse = np.diag([GA, GA, EA])
        self.Dbt = MP["Dbt"]
        self.Dse = MP["Dse"]
        self.Ds = np.concatenate((np.concatenate((self.Dse,np.zeros_like(self.Dse)),axis=-1),
                                 np.concatenate((np.zeros_like(self.Dbt),self.Dbt),axis=-1)), axis=-2)
        self.invKbt = np.linalg.inv(self.Kbt)
        self.invKse = np.linalg.inv(self.Kse)
        self.usr = MP["usr"]
        self.vsr = MP["vsr"]
        self.xisr = np.concatenate([self.vsr,self.usr])
        self.Kbtusr = self.Kbt@MP["usr"]
        self.Ksevsr = self.Kse@MP["vsr"]

        # g = np.array([0,0,-9.8])
        g = np.array([9.8,0,0])
        self.rho = MP["rho"]
        self.rhoAg = self.rho*A*g
        self.rhoA  = self.rho*A
        self.Ms = np.zeros((6,6))
        self.Ms[0:3,0:3] = self.rhoA*np.eye(3)
        self.Ms[3:6,3:6] = self.rho*self.I

        self.rt = MP["rt"]
        self.rt_hat = hat(self.rt)
        self.rt = self.rt.T
        self.udof = int(np.shape(self.rt)[-1]/2)

        # Generalized stiffness matrix
        self.K = np.sum(self.wq[:,None,None]*self.BqT@self.Ks[None,...]@self.Bq, axis=0)
        # Generalized damping matrix
        self.D = np.sum(self.wq[:,None,None]*self.BqT@self.Ds[None,...]@self.Bq, axis=0)
        
        # controllables
        self.tau = np.zeros_like(self.rt[0])
        self.R0 = np.eye(3)
        self.p0 = np.zeros(3)

        # external loads
        self.FL = np.zeros(3)
        # self.FL = np.array([1,0,0])
        self.ML = np.zeros(3)
        # self.ML = np.array([0,0.3,0])
        self.WL = np.concatenate((self.FL,self.ML))

        # cache
        # if batch == 1:
        #     self.qqq = np.zeros((3,self.dof))
        #     # self.qdot = np.zeros_like(self.q)
        #     # self.qddot = np.zeros_like(self.q)
        #     self.g = np.tile(np.eye(4),(self.ng,1,1))
        #     self.J = np.zeros((self.ng,6,self.dof))
        #     self.Jdot = np.zeros_like(self.J)
        #     self.eta = np.zeros((self.ng,6))
        #     self.M = np.zeros((self.dof,self.dof))
        #     self.C = np.zeros((self.dof,self.dof))
        #     self.A = np.zeros((self.dof,self.rt.shape[1]))
        #     self.f = np.zeros(self.dof)
        # else:
        self.qqq = np.zeros((batch,3,self.dof))
            # self.qdot = np.zeros_like(self.q)
            # self.qddot = np.zeros_like(self.q)
        self.g = np.tile(np.eye(4),(batch,self.ng,1,1))
        self.J = np.zeros((batch,self.ng,6,self.dof))
        self.Jdot = np.zeros_like(self.J)
        self.eta = np.zeros((batch,self.ng,6))
        self.M = np.zeros((batch,self.dof,self.dof))
        self.C = np.zeros((batch,self.dof,self.dof))
        self.A = np.zeros((batch,self.dof,self.rt.shape[1]))
        self.f = np.zeros((batch,self.dof))
        self.batch = batch

        # debugging
        # self.N = 41
        # self.xi = np.zeros((self.N,6))
        # self.xi[:,2] = np.ones(self.N)
        # self.xidot = np.zeros_like(self.xi)
        # self.xiddot = np.zeros_like(self.xidot)
        # self.s_step = np.linspace(0,self.L,self.N)
        # self.ds = self.s_step[1] - self.s_step[0]
        # self.Y = np.zeros((self.N,25))

    def set_state(self,q,qdot):
        self.qqq[...,0,:] = q
        self.qqq[...,1,:] = qdot
    
    def BDF_init(self,alpha,dt):
        # https://en.wikipedia.org/wiki/Backward_differentiation_formula
        self.dt = dt
        self.c0 = (1.5 + alpha)/(self.dt*(1 + alpha))
        self.c1 = -2/self.dt
        self.c2 = (0.5 + alpha)/(self.dt*(1 + alpha))
        self.d1 = alpha/(1 + alpha)

    def Newmark_init(self, dt=5e-3, beta=1/4, gamma=1/2):
        self.dt = dt
        self.Newmark = np.array([[-gamma/(beta*dt),(1-gamma/beta),(1-gamma/(2*beta))*dt],
                                 [-1/(beta*dt**2),-1/(beta*dt),1-1/(2*beta)]])
        self.hqdot = self.Newmark[0]@self.qqq
        self.hqddot = self.Newmark[1]@self.qqq
        # fn = self.q + dt*self.qdot + dt**2*(1/2-beta)*self.qddot
        # hn = self.qdot + dt*(1-gamma)*self.qddot
        # self.q = a*self.qddot + fn
        # self.qdot = b*self.qddot + hn
    
    def forward_kinematics(self, q, qdot):
        hg = self.hg[:,None]

        xi_z1 = np.squeeze(self.Bz1@q[...,None,:,None],axis=-1) + self.xisr # body twist at all Zanna quadrature points
        xi_z2 = np.squeeze(self.Bz2@q[...,None,:,None],axis=-1) + self.xisr # q shape (batch, dof), result shape (batch, nz, 6)
        ad_xi_z1 = ad(xi_z1) # (batch, nz, 6, 6)

        Z2 = np.sqrt(3)*hg**2/12
        Omega = hg/2*(xi_z1+xi_z2) + \
            Z2*np.squeeze((ad_xi_z1@xi_z2[...,None]),axis=-1) # 4th-order Zanna collocation (4.19)
        DOmegaDq  = hg[...,None]/2*(self.Bz1+self.Bz2) + \
                    Z2[...,None]*(ad_xi_z1@self.Bz2-ad(xi_z2)@self.Bz1) # (batch, nz, 6, dof)

        # D2OmegaDq2 = 2*Z2[...,None]*(ad(np.squeeze(self.Bz1@qdot[...,None,:,None],axis=-1))@self.Bz2)

        expOmega, dexpOmega = expTdSE3(Omega, np.squeeze(DOmegaDq@qdot[...,None,:,None],axis=-1), order=1) #, ddexpOmegadt

        AdinvexpOmega = Adinv(expOmega)
        J_rel = dexpOmega@DOmegaDq
        for i in range(self.ng-1):
            self.g[...,i+1,:,:] = self.g[...,i,:,:]@expOmega[...,i,:,:]
            self.J[...,i+1,:,:] = AdinvexpOmega[...,i,:,:]@(self.J[...,i,:,:] + J_rel[...,i,:,:])
        
        self.eta = np.squeeze(self.J@qdot[...,None,:,None],axis=-1)
        # Jdot_rel = ad(self.eta[...,:-1,:])@J_rel + ddexpOmegadt@DOmegaDq + dexpOmega@D2OmegaDq2
        # for i in range(self.ng-1):
        #     self.Jdot[...,i+1,:,:] = AdinvexpOmega[...,i,:,:]@(self.Jdot[...,i,:,:] + Jdot_rel[...,i,:,:])


    def Cosserat_dynamic_ODE(self, t, y):
        # Mqdd + (C+D)qd + Kq = Au + f
        q = y[0:self.dof]
        qdot = y[self.dof:]
        tau = self.tau(t)
        # tau = np.zeros(4)

        # forward kinematics
        self.forward_kinematics(q,qdot)
        
        # cache
        wq = self.L*self.wq[:,None,None]
        Bq = self.Bq
        Ms = self.Ms[None,...]
        # q_idx = np.mod(np.arange(self.ng), 4) != 0
        J = self.J[1:-1,...]
        JT = np.swapaxes(J,-1,-2)
        JLT = self.J[-1,...].T
        xiq = Bq@q + self.xisr
        rt = self.rt[None,...] # None x 3 x nt

        # Generalized mass matrix
        self.M = np.sum(wq*JT@Ms@J, axis=0)

        # Generalized centrifugal & Coriolis matrix
        self.C = np.sum(wq*JT@(Ms@self.Jdot[1:-1,...]+coad(self.eta[1:-1,:])@Ms@J), axis=0)

        # Generalized actuation
        pbs = np.cross(xiq[:,3:6,None],rt, axis=1) + xiq[:,0:3,None] # nq x 3 x nt
        pbs_norm = np.linalg.norm(pbs, ord=2, axis=1, keepdims=True) # nq x 1 x nt
        pbs_n = pbs / pbs_norm # nq x 3 x nt
        Act = np.concatenate((pbs_n, np.cross(rt,pbs_n, axis=1)), axis=1) # nq x 6 x nt

        # boundary condition of actuation
        # xiL = self.Bk[-1,...]@q + self.xisr
        # pbsL = np.cross(xiL[3:6,None],rt, axis=0) + xiL[0:3,None] # 3 x nt
        # pbsL_norm = np.linalg.norm(pbsL, ord=2, axis=0, keepdims=True) # 1 x nt
        # pbsL_n = pbsL / pbsL_norm # 3 x nt
        # ActL = np.concatenate((pbsL_n, np.cross(rt,pbsL_n, axis=0)), axis=0) # 6 x nt

        self.A = -np.sum(wq*self.BqT@Act, axis=0)# + JLT@ActL # dof x nt

        # Generalized external load
        fc = np.zeros(3) #fc[j] 
        f  = self.rhoAg + fc
        F = np.concatenate((f,np.zeros_like(f)),axis=0)
        self.f = np.sum(wq*JT@F, axis=0) + JLT@self.WL

        # self.qddot = solve(self.M, self.A@self.tau + self.f - (self.C+self.D)@self.qdot - self.K@self.q, assume_a="positive definite")
        qddot = np.linalg.solve(self.M, self.A@tau + self.f - (self.C+self.L*self.D)@qdot - self.L*self.K@q)

        return np.concatenate([qdot,qddot])
    
    def Cosserat_dynamic_residual(self, q, qdot, qddot, u):
        # Mqdd + (C+D)qd + Kq = Au + f

        # forward kinematics
        self.forward_kinematics(q,qdot)
        
        # cache
        wq = self.L*self.wq[:,None,None]
        Bq = self.Bq
        Ms = self.Ms#[None,...]
        J = self.J[...,1:-1,:,:]
        JT = np.swapaxes(J,-1,-2)
        JLT = np.swapaxes(self.J[...,-1,:,:],-1,-2)
        xiq = np.squeeze(Bq@q[...,None,:,None],axis=-1) + self.xisr # batch x nz x 6
        rt = self.rt[None,...] # None x 3 x nt

        # Generalized mass matrix
        self.M = np.sum(wq*JT@Ms@J, axis=-3) # batch x dof x dof

        # Generalized centrifugal & Coriolis matrix
        # self.C = np.sum(wq*JT@(Ms@self.Jdot[...,1:-1,:,:]+coad(self.eta[...,1:-1,:])@Ms@J), axis=-3)
        self.C = np.sum(wq*JT@(coad(self.eta[...,1:-1,:])@Ms@J), axis=-3)

        # Generalized actuation
        pbs = np.cross(xiq[...,3:6,None],rt, axis=-2) + xiq[...,0:3,None] # batch x nq x 3 x nt
        pbs_norm = np.linalg.norm(pbs, ord=2, axis=-2, keepdims=True) # batch x nq x 1 x nt
        pbs_n = pbs / pbs_norm # batch x nq x 3 x nt
        Act = np.concatenate((pbs_n, np.cross(rt,pbs_n, axis=-2)), axis=-2) # batch x nq x 6 x nt

        self.A = -np.sum(wq*self.BqT@Act, axis=-3) # batch x dof x nt

        # Generalized external load
        FL,fc = contactForce(self.g[...,1:,:,:],self.eta[...,1:,0:3],self.hg[-1])
        fc  = self.rhoAg@self.g[...,1:-1,0:3,0:3] + fc # batch x nq x 3
        FL = FL + self.FL@self.g[...,-1,0:3,0:3]
        w = np.concatenate((fc,np.zeros_like(fc)),axis=-1)
        self.f = np.sum(wq*JT@w[...,None], axis=-3) + \
                    JLT@(np.concatenate((FL,np.zeros_like(FL)),axis=-1)[...,None])
                    
        tau = np.concatenate((u,-u), axis=-1)
        return (self.M@qddot[...,None] + (self.C+self.L*self.D)@qdot[...,None] + 
                          self.L*self.K@q[...,None] - self.A@tau[...,None] - self.f).flatten()
    
    # def Cosserat_dynamic_residual(self, q, qdot, qddot, u):
    #     # Mqdd + (C+D)qd + Kq = Au + f

    #     # tau = self.tau(t)
    #     # tau = np.zeros(4)

    #     # forward kinematics
    #     self.forward_kinematics(q,qdot)
        
    #     # cache
    #     wq = self.L*self.wq[:,None,None]
    #     Bq = self.Bq
    #     Ms = self.Ms#[None,...]
    #     # q_idx = np.mod(np.arange(self.ng), 4) != 0
    #     J = self.J[1:-1,...]
    #     JT = np.swapaxes(J,-1,-2)
    #     JLT = self.J[-1,...].T
    #     xiq = Bq@q + self.xisr
    #     rt = self.rt[None,...] # None x 3 x nt

    #     # Generalized mass matrix
    #     self.M = np.sum(wq*JT@Ms@J, axis=0)

    #     # Generalized centrifugal & Coriolis matrix
    #     # self.C = np.sum(wq*JT@(Ms@self.Jdot[1:-1,:,:]+coad(self.eta[1:-1,:])@Ms@J), axis=0)
    #     self.C = np.sum(wq*JT@(coad(self.eta[1:-1,:])@Ms@J), axis=0)

    #     # Generalized actuation
    #     pbs = np.cross(xiq[:,3:6,None],rt, axis=1) + xiq[:,0:3,None] # nq x 3 x nt
    #     pbs_norm = np.linalg.norm(pbs, ord=2, axis=1, keepdims=True) # nq x 1 x nt
    #     pbs_n = pbs / pbs_norm # nq x 3 x nt
    #     Act = np.concatenate((pbs_n, np.cross(rt,pbs_n, axis=1)), axis=1) # nq x 6 x nt

    #     self.A = -np.sum(wq*self.BqT@Act, axis=0) # dof x nt

    #     # Generalized external load
    #     FL,fc = contactForce(self.g[1:,...],self.eta[1:,0:3],self.hg[-1])
    #     fc  = self.rhoAg + fc
    #     w = np.concatenate((fc,np.zeros_like(fc)),axis=-1)
    #     self.f = np.squeeze(np.sum(wq*JT@w[...,None], axis=0), axis=-1) + JLT@np.concatenate((FL,np.zeros(3)))

    #     return self.M@qddot + (self.C+self.L*self.D)@qdot + self.L*self.K@q - self.A@u - self.f
    
    def Newmark_residual(self, q, u):
        q = np.reshape(q,(-1,self.dof))
        qdot = -self.Newmark[0,0]*q + self.hqdot
        qddot = -self.Newmark[1,0]*q + self.hqddot
        return self.Cosserat_dynamic_residual(q, qdot, qddot, u)

    def Cosserat_static_residual(self, q, u):
        # Kq = Au + f

        # forward kinematics
        self.forward_kinematics(q, np.zeros_like(q)) # less calc
        
        # cache
        wq = self.L*self.wq[:,None,None]
        Bq = self.Bq
        # q_idx = np.mod(np.arange(self.ng), 4) != 0
        # J = self.J[1:-1,...]
        JT = np.swapaxes(self.J[...,1:-1,:,:],-1,-2)
        JLT = np.swapaxes(self.J[...,-1,:,:],-1,-2)
        xiq = np.squeeze(Bq@q[...,None,:,None],axis=-1) + self.xisr # batch x nz x 6
        rt = self.rt[None,...] # None x 3 x nt

        # Generalized actuation
        pbs = np.cross(xiq[...,3:6,None],rt, axis=-2) + xiq[...,0:3,None] # nq x 3 x nt
        pbs_norm = np.linalg.norm(pbs, ord=2, axis=-2, keepdims=True) # nq x 1 x nt
        pbs_n = pbs / pbs_norm # nq x 3 x nt
        Act = np.concatenate((pbs_n, np.cross(rt,pbs_n, axis=-2)), axis=-2) # nq x 6 x nt
        self.A = -np.sum(wq*self.BqT@Act, axis=-3)# + JLT@ActL # dof x nt

        # Generalized external load
        FL,fc = contactForce(self.g[...,1:,:,:],np.zeros((self.batch,self.ng-1,3)),self.hg[-1])
        fc  = self.rhoAg@self.g[...,1:-1,0:3,0:3] + fc
        w = np.concatenate((fc,np.zeros_like(fc)),axis=-1)
        FL = FL + self.FL@self.g[...,-1,0:3,0:3]
        # F = np.concatenate((np.zeros((JT.shape[0],4)),0.3*np.ones((JT.shape[0],1)),np.zeros((JT.shape[0],1))),axis=1)
        # WL = np.concatenate((self.g[-1,0:3,0:3].T@self.FL,self.g[-1,0:3,0:3].T@self .ML))
        self.f = np.sum(wq*JT@w[...,None], axis=-3) + \
                    JLT@(np.concatenate((FL,np.zeros_like(FL)),axis=-1)[...,None])

        tau = np.concatenate((u,-u), axis=-1)

        return (self.L*self.K@q[...,None] - (self.A@tau[...,None] + self.f)).flatten()
    
    def static_solve(self, u, ig):
        sol = root(lambda q: self.Cosserat_static_residual(q,u), x0=ig)
        print("success?",sol.success)
        print(sol.status)
        print(sol.message)
        print("nfev: ",sol.nfev)
        # print("njev: ",sol.njev)
        return sol.x
    
    def roll_out(self, u, t_span, method="Newmark"):
        # y0 = np.concatenate((self.q,self.qdot))
        # rollout = solve_ivp(self.Cosserat_dynamic_ODE, t_span, y0, method="Radau", t_eval=np.linspace(t_span[0],t_span[1],101))
        # print(rollout.message)
        # print("nfev = ",rollout.nfev)
        # print("njev = ",rollout.njev)
        # return rollout.t, rollout.y

        dt = 15e-3
        num_step = np.round((t_span[1]-t_span[0])/dt).astype(np.int64)
        t_eval = np.linspace(t_span[0],t_span[1],num_step+1)
        q_traj = np.zeros((self.batch,num_step+1,3,self.dof))
        p_traj = np.zeros((self.batch,num_step+1,self.ng,3))
        q_traj[:,0] = self.qqq
        p_traj[:,0] = self.g[...,0:3,3]
        fc_traj = np.zeros((self.batch,num_step+1,self.ng-1,3))
        Ek = np.zeros(num_step)
        Ep = np.zeros(num_step)
        Ee = np.zeros(num_step)

        if method == "Newmark":
            self.Newmark_init(dt = dt)
            for i in range(num_step):
                t0 = time.time()
                sol = root(lambda q: self.Newmark_residual(q,u[i]), x0=q_traj[...,i,0,:])
                t1 = time.time()
                print("root finding time: ",t1-t0)
                print(sol.message)
                q = np.reshape(sol.x,(-1,self.dof))
                qdot = -self.Newmark[0,0]*q + self.hqdot
                qddot = -self.Newmark[1,0]*q + self.hqddot
                self.qqq = np.stack((q,qdot,qddot),axis=-2)
                q_traj[:,i+1] = self.qqq
                self.forward_kinematics(q, qdot)
                p_traj[:,i+1] = self.g[...,:,0:3,3]
                self.hqdot = self.Newmark[0]@self.qqq
                self.hqddot = self.Newmark[1]@self.qqq

                fc_traj[:,i+1,-1,:],fc_traj[:,i+1,:-1,:] = contactForce(self.g[...,1:,:,:],self.eta[...,1:,0:3],1)
                # fc_traj[:,i+1,:-1,:] = fc_traj[:,i+1,:-1,:] + self.rhoAg

                J = self.J[...,1:-1,:,:]
                JT = np.swapaxes(J,-1,-2)
                wq = self.L*self.wq
                M = np.sum(wq[:,None,None]*JT@self.Ms@J, axis=-3) # batch x dof x dof
                Ek[i] = qdot@M@qdot.T/2
                Ep[i] = -self.rhoAg[0]*np.sum(wq*self.g[0,1:-1,0,3]) - self.FL[0]*self.g[0,-1,0,3]
                Ee[i] = self.L*q@self.K@q.T/2

        return t_eval, q_traj, p_traj, fc_traj, Ek, Ep, Ee

    
    def strong_form_dynamics(self, q, qdot):
        # for debugging only
        # Lambda^prime = Ms*eta^dot + coad(eta)*Ms*eta - coad(xi)*Lambda - Fext
        # Lambda = Ks*epsilon + Ds*xi^dot + Act*tau
        tau = np.zeros(4)

        # cache
        Bg = self.Bg
        Ms = self.Ms[None,...]
        Ks = self.Ks[None,...]
        Ds = self.Ds[None,...]
        # q_idx = np.mod(np.arange(self.ng), 4) != 0
        J = self.J
        JT = np.swapaxes(J,-1,-2)
        # JLT = self.J[-1,...].T
        epsilon = Bg@q
        xi = epsilon + self.xisr
        xidot = Bg@qdot
        rt = self.rt[None,...] # None x 3 x nt

        # forward kinematics
        self.forward_kinematics(q, qdot) # less calc

        # actuation
        pbs = np.cross(xi[:,3:6,None],rt, axis=1) + xi[:,0:3,None] # ng x 3 x nt
        pbs_norm = np.linalg.norm(pbs, ord=2, axis=1, keepdims=True) # ng x 1 x nt
        pbs_n = pbs / pbs_norm # ng x 3 x nt
        Act = np.concatenate((pbs_n, np.cross(rt,pbs_n, axis=1)), axis=1) # ng x 6 x nt

        Lambda = Ks@epsilon[...,None] + Ds@xidot[...,None]# + Act@tau
        # print(coad(xi)[0,...])
        # print(coad(xi)@Lambda)

        etadot = np.linalg.solve(Ms, -coad(self.eta)@Ms@self.eta[...,None] + coad(xi)@Lambda)

        return np.squeeze(etadot, axis=-1)
    
    # orientations integration with quaternion
    # Y: [p(0:3), h(3:7), u(7:10), v(10:13), q(13:16), w(16:19)], h being the quaternion
    def strong_form_dynamic_ODE(self, s, y, xi, xidot, xiddot):
        # for debugging only

        # cache
        Ms = self.Ms
        Ks = self.Ks
        Ds = self.Ds

        rt = self.rt

        h = y[3:7]
        R = Rh(h)
        eta = y[7:13]
        etadot = y[13:19]
        Lambdai = y[19:25]

        # forward kinematics
        ps  = R@xi[0:3]
        hs  = 0.5*hat_for_h(xi[3:6])@h
        etas = -ad(xi)@eta + xidot
        etadots = -ad(xi)@etadot - ad(xidot)@eta + xiddot

        # actuation
        # pbs = np.cross(xi[:,3:6,None],rt, axis=1) + xi[:,0:3,None] # ng x 3 x nt
        # pbs_norm = np.linalg.norm(pbs, ord=2, axis=1, keepdims=True) # ng x 1 x nt
        # pbs_n = pbs / pbs_norm # ng x 3 x nt
        # Act = np.concatenate((pbs_n, np.cross(rt,pbs_n, axis=1)), axis=1) # ng x 6 x nt
        
        Lambda = Ks@(xi-self.xisr) + Ds@xidot# + Act@tau
        Lambdas = Ms@etadot - ad(eta).T@Ms@eta + ad(xi).T@Lambda# - f

        return np.concatenate([ps,hs,etas,etadots,Lambdas]), Lambda-Lambdai
    
    def strong_form_dynamic_shooting(self,uyddot):
        self.xiddot[:,4] = uyddot
        h0 = np.array([1,0,0,0])
        eta0 = np.zeros(3)
        etadot0 = np.zeros(3)
        Lambda0 = self.Ks@(self.xi[0]-self.xisr) + self.Ds@self.xidot[0]
        self.Y[0] = np.concatenate([self.p0,h0,eta0,etadot0,Lambda0])

        # Y = euler_ivp(self.Cosserat_dynamic_ODE, (0,self.L), y0, self.N)
        ds = self.ds
        err = np.zeros((self.N))
        for j in range(self.N-1):
            ys,err[j] = self.Cosserat_dynamic_ODE(self.s_step[j],self.Y[j],self.xi[j],self.xidot[j],self.xiddot[j])
            self.Y[j+1] = self.Y[j] + ds*ys

        hL = self.Y[-1,3:7]
        RL = Rh(hL)
        WL = np.concatenate((RL.T@self.FL,RL.T@self.ML))
        LambdaL = self.Y[-1,19:25]

        return LambdaL - WL
    
    def Cosserat_dynamic_BVP(self, ig):
        t0 = time.time()
        sol = root(self.strong_form_dynamic_shooting,ig,method='lm')
        t1 = time.time()
        return sol.x, t1-t0


def main():
    delta = 0.005 # tendon offset 0.015
    MP = {
        "L": 1.0, # length 0.2
        "r": 0.005, # rod radius 0.001
        "E": 5e+7, # Young's modulus 200e9
        "nu": 0.5, # Poisson's ratio 0.25
        "rho": 1100, # density 1100
        "Dbt": 5e-3*np.eye(3), # damping 5e-7
        "Dse": 5e-4*np.eye(3), # damping 5e-8
        "usr": np.zeros(3), # natural twist
        "vsr": np.array([0,0,1]), # natural twist
        "g": np.zeros(3), # gravitational acceleration
        "rt": np.array([[delta,0,0],
                    [0,delta,0],
                    [-delta,0,0],
                    [0,-delta,0]])
    }
    
    TDCR = continuumRobot_GVS(MP, model="Kirchhoff", batch=1)

    def tau(t):
        # if t <= 0:
        #     return np.array([20,0,0,0])
        # else:
        #     return np.zeros(4)
        return np.zeros(4)

    
    # q = np.concatenate([np.zeros(TDCR.nb*3),np.ones(TDCR.nb),np.zeros(TDCR.nb*2)])
    q = np.zeros((TDCR.nb*3))
    # kappa = 1.90985932
    # q = np.concatenate([np.zeros(TDCR.nb),-np.ones(TDCR.nb)*kappa,np.zeros(TDCR.nb)])
    # q = TDCR.static_solve(np.array([0,0,0,0]), q)
    # qdot = np.concatenate([np.zeros(TDCR.nb),np.ones(TDCR.nb)*10,np.zeros(TDCR.nb)])
    qdot = np.zeros_like(q)
    # qdot = np.ones((TDCR.nb*3))*10
    TDCR.forward_kinematics(q,qdot)
    
    # TDCR.set_state(q,qdot)
    # TDCR.tau = tau
    # TDCR.tau = np.array([20,0,0,0])
    # TDCR.tau = np.zeros(4)
    # t0 = time.time()
    # TDCR.forward_kinematics(q,q)
    # x = TDCR.static_solve(np.array([1,0]), q)
    # print(x[TDCR.nb:2*TDCR.nb])
    # dy = TDCR.Cosserat_dynamic_ODE(0,np.concatenate((q,qdot)))
    # TDCR.forward_kinematics(q,dy[TDCR.dof:]*1e-3)

    t, q_traj, p_traj, fc_traj, Ek, Ep, Ee = TDCR.roll_out(tau,np.array([0,1.0]))

    # x = TDCR.static_solve(np.array([100,0,0,0]), q)
    # x = TDCR.static_solve(np.array([0,0,0,0]), q) # zero!
    # err = TDCR.Cosserat_static_error(q)
    # etadots = TDCR.strong_form_dynamics(q,qdot)
    # t1 = time.time()
    # print(t1-t0)

    num_step = np.shape(Ek)[0]

    # energy
    fig,ax = plt.subplots()
    # ax.plot(t,p_traj[0,:,-1,0])
    ax.plot(t[1:],Ek)
    ax.plot(t[1:],Ep)
    ax.plot(t[1:],Ee)
    ax.plot(t[1:],Ek+Ep+Ee)
    plt.show()

    # system states
    fig,ax = plt.subplots(3)
    xi_traj = np.squeeze(TDCR.Bg@q_traj[...,None,:,None],axis=-1)
    qline = ax[0].plot(TDCR.sg,xi_traj[0,0,0,:,4])
    ax[0].set_ylim((-20,20))
    qdline = ax[1].plot(TDCR.sg,xi_traj[0,0,1,:,4])
    ax[1].set_ylim((-50,50))
    qddline = ax[2].plot(TDCR.sg,xi_traj[0,0,2,:,4])
    ax[2].set_ylim((-100,100))
    def update(frame):
        # update the line plot:
        qline[0].set_ydata(xi_traj[0,frame,0,:,4])
        qdline[0].set_ydata(xi_traj[0,frame,1,:,4])
        qddline[0].set_ydata(xi_traj[0,frame,2,:,4])
        return qline,qdline,qddline
    ani = animation.FuncAnimation(fig=fig, func=update, frames=num_step, interval=5)
    plt.show()

    # workspace simulation
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    line = ax.plot(p_traj[0,0,:,0],p_traj[0,0,:,1],p_traj[0,0,:,2])
    points = ax.scatter(p_traj[0,0,:,0],p_traj[0,0,:,1],p_traj[0,0,:,2], s=2, c='y')
    ax.plot_surface(cylinder_x,cylinder_y,cylinder_z, alpha=0.5, color='r')
    # line = ax.plot(p[:,0],p[:,1],p[:,2])
    forces = [ax.quiver(p_traj[0,0,1:,0],p_traj[0,0,1:,1],p_traj[0,0,1:,2],
               fc_traj[0,0,:,0],fc_traj[0,0,:,1],fc_traj[0,0,:,2], normalize=False)]
    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_zlim([0,0.2])
    ax.set_aspect('equal')
    # plt.show()
    # print(line[0].get_data_3d())

    def update(frame):
        # update the line plot:
        line[0].set_data_3d(p_traj[0,frame,:,0],p_traj[0,frame,:,1],p_traj[0,frame,:,2])
        points.set_offsets(p_traj[0,frame,:,0:2]) # x,y
        points.set_3d_properties(p_traj[0,frame,:,2],zdir='z') # z

        forces[0].remove()
        forces[0] = ax.quiver(p_traj[0,frame,1:,0],p_traj[0,frame,1:,1],p_traj[0,frame,1:,2],
               fc_traj[0,frame,:,0],fc_traj[0,frame,:,1],fc_traj[0,frame,:,2], normalize=False)
        ax.set_aspect('equal')
        return line, points, forces[0]


    ani = animation.FuncAnimation(fig=fig, func=update, frames=num_step, interval=5)
    plt.show()
    # writer = animation.PillowWriter(fps=200,
    #                                 metadata=dict(artist='Me'),
    #                                 bitrate=1800)
    # ani.save(filename="/continuumSim/figures/rod_slipping.gif", writer=writer)

if __name__ == "__main__":
    main()