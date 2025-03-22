import numpy as np
# from casadi import *
from scipy.interpolate import BSpline
# from scipy.spatial.transform import Rotation
# from scipy.linalg import solve
from scipy.optimize import root
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from Utils.Math import *
from SE3 import *
# from playground import is_pd

class continuumRobot_GVS:
    def __init__(self, MP, deg=4, p=3, nb=12, model="Cosserat"):
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
        self.BDF(0,5e-3)

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
        self.Kbt_c0Dbt = self.Kbt + self.c0*MP["Dbt"]
        self.Kse_c0Dse = self.Kse + self.c0*MP["Dse"]
        self.invKbt = np.linalg.inv(self.Kbt)
        self.invKse = np.linalg.inv(self.Kse)
        self.inv_Kbt_c0Dbt = np.linalg.inv(self.Kbt + self.c0*MP["Dbt"])
        self.inv_Kse_c0Dse = np.linalg.inv(self.Kse + self.c0*MP["Dse"])
        self.usr = MP["usr"]
        self.vsr = MP["vsr"]
        self.xisr = np.concatenate([self.vsr,self.usr])
        self.Kbtusr = self.Kbt@MP["usr"]
        self.Ksevsr = self.Kse@MP["vsr"]

        # g = np.array([0,0,-9.8])
        g = np.array([0,0,0])
        self.rho = MP["rho"]
        self.rhoAg = self.rho*A*g
        self.rhoA  = self.rho*A
        self.Ms = np.zeros((6,6))
        self.Ms[0:3,0:3] = self.rhoA*np.eye(3)
        self.Ms[3:6,3:6] = self.rho*self.I

        self.rt = MP["rt"]
        self.rt_hat = hat(self.rt)
        self.rt = self.rt.T

        # Generalized stiffness matrix
        self.K = np.sum(self.wq[:,None,None]*self.BqT@self.Ks[None,...]@self.Bq, axis=0)
        # Generalized damping matrix
        self.D = np.sum(self.wq[:,None,None]*self.BqT@self.Ds[None,...]@self.Bq, axis=0)
        
        # controllables
        self.tau = np.zeros_like(self.rt[0])
        self.R0 = np.eye(3)
        self.p0 = np.zeros(3)

        # external loads
        # self.FL = np.zeros(3)
        self.FL = np.array([20,0,0])
        self.ML = np.zeros(3)
        # self.ML = np.array([0,0.3,0])
        self.WL = np.concatenate((self.FL,self.ML))

        # cache
        # self.Y = np.zeros((self.N,19)) # p,h,u,v,q,w
        # self.Z = np.zeros((self.N,18)) # u,v,q,w,us,vs
        # self.Zh = np.zeros((self.N,18))
        self.q = np.zeros(self.dof)
        self.qdot = np.zeros_like(self.q)
        self.qddot = np.zeros_like(self.q)
        self.g = np.tile(np.eye(4),(self.ng,1,1))
        self.J = np.zeros((self.ng,6,self.dof))
        self.Jdot = np.zeros_like(self.J)
        self.eta = np.zeros((self.ng,6))
        self.M = np.zeros((self.dof,self.dof))
        self.C = np.zeros((self.dof,self.dof))
        self.A = np.zeros((self.dof,self.rt.shape[1]))
        self.f = np.zeros(self.dof)

        # debugging
        self.N = 41
        self.xi = np.zeros((self.N,6))
        self.xi[:,2] = np.ones(self.N)
        self.xidot = np.zeros_like(self.xi)
        self.xiddot = np.zeros_like(self.xidot)
        self.s_step = np.linspace(0,self.L,self.N)
        self.ds = self.s_step[1] - self.s_step[0]
        self.Y = np.zeros((self.N,25))

    def set_state(self,q,qdot):
        self.q = q
        self.qdot = qdot
    
    def BDF(self,alpha,dt):
        # https://en.wikipedia.org/wiki/Backward_differentiation_formula
        self.dt = dt
        self.c0 = (1.5 + alpha)/(self.dt*(1 + alpha))
        self.c1 = -2/self.dt
        self.c2 = (0.5 + alpha)/(self.dt*(1 + alpha))
        self.d1 = alpha/(1 + alpha)

    def Newmark(self, dt=5e-3, beta=1/4, gamma=1/2):
        a = beta*dt**2
        b = gamma*dt
        fn = self.q + dt*self.qdot + dt**2*(1/2-beta)*self.qddot
        hn = self.qdot + dt*(1-gamma)*self.qddot
        self.q = a*self.qddot + fn
        self.qdot = b*self.qddot + hn
    
    def forward_kinematics(self, q, qdot):
        hg = self.hg[:,None]

        xi_z1 = self.Bz1@q + self.xisr # body twist at all Zanna quadrature points
        xi_z2 = self.Bz2@q + self.xisr # this expression only works for q shape (dof,)
        ad_xi_z1 = ad(xi_z1)

        Z2 = np.sqrt(3)*hg**2/12
        Omega = hg/2*(xi_z1+xi_z2) + \
            Z2*np.squeeze((ad_xi_z1@xi_z2[...,None]),axis=-1) # 4th-order Zanna collocation (4.19)
        DOmegaDq  = hg[...,None]/2*(self.Bz1+self.Bz2) + \
                    Z2[...,None]*(ad_xi_z1@self.Bz2-ad(xi_z2)@self.Bz1)

        D2OmegaDq2 = 2*Z2[...,None]*(ad(self.Bz1@qdot)@self.Bz2)

        expOmega, dexpOmega, ddexpOmegadt = expTdSE3(Omega, DOmegaDq@qdot)

        AdinvexpOmega = Adinv(expOmega)
        J_rel = dexpOmega@DOmegaDq
        for i in range(self.ng-1):
            self.g[i+1,:,:] = self.g[i,:,:]@expOmega[i,:,:]
            self.J[i+1,:,:] = AdinvexpOmega[i,:,:]@(self.J[i,:,:] + J_rel[i,:,:])
        
        self.eta = self.J@qdot
        Jdot_rel = ad(self.eta[:-1,...])@J_rel + ddexpOmegadt@DOmegaDq + dexpOmega@D2OmegaDq2
        for i in range(self.ng-1):
            self.Jdot[i+1,:,:] = AdinvexpOmega[i,:,:]@(self.Jdot[i,:,:] + Jdot_rel[i,:,:])


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
        with np.printoptions(threshold=np.inf):
            print(J[-1,...])
            fig,ax = plt.subplots()
            plt.spy(J[-1,...])
            plt.show()
            # print(JT[-1,...]@J[-1,...])
            # fig,ax = plt.subplots()
            # plt.spy(JT[-1,...]@J[-1,...])
            # plt.show()

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
        # print(tau)
        # print(qdot)
        # print(self.f)
        # print(self.M)
        # print(self.A@tau + self.f - (self.C+self.L*self.D)@qdot - self.L*self.K@q)

        return np.concatenate([qdot,qddot])

    def Cosserat_static_error(self, q):
        # Kq = Au + f

        # forward kinematics
        self.forward_kinematics(q, np.zeros_like(q)) # less calc
        
        # cache
        wq = self.L*self.wq[:,None,None]
        Bq = self.Bq
        # q_idx = np.mod(np.arange(self.ng), 4) != 0
        # J = self.J[1:-1,...]
        JT = np.swapaxes(self.J[1:-1,...],-1,-2)
        JLT = self.J[-1,...].T
        xiq = Bq@q + self.xisr
        rt = self.rt[None,...] # None x 3 x nt

        # Generalized actuation
        pbs = np.cross(xiq[:,3:6,None],rt, axis=1) + xiq[:,0:3,None] # nq x 3 x nt
        pbs_norm = np.linalg.norm(pbs, ord=2, axis=1, keepdims=True) # nq x 1 x nt
        pbs_n = pbs / pbs_norm # nq x 3 x nt
        Act = np.concatenate((pbs_n, np.cross(rt,pbs_n, axis=1)), axis=1) # nq x 6 x nt
        self.A = -np.sum(wq*self.BqT@Act, axis=0)# + JLT@ActL # dof x nt

        # Generalized external load
        fc = np.zeros(3) #fc[j] 
        f  = self.rhoAg + fc
        F = np.zeros((JT.shape[0],6))
        # F = np.concatenate((np.zeros((JT.shape[0],4)),0.3*np.ones((JT.shape[0],1)),np.zeros((JT.shape[0],1))),axis=1)
        # WL = np.concatenate((self.g[-1,0:3,0:3].T@self.FL,self.g[-1,0:3,0:3].T@self .ML))
        self.f = np.squeeze(np.sum(wq*JT@F[...,None], axis=0),axis=-1) + JLT@self.WL

        return self.L*self.K@q - (self.A@self.tau + self.f)
    
    def static_solve(self, tau, ig):
        self.tau = tau
        sol = root(self.Cosserat_static_error, x0=ig)
        print("success?",sol.success)
        print(sol.status)
        print(sol.message)
        print("nfev: ",sol.nfev)
        # print("njev: ",sol.njev)
        return sol.x
        
    
    def roll_out(self, tau, t_span):
        self.tau = tau
        y0 = np.concatenate((self.q,self.qdot))
        rollout = solve_ivp(self.Cosserat_dynamic_ODE, t_span, y0, method="BDF", t_eval=np.linspace(t_span[0],t_span[1],101))
        print(rollout.message)
        print("nfev = ",rollout.nfev)
        print("njev = ",rollout.njev)
        return rollout.t, rollout.y
    
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
        print(coad(xi)[0,...])
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

    
    # def Cosserat_dynamic_sim(self, dt, num_steps, alpha, input, ig):
    #     self.BDF(alpha,dt)
    #     trajectory = np.zeros((num_steps+1,*np.shape(self.Y)))
    #     self.tau = input[0]

    #     t0 = time.time()
    #     sol = root(self.Cosserat_static_shooting,ig,method='lm')
    #     ig = sol.x
    #     t1 = time.time()
    #     print("---------------------------------------------------")
    #     print("time step:", -1)
    #     print("converged:", sol.success)
    #     print("max abs error:", np.max(np.abs(sol.fun)))
    #     print("xi (nfev):", sol.nfev)
    #     print("time used:",t1-t0)
    #     trajectory[0] = self.Y
    #     Z_prev = self.Z

    #     for i in range(num_steps):
    #         t0 = t1
    #         self.Zh = self.c1*self.Z + self.c2*Z_prev
    #         Z_prev = self.Z
    #         self.tau = input[i+1]
    #         print(ig)
    #         sol = root(self.Cosserat_dynamic_shooting,ig,method='lm')
    #         ig = sol.x
    #         t1 = time.time()
    #         print("---------------------------------------------------")
    #         print("time step:", i)
    #         print("converged:", sol.success)
    #         print("max abs error:", np.max(np.abs(sol.fun)))
    #         print("xi (nfev):", sol.nfev)
    #         print("time used:",t1-t0)

    #         trajectory[i+1] = self.Y

    #     # t0 = t1
    #     # self.Zh = self.c1*self.Z + self.c2*Z_prev
    #     # Z_prev = self.Z
    #     # self.tau = input[0]
    #     # # print(ig)
    #     # res = self.Cosserat_dynamic_shooting(ig)
    #     # # ig = sol.x
    #     # print(res)
    #     # t1 = time.time()

    #     # trajectory[0+1] = self.Y

    #     return trajectory


def main():
    delta = 0.015 # tendon offset 0.015
    MP = {
        "L": 0.2, # length 0.2
        "r": 0.001, # rod radius 1
        "E": 200e+9, # Young's modulus 200
        "nu": 0.25, # Poisson's ratio
        "rho": 1100, # density
        "Dbt": 0*np.eye(3), # damping 5e-7
        "Dse": 0*np.eye(3), # damping 5e-8
        "usr": np.zeros(3), # natural twist
        "vsr": np.array([0,0,1]), # natural twist
        "g": np.zeros(3), # gravitational acceleration
        "rt": np.array([[delta,0,0],
                    [0,delta,0],
                    [-delta,0,0],
                    [0,-delta,0]])
    }
    
    TDCR = continuumRobot_GVS(MP, model="Kirchhoff")
    # print(TDCR.rhoA)
    # print(TDCR.rho*TDCR.I)
    # print(TDCR.Ms)

    def tau(t):
        if t < 0:
            return np.array([20,0,0,0])
        else:
            return np.zeros(4)

    # q = np.concatenate([np.zeros(TDCR.nb*3),np.ones(TDCR.nb),np.zeros(TDCR.nb*2)])
    q = np.zeros((TDCR.nb*3))
    kappa = 1.90985932
    # q = np.concatenate([np.zeros(TDCR.nb),np.ones(TDCR.nb)*kappa,np.zeros(TDCR.nb)])
    qdot = np.concatenate([np.ones(TDCR.nb),np.zeros(TDCR.nb*2)])
    # qdot = np.zeros(TDCR.nb*3)
    
    TDCR.set_state(q,qdot)
    TDCR.tau = tau
    # TDCR.tau = np.array([20,0,0,0])
    # TDCR.tau = np.zeros(4)
    t0 = time.time()
    # TDCR.forward_kinematics(q,q)
    # x = TDCR.static_solve(np.array([0,0,0,0]), q)
    dy = TDCR.Cosserat_dynamic_ODE(0,np.concatenate((q,qdot)))
    # TDCR.forward_kinematics(q,dy[TDCR.dof:]*1e-3)
    # t, y = TDCR.roll_out(tau,np.array([0,0.5]))
    # x = TDCR.static_solve(np.array([100,0,0,0]), q)
    # x = TDCR.static_solve(np.array([0,0,0,0]), q) # zero!
    # err = TDCR.Cosserat_static_error(q)
    # etadots = TDCR.strong_form_dynamics(q,qdot)
    t1 = time.time()
    print(t1-t0)
    print(np.max(TDCR.J))
    #with np.printoptions(threshold=np.inf):
    print(TDCR.M.shape)
    # print(is_pd(TDCR.M))
    # print(TDCR.K@x)
    fig,ax = plt.subplots()
    plt.spy(TDCR.M)
    plt.show()

    # print(TDCR.M)
    # print(x-q)
    # print(dy[TDCR.dof:])
    eta_t = TDCR.J@dy[TDCR.dof:]
    # print(etadots[:,0])
    # print(eta_t[:,0])
    # print(x)
    # print(err)
    # print(TDCR.Bq[:,4,:])
    xq = TDCR.Bq@dy[TDCR.dof:]
    fig,ax = plt.subplots()
    ax.plot(TDCR.sq,xq[:,4])
    plt.show()
    # print(err)

    fig,ax = plt.subplots()
    # ax.plot(TDCR.sg,etadots[:,0])
    ax.plot(TDCR.sg,eta_t[:,0])
    plt.show()


    # print(t)
    # x_traj = np.zeros(y.shape[1])
    # for i in range(y.shape[1]):
    #     TDCR.forward_kinematics(y[:TDCR.dof,i],y[TDCR.dof:,i])
    #     x_traj[i] = TDCR.g[-1,1,3]

    # theta = TDCR.L*kappa
    # pr = np.array([1-np.cos(theta),np.sin(theta)])/kappa
    # vr = (-pr/kappa + TDCR.L/kappa*np.array([np.sin(theta),np.cos(theta)]))*kappa
    # print(pr)
    # print(vr)

    p = TDCR.g[:,0:3,3]
    v = np.squeeze(TDCR.g[...,0:3,0:3]@TDCR.eta[...,0:3,None],axis=-1)
    # print(p[-1,:])
    # print(v[-1,:])
    # print(TDCR.eta[-1,:])
    # print(TDCR.L*kappa)

    # num_steps = 40
    # # tendon release
    # input = np.zeros((num_steps+1,4))
    # input[0,0] = 20

    # traj = TDCR.Cosserat_dynamic_sim(0.002, num_steps, 0, input, np.array([0,0,0,0,0,1]))

    # fig,ax = plt.subplots()
    # ax.plot(t,x_traj)
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # line = ax.plot(traj[1,:,0],traj[1,:,1],traj[1,:,2])
    line = ax.plot(p[:,0],p[:,1],p[:,2])
    ax.quiver(p[:,0],p[:,1],p[:,2], v[:,0],v[:,1],v[:,2], normalize=False)
    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # plt.show()

    # def update(frame):
    #     # update the line plot:
    #     line[0].set_xdata(traj[frame,:,0])
    #     line[0].set_ydata(traj[frame,:,1])
    #     line[0].set_ydata(traj[frame,:,2])
    #     return line


    # ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=5)
    plt.show()

if __name__ == "__main__":
    main()