import numpy as np
import casadi as ca
from scipy.interpolate import BSpline
# from scipy.optimize import root
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from SE3_casadi import *
from contact_casadi import *

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
        self.nq = len(self.sq)
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

        self.DOmegaDq_term1 = self.hg[...,None,None]/2*(self.Bz1+self.Bz2)

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
        # self.rt_hat = hat(self.rt)
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
        self.FL = ca.SX(3,1)
        # self.FL = np.array([1,0,0])
        # self.ML = np.zeros(3)
        # self.ML = np.array([0,0.3,0])
        # self.WL = np.concatenate((self.FL,self.ML))

        # symbolic variables
        self.q = ca.SX.sym('q',self.dof)
        self.qdot = ca.SX.sym('qdot',self.dof)
        self.qddot = ca.SX.sym('qddot',self.dof)
        self.qqq = ca.horzcat(self.q,self.qdot,self.qddot)
        self.u = ca.SX.sym('u',self.udof)

        # cache
        self.g = [ca.SX.eye(4)] # better implemented as R, p
        self.J = [ca.SX(6,self.dof)]
        self.Jdot = [ca.SX(6,self.dof)]
        self.eta = [ca.SX(6,1)]
        for i in range(self.ng):
            self.g.append(ca.SX.eye(4))
            self.J.append(ca.SX(6,self.dof))
            self.Jdot.append(ca.SX(6,self.dof))
            self.eta.append(ca.SX(6,1))
        self.M = ca.SX(self.dof,self.dof)
        self.C = ca.SX(self.dof,self.dof)
        self.A = ca.SX(self.dof,self.rt.shape[1])
        self.f = ca.SX(self.dof,1)

        # casadi functions
        self.static_residual = ca.Function('static_residual',[self.q, self.u], [self.Cosserat_static_residual(self.q, self.u)],
                                           ['q','u'],['residual'])
        self.static_solver = ca.rootfinder('static_solver','newton',self.static_residual)

        J = ca.jacobian(self.static_residual(self.q,self.u), self.q)
        self.J_fun = ca.Function('J', [self.q,self.u], [J], ['q','u'],['static_jacobian'])

        # self.dynamic_residual = ca.Function('dynamic_residual',[self.q, self.qdot, self.u], self.Cosserat_dynamic_residual(self.q, self.qdot, self.u))


        

    def set_state(self,q,qdot):
        self.qqq[...,0,:] = q
        self.qqq[...,1,:] = qdot

    def Newmark_init(self, dt=5e-3, beta=1/4, gamma=1/2):
        self.dt = dt
        self.Newmark = np.array([[-gamma/(beta*dt),(1-gamma/beta),(1-gamma/(2*beta))*dt],
                                 [-1/(beta*dt**2),-1/(beta*dt),1-1/(2*beta)]])
        self.hqdot = self.Newmark[0]@self.qqq
        self.hqddot = self.Newmark[1]@self.qqq
    
    def forward_kinematics(self, q, qdot):
        hg = self.hg
        Z2 = np.sqrt(3)*hg**2/12

        for i in range(self.ng-1):
            xi_z1 = self.Bz1[i]@q + self.xisr # body twist at all Zanna quadrature points
            xi_z2 = self.Bz2[i]@q + self.xisr # q shape (dof), result shape (6)
            ad_xi_z1 = ad(xi_z1) # (6, 6)

            # print((Z2[i]*(ad_xi_z1@self.Bz2[i]-ad(xi_z2)@self.Bz1[i])).shape)

            Omega = hg[i]/2*(xi_z1+xi_z2) + Z2[i]*ad_xi_z1@xi_z2 # 4th-order Zanna collocation (4.19)
            DOmegaDq  = ca.SX(self.DOmegaDq_term1[i]) + \
                        Z2[i]*(ad_xi_z1@self.Bz2[i]-ad(xi_z2)@self.Bz1[i]) # (6, dof)

            # D2OmegaDq2 = 2*Z2[...,None]*(ad(np.squeeze(self.Bz1@qdot[...,None,:,None],axis=-1))@self.Bz2)

            expOmega, dexpOmega = expTdSE3(Omega, DOmegaDq@qdot, order=1) #, ddexpOmegadt

            AdinvexpOmega = Adinv(expOmega)
            J_rel = dexpOmega@DOmegaDq
            self.g[i+1] = self.g[i]@expOmega
            self.J[i+1] = AdinvexpOmega@(self.J[i] + J_rel)
        
            self.eta[i+1] = self.J[i]@qdot
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
        tau = ca.vertcat(u,-u)

        # forward kinematics
        self.forward_kinematics(q,qdot)
        
        # cache
        wq = self.L*self.wq
        Bq = self.Bq
        Ms = self.Ms#[None,...]
        J = self.J[1:-1]
        JLT = self.J[-1].T
        rt = self.rt # None x 3 x nt

        # external forces
        FL,fc = contactForce(self.g[1:],self.eta[1:],self.hg[-1])
        FL += self.g[-1][0:3,0:3].T@self.FL
        self.f += JLT[:,0:3]@FL

        for i in range(self.nq):
            xiq = Bq[i]@q + self.xisr # 6 x 1

            # Generalized mass matrix
            self.M += wq[i]*J[i].T@Ms@J[i] # batch x dof x dof

            # Generalized centrifugal & Coriolis matrix
            # self.C = np.sum(wq*JT@(Ms@self.Jdot[...,1:-1,:,:]+coad(self.eta[...,1:-1,:])@Ms@J), axis=-3)
            self.C += wq[i]*J[i].T@(coad(self.eta[i+1])@Ms@J[i])

            # Generalized actuation
            pbs = ca.cross(ca.repmat(xiq[3:6],1,np.shape(rt)[-1]),rt) + xiq[0:3] # 3 x nt
            pbs_norm = ca.sqrt(ca.sum2(pbs**2)) # 1 x nt
            pbs_n = pbs / pbs_norm # 3 x nt
            Act = ca.vertcat(pbs_n, ca.cross(rt,pbs_n)) # 6 x nt

            self.A -= wq[i]*self.BqT[i]@Act # dof x nt

            # Generalized external load
            fc[i] += self.g[i+1][0:3,0:3].T@self.rhoAg # 3
            self.f += wq[i]*J[i][0:3,:].T@fc[i]

        return self.M@qddot + (self.C+self.L*self.D)@qdot + self.L*self.K@q - self.A@tau - self.f
    
    def Newmark_residual(self, q, u):
        q = np.reshape(q,(-1,self.dof))
        qdot = -self.Newmark[0,0]*q + self.hqdot
        qddot = -self.Newmark[1,0]*q + self.hqddot
        return self.Cosserat_dynamic_residual(q, qdot, qddot, u)

    def Cosserat_static_residual(self, q, u):
        # Kq = Au + f

        tau = ca.vertcat(u,-u)

        # forward kinematics
        self.forward_kinematics(q,np.zeros(self.dof))
        
        # cache
        wq = self.L*self.wq
        Bq = self.Bq
        J = self.J[1:-1]
        JLT = self.J[-1].T
        rt = self.rt#[None,...] # None x 3 x nt

        # external forces
        FL,fc = contactForce(self.g[1:],self.eta[1:],self.hg[-1])
        FL += self.g[-1][0:3,0:3].T@self.FL
        self.f += JLT[:,0:3]@FL

        for i in range(self.nq):
            xiq = Bq[i]@q + self.xisr # 6 x 1

            # Generalized actuation
            pbs = ca.cross(ca.repmat(xiq[3:6],1,np.shape(rt)[-1]),rt) + xiq[0:3] # 3 x nt
            pbs_norm = ca.sqrt(ca.sum2(pbs**2)) # 1 x nt
            pbs_n = pbs / pbs_norm # 3 x nt
            Act = ca.vertcat(pbs_n, ca.cross(rt,pbs_n)) # 6 x nt

            self.A -= wq[i]*self.BqT[i]@Act # dof x nt

            # Generalized external load
            fc[i] += self.g[i+1][0:3,0:3].T@self.rhoAg # 3
            self.f += wq[i]*J[i][0:3,:].T@fc[i]

        return self.L*self.K@q - self.A@tau - self.f
    
    # def static_solve(self, u, ig):
    #     sol = root(lambda q: self.Cosserat_static_residual(q,u), x0=ig)
    #     print("success?",sol.success)
    #     print(sol.status)
    #     print(sol.message)
    #     print("nfev: ",sol.nfev)
    #     # print("njev: ",sol.njev)
    #     return sol.x
    
    # def roll_out(self, u, t_span, method="Newmark"):
    #     # y0 = np.concatenate((self.q,self.qdot))
    #     # rollout = solve_ivp(self.Cosserat_dynamic_ODE, t_span, y0, method="Radau", t_eval=np.linspace(t_span[0],t_span[1],101))
    #     # print(rollout.message)
    #     # print("nfev = ",rollout.nfev)
    #     # print("njev = ",rollout.njev)
    #     # return rollout.t, rollout.y

    #     dt = 15e-3
    #     num_step = np.round((t_span[1]-t_span[0])/dt).astype(np.int64)
    #     t_eval = np.linspace(t_span[0],t_span[1],num_step+1)
    #     q_traj = np.zeros((self.batch,num_step+1,3,self.dof))
    #     p_traj = np.zeros((self.batch,num_step+1,self.ng,3))
    #     q_traj[:,0] = self.qqq
    #     p_traj[:,0] = self.g[...,0:3,3]
    #     fc_traj = np.zeros((self.batch,num_step+1,self.ng-1,3))
    #     Ek = np.zeros(num_step)
    #     Ep = np.zeros(num_step)
    #     Ee = np.zeros(num_step)

    #     if method == "Newmark":
    #         self.Newmark_init(dt = dt)
    #         for i in range(num_step):
    #             t0 = time.time()
    #             sol = root(lambda q: self.Newmark_residual(q,u[i]), x0=q_traj[...,i,0,:])
    #             t1 = time.time()
    #             print("root finding time: ",t1-t0)
    #             print(sol.message)
    #             q = np.reshape(sol.x,(-1,self.dof))
    #             qdot = -self.Newmark[0,0]*q + self.hqdot
    #             qddot = -self.Newmark[1,0]*q + self.hqddot
    #             self.qqq = np.stack((q,qdot,qddot),axis=-2)
    #             q_traj[:,i+1] = self.qqq
    #             self.forward_kinematics(q, qdot)
    #             p_traj[:,i+1] = self.g[...,:,0:3,3]
    #             self.hqdot = self.Newmark[0]@self.qqq
    #             self.hqddot = self.Newmark[1]@self.qqq

    #             fc_traj[:,i+1,-1,:],fc_traj[:,i+1,:-1,:] = contactForce(self.g[...,1:,:,:],self.eta[...,1:,0:3],1)
    #             # fc_traj[:,i+1,:-1,:] = fc_traj[:,i+1,:-1,:] + self.rhoAg

    #             J = self.J[...,1:-1,:,:]
    #             JT = np.swapaxes(J,-1,-2)
    #             wq = self.L*self.wq
    #             M = np.sum(wq[:,None,None]*JT@self.Ms@J, axis=-3) # batch x dof x dof
    #             Ek[i] = qdot@M@qdot.T/2
    #             Ep[i] = -self.rhoAg[0]*np.sum(wq*self.g[0,1:-1,0,3]) - self.FL[0]*self.g[0,-1,0,3]
    #             Ee[i] = self.L*q@self.K@q.T/2

    #     return t_eval, q_traj, p_traj, fc_traj, Ek, Ep, Ee


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
    
    TDCR = continuumRobot_GVS(MP, model="Kirchhoff")

    def tau(t):
        # if t <= 0:
        #     return np.array([20,0,0,0])
        # else:
        #     return np.zeros(4)
        return np.zeros(4)

    
    # # q = np.concatenate([np.zeros(TDCR.nb*3),np.ones(TDCR.nb),np.zeros(TDCR.nb*2)])
    # q = np.zeros((TDCR.nb*3))
    # # kappa = 1.90985932
    # # q = np.concatenate([np.zeros(TDCR.nb),-np.ones(TDCR.nb)*kappa,np.zeros(TDCR.nb)])
    # # q = TDCR.static_solve(np.array([0,0,0,0]), q)
    # # qdot = np.concatenate([np.zeros(TDCR.nb),np.ones(TDCR.nb)*10,np.zeros(TDCR.nb)])
    # qdot = np.zeros_like(q)
    # # qdot = np.ones((TDCR.nb*3))*10
    # TDCR.forward_kinematics(q,qdot)

    J0 = TDCR.J_fun(q = np.zeros((TDCR.nb*3)), u=[0,0])

    for i in range(TDCR.dof):
        print(J0['static_jacobian'][i,:])

    t0 = time.time()
    q = TDCR.static_solver(q0=np.zeros((TDCR.nb*3)), u=[0,0])

    # t, q_traj, p_traj, fc_traj, Ek, Ep, Ee = TDCR.roll_out(tau,np.array([0,1.0]))

    t1 = time.time()
    print("total time: ", t1-t0)

    print(q)

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