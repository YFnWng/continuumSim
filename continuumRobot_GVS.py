import numpy as np
# from casadi import *
from scipy.interpolate import BSpline
# from scipy.spatial.transform import Rotation
from scipy.optimize import root
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from Utils.Math import *
from SE3 import *

class continuumRobot_GVS:
    def __init__(self, MP, deg=3, p=3, nb=12):
        # space integration parameters
        # self.N = 41
        self.L = MP["L"]
        # self.s_step = np.linspace(0,self.L,self.N)
        # self.ds = self.s_step[1] - self.s_step[0]
        xq, self.wq = np.polynomial.legendre.leggauss(deg) # Legendre zeros and quadrature weights
        self.xq = (xq + 1)/2
        xz, _ = np.polynomial.legendre.leggauss(2) # 4th-order Zanna quadrature (4.17)
        self.xz = (xz + 1)/2

        self.nb = nb # number of basis/control points
        self.sk = np.linspace(0, 1, p+nb+1-2*p) # joints of polynomial pieces, uniformly distributed
        knots = np.concatenate((np.zeros(p), self.sk, np.ones(p))) # knot points for B-spline
        # self.sk = np.unique(knots) # joints of polynomial pieces
        self.hk = self.sk[1:] - self.sk[:-1] # quadrature intervals
        self.sq = (self.sk[:-1,None] + self.hk[:,None]*self.xq[None,:]).flatten() # quadrature points between knots
        self.sg = np.sort(np.concatenate([self.sk,self.sq])) # locations of all sig points
        self.hg = self.sg[1:] - self.sg[:-1] # Magnus expansion intervals
        self.sz1 = self.sg[:-1] + self.hg*self.xz[0] # Zanna quadrature points between sig points
        self.sz2 = self.sg[:-1] + self.hg*self.xz[1] # Zanna quadrature points between sig points
        self.p = p # degree of B-spline
        # self.nb = len(knots) - p - 1 # number of B-spline basis
        self.ng = len(self.sg)
        self.N = 0
        self.dof = 6*self.nb
        # self.col = sites # collocation sites

        # collocation basis
        nz = self.ng-1
        Bk = np.zeros((len(self.sk),self.nb))
        Bq = np.zeros((len(self.sq),self.nb))
        Bz1 = np.zeros((nz,self.nb))
        Bz2 = np.zeros((nz,self.nb))
        # self.D = np.zeros_like(self.S,device=device)
        for i in range(self.nb):
            b = BSpline.basis_element(knots[i:i+p+2])
            active_sk = (self.sk>=knots[i]) & (self.sk<=knots[i+p+1])
            Bk[active_sk,i] = b(self.sk[active_sk])
            active_sq = (self.sq>=knots[i]) & (self.sq<=knots[i+p+1])
            Bq[active_sq,i] = b(self.sq[active_sq])
            active_sz1 = (self.sz1>=knots[i]) & (self.sz1<=knots[i+p+1])
            Bz1[active_sz1,i] = b(self.sz1[active_sz1])
            active_sz2 = (self.sz2>=knots[i]) & (self.sz2<=knots[i+p+1])
            Bz2[active_sz2,i] = b(self.sz2[active_sz2])
        Bk[-1,-1] = 1
        self.Bk = np.zeros((len(self.sk),6,self.dof))
        self.Bq = np.zeros((len(self.sq),6,self.dof))
        self.Bz1 = np.zeros((nz,6,self.dof))
        self.Bz2 = np.zeros((nz,6,self.dof))
        for i in range(6):
            self.Bk[:,i,i*self.nb:(i+1)*self.nb] = Bk
            self.Bq[:,i,i*self.nb:(i+1)*self.nb] = Bq
            self.Bz1[:,i,i*self.nb:(i+1)*self.nb] = Bz1
            self.Bz2[:,i,i*self.nb:(i+1)*self.nb] = Bz2
        

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
        self.K = np.diag([GA, GA, EA, EI, EI, GI3])
        self.Kbt = np.diag([EI, EI, GI3])
        self.Kse = np.diag([GA, GA, EA])
        self.Bbt = MP["Bbt"]
        self.Bse = MP["Bse"]
        self.Kbt_c0Bbt = self.Kbt + self.c0*MP["Bbt"]
        self.Kse_c0Bse = self.Kse + self.c0*MP["Bse"]
        self.invKbt = np.linalg.inv(self.Kbt)
        self.invKse = np.linalg.inv(self.Kse)
        self.inv_Kbt_c0Bbt = np.linalg.inv(self.Kbt + self.c0*MP["Bbt"])
        self.inv_Kse_c0Bse = np.linalg.inv(self.Kse + self.c0*MP["Bse"])
        self.usr = MP["usr"]
        self.vsr = MP["vsr"]
        self.xisr = np.concatenate([self.vsr,self.usr])
        self.Kbtusr = self.Kbt@MP["usr"]
        self.Ksevsr = self.Kse@MP["vsr"]

        # g = np.array([0,0,-9.8])
        g = np.array([0,0,0])
        self.rhoAg = MP["rho"]*A*g
        self.rhoA  = MP["rho"]*A
        self.Ms = np.diag(np.concatenate((self.rhoA*np.ones(3),[I1,I2,I3]))) #?

        self.rt = MP["rt"]
        self.rt_hat = hat(self.rt)
        
        # controllables
        self.tau = np.zeros_like(self.rt)
        self.R0 = np.eye(3)
        self.p0 = np.zeros(3)

        # external loads
        self.FL = np.zeros(3)
        self.ML = np.zeros(3)

        # cache
        self.Y = np.zeros((self.N,19)) # p,h,u,v,q,w
        self.Z = np.zeros((self.N,18)) # u,v,q,w,us,vs
        self.Zh = np.zeros((self.N,18))
        self.q = np.zeros(self.dof)
        self.qdot = None
        self.g = np.tile(np.eye(4),(self.ng,1,1))
        self.J = np.zeros((self.ng,6,self.dof))
        self.Jdot = None
        self.M = None
        self.C = None

    def set_q(self,q):
        self.q = q
    
    def BDF(self,alpha,dt):
        self.dt = dt
        self.c0 = (1.5 + alpha)/(self.dt*(1 + alpha))
        self.c1 = -2/self.dt
        self.c2 = (0.5 + alpha)/(self.dt*(1 + alpha))
        self.d1 = alpha/(1 + alpha)
    
    def forward_kinematics(self):
        xi_z1 = self.Bz1@self.q + self.xisr # body twist at all Zanna quadrature points
        xi_z2 = self.Bz2@self.q + self.xisr # note that this expression only works for q shape (dof,)
        ad_xi_z1 = ad(xi_z1)
        Omega = self.hg[:,None]/2*(xi_z1+xi_z2) + \
            np.sqrt(3)*self.hg[:,None]**2/12*np.squeeze((ad_xi_z1@xi_z2[...,None]),axis=-1) # 4th-order Zanna collocation (4.19)
        DOmegaDq  = self.hg[:,None,None]/2*(self.Bz1+self.Bz2) + \
                    (np.sqrt(3)*self.hg[:,None,None]**2)/12*(ad_xi_z1@self.Bz2-ad(xi_z2)@self.Bz1)
        expOmega, TexpOmega = expTSE3(Omega)
        AdinvexpOmega = Adinv(expOmega)
        J_rel = TexpOmega@DOmegaDq
        for i in range(self.ng-1):
            self.g[i+1,:,:] = self.g[i,:,:]@expOmega[i,:,:]
            self.J[i+1,:,:] = AdinvexpOmega[i,:,:]@(self.J[i,:,:] + J_rel[i,:,:])

    # def Cosserat_dynamic_ODE(self):
    #     # Generalized mass matrix
    #     self.M = np.sum(self.wq*self.J@self.Ms@self.J)

    #     # Generalized centrifugal & Coriolis matrix
    #     # self.C = self.wq*self.J'@(self.Ms@self.Jd+ad(eta)'@self.Ms*self.J)

    #     # Generalized actuation
    #     pbs = np.cross(u[None,:],rt) + v[None,:] # nt x 3
    #     pbs_norm = np.linalg.norm(pbs,ord=2,axis=-1) # nt
    #     pbs_n = pbs / pbs_norm[:,None]
    #     pbs_hat = hat(pbs_n) # nt x 3 x 3
    #     A_i = -pbs_hat @ pbs_hat * (self.tau[:,None,None] / (pbs_norm[:,None,None])) # nt x 3 x 3
    #     G_i = -A_i @ rt_hat # nt x 3 x 3
    #     a_i = np.squeeze(A_i @ np.cross(u,pbs)[...,None],axis=-1) # nt x 3
        
    #     a = np.sum(a_i,axis=0)
    #     b = np.sum(np.cross(rt,a_i),axis=0)
    #     A = np.sum(A_i,axis=0)
    #     G = np.sum(G_i,axis=0)
    #     H = np.sum(rt_hat@G_i,axis=0)

    #     # Generalized external load
    #     fc = np.zeros(3) #fc[j] 
    #     f  = self.rhoAg + fc
    

    #     rt = self.rt
    #     rt_hat = self.rt_hat
    #     Bbt = self.Bbt
    #     Bse = self.Bse
    #     usr = self.usr
    #     vsr = self.vsr
    #     Kbt = self.Kbt
    #     Kse = self.Kse
    #     I     = self.I
    #     c0    = self.c0

    #     h = y[3:7]
    #     R = Rh(h)
    #     u = y[7:10]
    #     v = y[10:13]
    #     q = y[13:16]
    #     w = y[16:19]
        
    #     uh = zh[0:3]
    #     vh = zh[3:6]
    #     qh = zh[6:9]
    #     wh = zh[9:12]
    #     ush = zh[12:15]
    #     vsh = zh[15:18]



    #     u_hat = hat(u)
    #     w_hat = hat(w)

    #     ut = c0*u + uh
    #     vt = c0*v + vh
    #     qt = c0*q + qh
    #     wt = c0*w + wh
        
        
        
    #     K = np.vstack((np.hstack((H + self.Kbt_c0Bbt, G.T)),
    #                     np.hstack((G, A + self.Kse_c0Bse))))
        
    #     mb = Kbt@(u - usr) + Bbt@ut
    #     nb = Kse@(v - vsr) + Bse@vt
        
    #     rhs = np.hstack([-b + self.rho*(np.cross(w,I@w) + I@wt) - np.cross(v,nb) - np.cross(u,mb) - Bbt@ush,
    #                     -a + self.rhoA*(np.cross(w,q)+qt) - R.T@f - np.cross(u,nb) - Bse@vsh]) # + C*q*norm(q) drag
        
    #     ps  = R@v
    #     hs  = 0.5*hat_for_h(u)@h
    #     us_vs = np.linalg.solve(K,rhs)
    #     qs  = vt - u_hat@q + w_hat@v
    #     ws  = ut - u_hat@w

    #     return np.concatenate([ps,hs,us_vs,qs,ws])

    # # Y : [p(0:3), h(3:12), u(12:15), v(15:18)]
    # # orientations integration with rotation matrices (in our numerical applications the rod is straight in rest configuration)
    # def Cosserat_static_ODE(self, s, y):
    #     del s
    #     f = self.rhoAg
    #     usr = self.usr
    #     vsr = self.vsr
    #     Kbt = self.Kbt
    #     Kse = self.Kse
    #     rt = self.rt
    #     rt_hat = self.rt_hat

    #     h = y[3:7]
    #     R = Rh(h)
    #     u = y[7:10]
    #     v = y[10:13]
        
    #     pbs = np.cross(u[None,:],rt) + v[None,:] # nt x 3
    #     pbs_norm = np.linalg.norm(pbs,ord=2,axis=-1) # nt
    #     pbs_n = pbs / pbs_norm[:,None]
    #     # print(pbs_norm)
    #     pbs_hat = hat(pbs_n) # nt x 3 x 3
    #     A_i = -pbs_hat @ pbs_hat * (self.tau[:,None,None] / (pbs_norm[:,None,None])) # nt x 3 x 3
    #     G_i = -A_i @ rt_hat # nt x 3 x 3
    #     a_i = np.squeeze(A_i @ np.cross(u,pbs)[...,None],axis=-1) # nt x 3
        
    #     a = np.sum(a_i,axis=0)
    #     b = np.sum(np.cross(rt,a_i),axis=0)
    #     A = np.sum(A_i,axis=0)
    #     G = np.sum(G_i,axis=0)
    #     H = np.sum(rt_hat@G_i,axis=0)
        
    #     K = np.vstack((np.hstack((H + Kbt, G.T)),
    #                     np.hstack((G, A + Kse))))

    #     mb = Kbt@(u - usr)
    #     nb = Kse@(v - vsr)
        
    #     rhs = -np.hstack((np.cross(u,mb) + np.cross(v,nb) + b,
    #                     np.cross(u,nb) + a + R.T@f))

    #     ps  = R@v
    #     hs  = 0.5*hat_for_h(u)@h # + 0.1*(1-h@h.T)*h
    #     us_vs = np.linalg.solve(K,rhs)
    #     return np.concatenate([ps,hs,us_vs])
    
    # def Cosserat_static_shooting(self, twist0):
    #     # Y = np.zeros((self.N,19))
    #     h0 = np.roll(Rotation.from_matrix(self.R0.reshape((3,3))).as_quat(),1)
    #     self.Y[0,0:13] = np.concatenate([self.p0,h0,twist0])

    #     # Y = euler_ivp(self.Cosserat_static_ODE, (0,self.L), y0, self.N)
    #     ds = self.ds
    #     for j in range(self.N-1):
    #         ys = self.Cosserat_static_ODE(self.s_step[j],self.Y[j])
    #         self.Y[j+1,0:13] = self.Y[j,0:13] + ds*ys
    #         self.Z[j] = np.concatenate([self.Y[j,7:19],ys[7:13]])
    #     # ys = self.Cosserat_static_ODE(self.s_step[-1],self.Y[-1])
    #     # self.Z[-1] = np.concatenate([self.Y[-1,7:19],ys[7:13]])
    #     # self.Y[0:13] = Y

    #     hL = self.Y[-1,3:7]
    #     RL = Rh(hL)
    #     uL = self.Y[-1,7:10]
    #     vL = self.Y[-1,10:13]
    #     mbL = self.Kbt@(uL - self.usr)
    #     nbL = self.Kse@(vL - self.vsr)

    #     pbs = np.cross(uL,self.rt) + vL # nt x 3
    #     Fb_i = -self.tau[:,None] * pbs / np.linalg.norm(pbs,ord=2,axis=-1)[:,None] # nt x 3
    #     res_n = np.sum(Fb_i,axis=0) + RL.T@self.FL - nbL
    #     res_m = np.sum(np.cross(self.rt,Fb_i),axis=0) + RL.T@self.ML - mbL

    #     return np.concatenate([res_m,res_n])
    
    # def Cosserat_dynamic_shooting(self, twist0):
    #     # Y = np.zeros((self.N,19))
    #     h0 = np.roll(Rotation.from_matrix(self.R0.reshape((3,3))).as_quat(),1)
    #     self.Y[0] = np.concatenate([self.p0,h0,twist0,np.zeros(6)])

    #     # Y = euler_ivp(self.Cosserat_dynamic_ODE, (0,self.L), y0, self.N)
    #     ds = self.ds
    #     for j in range(self.N-1):
    #         ys = self.Cosserat_dynamic_ODE(self.s_step[j],self.Y[j],self.Zh[j])
    #         self.Y[j+1] = self.Y[j] + ds*ys
    #         self.Z[j] = np.concatenate([self.Y[j,7:19],ys[7:13]])
    #     # self.Y = Y

    #     hL = self.Y[-1,3:7]
    #     RL = Rh(hL)
    #     uL = self.Y[-1,7:10]
    #     vL = self.Y[-1,10:13]
    #     uLt = self.c0*uL + self.Zh[-1,0:3]
    #     vLt = self.c0*vL + self.Zh[-1,3:6]
    #     mbL = self.Kbt@(uL - self.usr) + self.Bse@vLt
    #     nbL = self.Kse@(vL - self.vsr) + self.Bbt@uLt

    #     pbs = np.cross(uL,self.rt) + vL # nt x 3
    #     Fb_i = -self.tau[:,None] * pbs / np.linalg.norm(pbs,ord=2,axis=-1)[:,None] # nt x 3
    #     res_n = np.sum(Fb_i,axis=0) + RL.T@self.FL - nbL
    #     res_m = np.sum(np.cross(self.rt,Fb_i),axis=0) + RL.T@self.ML - mbL

    #     return np.concatenate([res_m,res_n])
    
    # def Cosserat_static_BVP(self, ig):
    #     t0 = time.time()
    #     sol = root(self.Cosserat_static_shooting,ig,method='lm')
    #     t1 = time.time()
    #     return sol.x, t1-t0
    
    # def Cosserat_dynamic_BVP(self, ig):
    #     t0 = time.time()
    #     sol = root(self.Cosserat_static_shooting,ig,method='lm')
    #     t1 = time.time()
    #     return sol.x, t1-t0
    
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
    delta = 0.015 # tendon offset
    MP = {
        "L": 0.2, # length
        "r": 0.001, # rod radius
        "E": 200e+9, # Young's modulus
        "nu": 0.25, # Poisson's ratio
        "rho": 1100, # density
        "Bbt": 0*np.eye(3), # damping 5e-7
        "Bse": 0*np.eye(3), # damping 5e-8
        "usr": np.zeros(3), # natural twist
        "vsr": np.array([0,0,1]), # natural twist
        "g": np.zeros(3), # gravitational acceleration
        "rt": np.array([[delta,0,0],
                    [0,delta,0],
                    [-delta,0,0],
                    [0,-delta,0]])
    }
    
    TDCR = continuumRobot_GVS(MP)

    q = np.concatenate([np.zeros(TDCR.nb*3),np.ones(TDCR.nb),np.zeros(TDCR.nb*2)])
    TDCR.set_q(q)
    t0 = time.time()
    TDCR.forward_kinematics()
    t1 = time.time()
    print(t1-t0)
    p = TDCR.g[:,0:3,3]
    print(TDCR.ng)
    eta = TDCR.J@q

    # num_steps = 40
    # # tendon release
    # input = np.zeros((num_steps+1,4))
    # input[0,0] = 20

    # traj = TDCR.Cosserat_dynamic_sim(0.002, num_steps, 0, input, np.array([0,0,0,0,0,1]))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # line = ax.plot(traj[1,:,0],traj[1,:,1],traj[1,:,2])
    line = ax.plot(p[:,0],p[:,1],p[:,2])
    ax.quiver(p[:,0],p[:,1],p[:,2], eta[:,0],eta[:,1],eta[:,2], normalize=False)
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