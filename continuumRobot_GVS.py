import numpy as np
from casadi import *
from scipy.spatial.transform import Rotation
from scipy.optimize import root
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Utils.Math import *

class continuumRobot_GVS:
    def __init__(self, MP):
        # integration parameters
        self.N = 41
        self.L = MP["L"]
        self.s_step = np.linspace(0,self.L,self.N)
        self.ds = self.s_step[1] - self.s_step[0]
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

        self.J   = np.diag([I1,I2,I3])
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
        self.Kbtusr = self.Kbt@MP["usr"]
        self.Ksevsr = self.Kse@MP["vsr"]

        # g = np.array([0,0,-9.8])
        g = np.array([0,0,0])
        self.rhoAg = MP["rho"]*A*g
        self.rhoA  = MP["rho"]*A

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
    
    def BDF(self,alpha,dt):
        self.dt = dt
        self.c0 = (1.5 + alpha)/(self.dt*(1 + alpha))
        self.c1 = -2/self.dt
        self.c2 = (0.5 + alpha)/(self.dt*(1 + alpha))
        self.d1 = alpha/(1 + alpha)
    
    def forward_kinematics(self):

    # orientations integration with quaternion
    # Y: [p(0:3), h(3:7), u(7:10), v(10:13), q(13:16), w(16:19)], h being the quaternion
    def Cosserat_dynamic_ODE(self, s, y, zh):
        del s
        rt = self.rt
        rt_hat = self.rt_hat
        Bbt = self.Bbt
        Bse = self.Bse
        usr = self.usr
        vsr = self.vsr
        Kbt = self.Kbt
        Kse = self.Kse
        J     = self.J
        c0    = self.c0

        h = y[3:7]
        R = Rh(h)
        u = y[7:10]
        v = y[10:13]
        q = y[13:16]
        w = y[16:19]
        
        uh = zh[0:3]
        vh = zh[3:6]
        qh = zh[6:9]
        wh = zh[9:12]
        ush = zh[12:15]
        vsh = zh[15:18]

        fc = np.zeros(3) #fc[j] 
        f  = self.rhoAg + fc

        u_hat = hat(u)
        w_hat = hat(w)

        ut = c0*u + uh
        vt = c0*v + vh
        qt = c0*q + qh
        wt = c0*w + wh
        
        pbs = np.cross(u[None,:],rt) + v[None,:] # nt x 3
        pbs_norm = np.linalg.norm(pbs,ord=2,axis=-1) # nt
        pbs_n = pbs / pbs_norm[:,None]
        # print(pbs_norm)
        pbs_hat = hat(pbs_n) # nt x 3 x 3
        A_i = -pbs_hat @ pbs_hat * (self.tau[:,None,None] / (pbs_norm[:,None,None])) # nt x 3 x 3
        G_i = -A_i @ rt_hat # nt x 3 x 3
        a_i = np.squeeze(A_i @ np.cross(u,pbs)[...,None],axis=-1) # nt x 3
        
        a = np.sum(a_i,axis=0)
        b = np.sum(np.cross(rt,a_i),axis=0)
        A = np.sum(A_i,axis=0)
        G = np.sum(G_i,axis=0)
        H = np.sum(rt_hat@G_i,axis=0)
        
        K = np.vstack((np.hstack((H + self.Kbt_c0Bbt, G.T)),
                        np.hstack((G, A + self.Kse_c0Bse))))
        
        mb = Kbt@(u - usr) + Bbt@ut
        nb = Kse@(v - vsr) + Bse@vt
        
        rhs = np.hstack([-b + self.rho*(np.cross(w,J@w) + J@wt) - np.cross(v,nb) - np.cross(u,mb) - Bbt@ush,
                        -a + self.rhoA*(np.cross(w,q)+qt) - R.T@f - np.cross(u,nb) - Bse@vsh]) # + C*q*norm(q) drag
        
        ps  = R@v
        hs  = 0.5*hat_for_h(u)@h
        us_vs = np.linalg.solve(K,rhs)
        qs  = vt - u_hat@q + w_hat@v
        ws  = ut - u_hat@w

        return np.concatenate([ps,hs,us_vs,qs,ws])

    # Y : [p(0:3), h(3:12), u(12:15), v(15:18)]
    # orientations integration with rotation matrices (in our numerical applications the rod is straight in rest configuration)
    def Cosserat_static_ODE(self, s, y):
        del s
        f = self.rhoAg
        usr = self.usr
        vsr = self.vsr
        Kbt = self.Kbt
        Kse = self.Kse
        rt = self.rt
        rt_hat = self.rt_hat

        h = y[3:7]
        R = Rh(h)
        u = y[7:10]
        v = y[10:13]
        
        pbs = np.cross(u[None,:],rt) + v[None,:] # nt x 3
        pbs_norm = np.linalg.norm(pbs,ord=2,axis=-1) # nt
        pbs_n = pbs / pbs_norm[:,None]
        # print(pbs_norm)
        pbs_hat = hat(pbs_n) # nt x 3 x 3
        A_i = -pbs_hat @ pbs_hat * (self.tau[:,None,None] / (pbs_norm[:,None,None])) # nt x 3 x 3
        G_i = -A_i @ rt_hat # nt x 3 x 3
        a_i = np.squeeze(A_i @ np.cross(u,pbs)[...,None],axis=-1) # nt x 3
        
        a = np.sum(a_i,axis=0)
        b = np.sum(np.cross(rt,a_i),axis=0)
        A = np.sum(A_i,axis=0)
        G = np.sum(G_i,axis=0)
        H = np.sum(rt_hat@G_i,axis=0)
        
        K = np.vstack((np.hstack((H + Kbt, G.T)),
                        np.hstack((G, A + Kse))))

        mb = Kbt@(u - usr)
        nb = Kse@(v - vsr)
        
        rhs = -np.hstack((np.cross(u,mb) + np.cross(v,nb) + b,
                        np.cross(u,nb) + a + R.T@f))

        ps  = R@v
        hs  = 0.5*hat_for_h(u)@h # + 0.1*(1-h@h.T)*h
        us_vs = np.linalg.solve(K,rhs)
        return np.concatenate([ps,hs,us_vs])
    
    def Cosserat_static_shooting(self, twist0):
        # Y = np.zeros((self.N,19))
        h0 = np.roll(Rotation.from_matrix(self.R0.reshape((3,3))).as_quat(),1)
        self.Y[0,0:13] = np.concatenate([self.p0,h0,twist0])

        # Y = euler_ivp(self.Cosserat_static_ODE, (0,self.L), y0, self.N)
        ds = self.ds
        for j in range(self.N-1):
            ys = self.Cosserat_static_ODE(self.s_step[j],self.Y[j])
            self.Y[j+1,0:13] = self.Y[j,0:13] + ds*ys
            self.Z[j] = np.concatenate([self.Y[j,7:19],ys[7:13]])
        # ys = self.Cosserat_static_ODE(self.s_step[-1],self.Y[-1])
        # self.Z[-1] = np.concatenate([self.Y[-1,7:19],ys[7:13]])
        # self.Y[0:13] = Y

        hL = self.Y[-1,3:7]
        RL = Rh(hL)
        uL = self.Y[-1,7:10]
        vL = self.Y[-1,10:13]
        mbL = self.Kbt@(uL - self.usr)
        nbL = self.Kse@(vL - self.vsr)

        pbs = np.cross(uL,self.rt) + vL # nt x 3
        Fb_i = -self.tau[:,None] * pbs / np.linalg.norm(pbs,ord=2,axis=-1)[:,None] # nt x 3
        res_n = np.sum(Fb_i,axis=0) + RL.T@self.FL - nbL
        res_m = np.sum(np.cross(self.rt,Fb_i),axis=0) + RL.T@self.ML - mbL

        return np.concatenate([res_m,res_n])
    
    def Cosserat_dynamic_shooting(self, twist0):
        # Y = np.zeros((self.N,19))
        h0 = np.roll(Rotation.from_matrix(self.R0.reshape((3,3))).as_quat(),1)
        self.Y[0] = np.concatenate([self.p0,h0,twist0,np.zeros(6)])

        # Y = euler_ivp(self.Cosserat_dynamic_ODE, (0,self.L), y0, self.N)
        ds = self.ds
        for j in range(self.N-1):
            ys = self.Cosserat_dynamic_ODE(self.s_step[j],self.Y[j],self.Zh[j])
            self.Y[j+1] = self.Y[j] + ds*ys
            self.Z[j] = np.concatenate([self.Y[j,7:19],ys[7:13]])
        # self.Y = Y

        hL = self.Y[-1,3:7]
        RL = Rh(hL)
        uL = self.Y[-1,7:10]
        vL = self.Y[-1,10:13]
        uLt = self.c0*uL + self.Zh[-1,0:3]
        vLt = self.c0*vL + self.Zh[-1,3:6]
        mbL = self.Kbt@(uL - self.usr) + self.Bse@vLt
        nbL = self.Kse@(vL - self.vsr) + self.Bbt@uLt

        pbs = np.cross(uL,self.rt) + vL # nt x 3
        Fb_i = -self.tau[:,None] * pbs / np.linalg.norm(pbs,ord=2,axis=-1)[:,None] # nt x 3
        res_n = np.sum(Fb_i,axis=0) + RL.T@self.FL - nbL
        res_m = np.sum(np.cross(self.rt,Fb_i),axis=0) + RL.T@self.ML - mbL

        return np.concatenate([res_m,res_n])
    
    def Cosserat_static_BVP(self, ig):
        t0 = time.time()
        sol = root(self.Cosserat_static_shooting,ig,method='lm')
        t1 = time.time()
        return sol.x, t1-t0
    
    def Cosserat_dynamic_BVP(self, ig):
        t0 = time.time()
        sol = root(self.Cosserat_static_shooting,ig,method='lm')
        t1 = time.time()
        return sol.x, t1-t0
    
    def Cosserat_dynamic_sim(self, dt, num_steps, alpha, input, ig):
        self.BDF(alpha,dt)
        trajectory = np.zeros((num_steps+1,*np.shape(self.Y)))
        self.tau = input[0]

        t0 = time.time()
        sol = root(self.Cosserat_static_shooting,ig,method='lm')
        ig = sol.x
        t1 = time.time()
        print("---------------------------------------------------")
        print("time step:", -1)
        print("converged:", sol.success)
        print("max abs error:", np.max(np.abs(sol.fun)))
        print("xi (nfev):", sol.nfev)
        print("time used:",t1-t0)
        trajectory[0] = self.Y
        Z_prev = self.Z

        for i in range(num_steps):
            t0 = t1
            self.Zh = self.c1*self.Z + self.c2*Z_prev
            Z_prev = self.Z
            self.tau = input[i+1]
            print(ig)
            sol = root(self.Cosserat_dynamic_shooting,ig,method='lm')
            ig = sol.x
            t1 = time.time()
            print("---------------------------------------------------")
            print("time step:", i)
            print("converged:", sol.success)
            print("max abs error:", np.max(np.abs(sol.fun)))
            print("xi (nfev):", sol.nfev)
            print("time used:",t1-t0)

            trajectory[i+1] = self.Y

        # t0 = t1
        # self.Zh = self.c1*self.Z + self.c2*Z_prev
        # Z_prev = self.Z
        # self.tau = input[0]
        # # print(ig)
        # res = self.Cosserat_dynamic_shooting(ig)
        # # ig = sol.x
        # print(res)
        # t1 = time.time()

        # trajectory[0+1] = self.Y

        return trajectory


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

    num_steps = 40
    # tendon release
    input = np.zeros((num_steps+1,4))
    input[0,0] = 20

    traj = TDCR.Cosserat_dynamic_sim(0.002, num_steps, 0, input, np.array([0,0,0,0,0,1]))

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # line = ax.plot(traj[1,:,0],traj[1,:,1],traj[1,:,2])
    # ax.axis('equal')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # # plt.show()

    # def update(frame):
    #     # update the line plot:
    #     line[0].set_xdata(traj[frame,:,0])
    #     line[0].set_ydata(traj[frame,:,1])
    #     line[0].set_ydata(traj[frame,:,2])
    #     return line


    # ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=5)
    # plt.show()

    # print(traj[:,-1,0])
    fig,ax = plt.subplots()
    ax.plot(np.linspace(0,num_steps,num_steps+1),traj[:,-1,0])
    plt.show()


if __name__ == "__main__":
    main()