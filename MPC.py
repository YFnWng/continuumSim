import numpy as np
# from casadi import *
from scipy.interpolate import BSpline
# from scipy.spatial.transform import Rotation
# from scipy.linalg import solve
from scipy.optimize import minimize, Bounds
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from SE3 import *
from continuumRobot_GVS import continuumRobot_GVS
from contact import *

class Newmark_MPC:
    def __init__(self, TDCR, MP, weights, bounds, dt=5e-3, h=6):
        self.h = h # horizon
        self.plant = TDCR(MP, model="Kirchhoff", batch=h)
        self.Q = weights["Q"]
        self.R = weights["R"]
        self.S = weights["S"]
        xub = np.concatenate((bounds["qub"]*np.ones(self.plant.dof*self.h),
                              bounds["uub"]*np.ones(self.plant.udof*self.h)))
        xlb = np.concatenate((bounds["qlb"]*np.ones(self.plant.dof*self.h),
                              bounds["ulb"]*np.ones(self.plant.udof*self.h)))
        self.b = Bounds(lb = xlb, ub = xub)
        self.uid = self.plant.dof*self.h
        self.plant.Newmark_init(dt = dt)
        self.Newmark_a = -self.plant.Newmark[:,0] # 2
        C = self.plant.Newmark[:,1:] # 2 x 2
        self.Newmark_C = np.zeros((h,h+1,2,2)) # h x (h+1) x 2 x 2
        for i in range(h):
            self.Newmark_C[i,i+1] = np.eye(2)
            for j in range(i+1,0,-1):
                self.Newmark_C[i,j-1] = C@self.Newmark_C[i,j]


    def cost(self,x,qqq0,u0,p_d):
        q = np.reshape(x[:self.uid], (self.h,-1))
        u = np.reshape(x[self.uid:], (self.h,-1))
        du = u[1:] - u[:-1]
        dq = q[1:] - q[:-1] # (h-1) x dof
        q_ = np.vstack((qqq0[None,1:],
                        self.Newmark_a[None,:,None]*(q[0] - qqq0[0]),
                        self.Newmark_a[:,None]*dq[:,None,:])) # (h+1) x 2 x dof
        qd_qdd = np.sum(self.Newmark_C@q_, axis=1) # h x 2 x dof
        qdot = qd_qdd[:,0] # h x dof
        # qddot = qd_qdd[:,1]
        self.plant.forward_kinematics(q,qdot)
        err = p_d - self.plant.g[:,-1,0:3,3]
        return self.Q*np.sum(err**2) + self.R*(np.sum((u[0]-u0)**2)+np.sum(du**2))

    def dynamic_constraint(self,x,qqq0):
        q = np.reshape(x[:self.uid], (self.h,-1))
        u = np.reshape(x[self.uid:], (self.h,-1))
        dq = q[1:] - q[:-1] # (h-1) x dof
        q_ = np.vstack((qqq0[None,1:],
                        self.Newmark_a[None,:,None]*(q[0] - qqq0[0]),
                        self.Newmark_a[:,None]*dq[:,None,:])) # (h+1) x 2 x dof
        qd_qdd = np.sum(self.Newmark_C@q_, axis=1) # h x 2 x dof
        qdot = qd_qdd[:,0] # h x dof
        qddot = qd_qdd[:,1]
        return self.plant.Cosserat_dynamic_residual(q,qdot,qddot,u)

    def ineq_constraint(self):
        pass

    def solve_MPC(self,guess,qqq0,u0,p_d):
        eq_cons = {'type': 'eq',
                   'fun': lambda x: self.dynamic_constraint(x,qqq0)}

        res = minimize(lambda x: self.cost(x,qqq0,u0,p_d), x0=guess, method='SLSQP',
                       bounds = self.b, constraints=[eq_cons], options={'ftol': 0.75e-3})

        print(res.success)
        print(res.message)
        print("nfev: ",res.nfev)
        # print("max constraint violation: ", res.maxcv)
        return res.x
    
def main():
    delta = 0.015 # tendon offset 0.015
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
    weights = {
        "Q": 1.0e+3,
        "R": 0.0,
        "S": 1.0
    }
    bounds = {
        "qub": 20.0,
        "qlb": -20.0,
        "uub": 50.0,
        "ulb": -50.0
    }
    
    # TDCR = continuumRobot_GVS(MP, model="Kirchhoff", batch=1)
    horizon = 6
    dt = 15e-3
    MPC = Newmark_MPC(continuumRobot_GVS, MP, weights, bounds, dt=dt, h=horizon)

    # Initial conditions
    # q = np.concatenate([np.zeros(TDCR.nb*3),np.ones(TDCR.nb),np.zeros(TDCR.nb*2)])
    q0 = np.zeros((MPC.plant.nb*3))
    # kappa = 1.90985932
    # q = np.concatenate([np.zeros(TDCR.nb),-np.ones(TDCR.nb)*kappa,np.zeros(TDCR.nb)])
    # q = TDCR.static_solve(np.array([0,0,0,0]), q)
    # qdot = np.concatenate([np.zeros(TDCR.nb),np.ones(TDCR.nb)*10,np.zeros(TDCR.nb)])
    qdot0 = np.zeros_like(q0)
    # qdot = np.ones((TDCR.nb*3))*10
    qddot0 = np.zeros_like(q0)
    u0 = np.zeros(MPC.plant.udof)
    qqq0 = np.vstack((q0,qdot0,qddot0))

    guess = np.concatenate((np.tile(q0,horizon), np.tile(u0,horizon)))

    p_d = np.zeros((horizon,3))
    p_d[:,2] = 1.0

    t0 = time.time()
    x = MPC.solve_MPC(guess,qqq0,u0,p_d)
    t1 = time.time()
    print("MPC time: ",t1-t0)

    # MPC.cost(x,qqq0,u0,p_d)
    cv = MPC.dynamic_constraint(x,qqq0)
    print("maxcv: ",np.max(np.absolute(cv)))

    q = np.reshape(x[:MPC.uid], (MPC.h,-1))
    u = np.reshape(x[MPC.uid:], (MPC.h,-1))
    du = u[1:] - u[:-1]
    dq = q[1:] - q[:-1] # (h-1) x dof
    q_ = np.vstack((qqq0[None,1:],
                    MPC.Newmark_a[None,:,None]*(q[0] - qqq0[0]),
                    MPC.Newmark_a[:,None]*dq[:,None,:])) # (h+1) x 2 x dof
    qd_qdd = np.sum(MPC.Newmark_C@q_, axis=1) # h x 2 x dof
    # qdot = qd_qdd[:,0] # h x dof
    # qddot = qd_qdd[:,1]

    # verify Newmark transcription
    qqq_it = np.zeros((horizon+1,3,MPC.plant.dof))
    qqq_it[1:,0,:] = q
    qqq_it[0,:,:] = qqq0
    for i in range(horizon):
        hqdot = MPC.plant.Newmark[0]@qqq_it[i]
        hqddot = MPC.plant.Newmark[1]@qqq_it[i]
        qqq_it[i+1,1,:] = -MPC.plant.Newmark[0,0]*q[i] + hqdot
        qqq_it[i+1,2,:] = -MPC.plant.Newmark[1,0]*q[i] + hqddot
    print("max Newmark error: ",np.max(np.absolute(qd_qdd-qqq_it[1:,1:,:])))

    p_traj = MPC.plant.g[:,:,0:3,3]

    TDCR = continuumRobot_GVS(MP, model="Kirchhoff")
    x = TDCR.static_solve(np.array([-5,0]),q[-1])
    print(q[-1,TDCR.nb:2*TDCR.nb])
    print(x[TDCR.nb:2*TDCR.nb])
    TDCR.forward_kinematics(x,np.zeros_like(x))
    p_static = TDCR.g[0,:,0:3,3]

    fig,ax = plt.subplots()
    t = np.linspace(0,horizon,horizon+1)*dt
    ax.plot(t, np.insert(u[:,0],0,u0[0],axis=0))
    ax.plot(t, np.insert(u[:,1],0,u0[1],axis=0))
    plt.show()

    # fig,ax = plt.subplots(3)
    # xi_traj = np.squeeze(TDCR.Bg@q_traj[...,None,:,None],axis=-1)
    # qline = ax[0].plot(TDCR.sg,xi_traj[0,0,0,:,4])
    # ax[0].set_ylim((-20,20))
    # qdline = ax[1].plot(TDCR.sg,xi_traj[0,0,1,:,4])
    # ax[1].set_ylim((-50,50))
    # qddline = ax[2].plot(TDCR.sg,xi_traj[0,0,2,:,4])
    # ax[2].set_ylim((-100,100))
    # def update(frame):
    #     # update the line plot:
    #     qline[0].set_ydata(xi_traj[0,frame,0,:,4])
    #     qdline[0].set_ydata(xi_traj[0,frame,1,:,4])
    #     qddline[0].set_ydata(xi_traj[0,frame,2,:,4])
    #     return qline,qdline,qddline
    # ani = animation.FuncAnimation(fig=fig, func=update, frames=200, interval=5)
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(p_static[:,0],p_static[:,1],p_static[:,2])
    line = ax.plot(p_traj[0,:,0],p_traj[0,:,1],p_traj[0,:,2])
    points = ax.scatter(p_traj[0,:,0],p_traj[0,:,1],p_traj[0,:,2], s=2, c='y')
    ax.plot_surface(cylinder_x,cylinder_y,cylinder_z, alpha=0.5, color='r')
    # forces = [ax.quiver(p_traj[0,1:,0],p_traj[0,1:,1],p_traj[0,1:,2],
    #            fc_traj[0,:,0],fc_traj[0,:,1],fc_traj[0,:,2], normalize=False)]
    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_zlim([0,0.2])
    ax.set_aspect('equal')

    def update(frame):
        # update the line plot:
        line[0].set_data_3d(p_traj[frame,:,0],p_traj[frame,:,1],p_traj[frame,:,2])
        points.set_offsets(p_traj[frame,:,0:2]) # x,y
        points.set_3d_properties(p_traj[frame,:,2],zdir='z') # z

        # forces[0].remove()
        # forces[0] = ax.quiver(p_traj[frame,1:,0],p_traj[frame,1:,1],p_traj[frame,1:,2],
        #        fc_traj[frame,:,0],fc_traj[frame,:,1],fc_traj[frame,:,2], normalize=False)
        # ax.set_aspect('equal')
        return line, points#, forces[0]


    ani = animation.FuncAnimation(fig=fig, func=update, frames=horizon, interval=5)
    plt.show()
    # writer = animation.PillowWriter(fps=200,
    #                                 metadata=dict(artist='Me'),
    #                                 bitrate=1800)
    # ani.save(filename="/continuumSim/figures/rod_slipping.gif", writer=writer)

if __name__ == "__main__":
    main()