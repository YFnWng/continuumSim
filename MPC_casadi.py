import numpy as np
import casadi as ca
from scipy.optimize import minimize, Bounds
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from SE3_casadi import *
from continuumRobot_GVS_casadi import continuumRobot_GVS
from contact_casadi import *

class Newmark_MPC:
    def __init__(self, TDCR, MP, weights, bounds, dt=5e-3, h=6):
        self.h = h # horizon
        self.plant = TDCR(MP, dt=dt, model="Kirchhoff")
        self.Q = weights["Q"]
        self.R = weights["R"]
        self.S = weights["S"]
        self.xub = np.concatenate((bounds["qub"]*np.ones(self.plant.dof*self.h),
                              bounds["uub"]*np.ones(self.plant.udof*self.h)))
        self.xlb = np.concatenate((bounds["qlb"]*np.ones(self.plant.dof*self.h),
                              bounds["ulb"]*np.ones(self.plant.udof*self.h)))
        self.glb = np.zeros(self.plant.dof*self.h)
        self.gub = np.zeros_like(self.glb)
        self.uid = self.plant.dof*self.h

        # decision variables
        q = ca.SX.sym('q',self.plant.dof,self.h)
        u = ca.SX.sym('q',self.plant.udof,self.h)
        x = ca.vertcat(ca.reshape(q,self.plant.dof*self.h,1),
                       ca.reshape(u,self.plant.udof*self.h,1))
        
        # parameters
        qqq0 = ca.SX.sym('qqq0',self.plant.dof,3)
        u0 = ca.SX.sym('u0',self.plant.udof)
        p_d = ca.SX.sym('p_d',3,self.h)
        parameters = ca.vertcat(ca.reshape(qqq0,self.plant.dof*3,1),
                                u0,
                                ca.reshape(p_d,self.h*3,1))

        # self.Newmark_a = -self.plant.Newmark[:,0] # 2
        # C = self.plant.Newmark[:,1:] # 2 x 2
        # self.Newmark_C = np.zeros((h,h+1,2,2)) # h x (h+1) x 2 x 2
        # for i in range(h):
        #     self.Newmark_C[i,i+1] = np.eye(2)
        #     for j in range(i+1,0,-1):
        #         self.Newmark_C[i,j-1] = C@self.Newmark_C[i,j]

        # define casadi functions
        qdot, qddot = self.Newmark_rollout(q,qqq0)
        self.transcript = ca.Function('transcript',[q,qqq0],[qdot,qddot],
                                      ['q','qqq0'],['qdot','qddot'])
        
        q_vec = ca.SX.sym('q_vec',self.plant.dof,1)
        qdot_vec = ca.SX.sym('qdot_vec',self.plant.dof,1)
        qddot_vec = ca.SX.sym('qddot_vec',self.plant.dof,1)
        u_vec = ca.SX.sym('u_vec',self.plant.udof,1)
        res_vec, g_vec, M_vec = self.plant.Cosserat_dynamic_residual(q_vec, qdot_vec, qddot_vec, u_vec)
        self.dynamic_residual_map = ca.Function('dynamic_residual',
                                            [q_vec, qdot_vec, qddot_vec, u_vec],
                                            [res_vec, g_vec, M_vec]).map(self.h) #,"thread",self.h
        # self.dynamic_residual_map = self.dynamic_residual.map(self.h,"thread",self.h)
        res, g, _ = self.dynamic_residual_map(q, qdot, qddot, u)

        # self.dynamic_constraint = ca.Function('dynamic_constraint',[x,parameters],[res],['x','p'],['res'])

        p = g[-4:-1,:] # tip positions
        # self.cost = ca.Function('vanilla_cost',[x,parameters],[self.vanilla_cost(u0,u,p,p_d)],['x','p'],['cost'])
        cost = self.vanilla_cost(u0,u,p,p_d)

        # Create an NLP solver
        prob = {
            'f': cost, 
            'x': x, 
            'g': ca.reshape(res,self.plant.dof*self.h,1), 
            'p': parameters
            }
        self.solver = ca.nlpsol('solver', 'ipopt', prob)

    def Newmark_rollout(self,q,qqq0):
        qqq = qqq0
        qdot = ca.SX(self.plant.dof,self.h)
        qddot = ca.SX(self.plant.dof,self.h)
        for i in range(self.h):
            qdot[:,i] = -self.plant.Newmark[0,0]*q[:,i] + qqq@self.plant.Newmark[0].T
            qddot[:,i] = -self.plant.Newmark[1,0]*q[:,i] + qqq@self.plant.Newmark[1].T
            qqq = ca.horzcat(q[:,i],qdot[:,i],qddot[:,i])
        return qdot, qddot

    def vanilla_cost(self,u0,u,p,p_d):
        du = u[:,1:] - u[:,:-1]
        err = p_d - p
        return self.Q*ca.sum(err**2) + self.R*(ca.sum((u[:,0]-u0)**2)+ca.sum(du**2))

    def solve_MPC(self,x0,qqq0,u0,p_d):
        parameters = np.concatenate((qqq0.flatten(),
                                    u0,
                                    p_d.flatten()))
        
        sol = self.solver(x0=x0, p=parameters, lbx=self.xlb, ubx=self.xub, lbg=self.glb, ubg=self.gub)
        x = sol['x']
        q = ca.reshape(x[:self.uid],self.plant.dof,self.h)
        u = ca.reshape(x[self.uid:],self.plant.udof,self.h)

        # print(res.success)
        # print(res.message)
        # print("nfev: ",res.nfev)
        # print("max constraint violation: ", res.maxcv)
        return q,u
    
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
        "R": 0.5,
        "S": 1.0
    }
    bounds = {
        "qub": 20.0,
        "qlb": -20.0,
        "uub": 50.0,
        "ulb": -50.0
    }
    
    # TDCR = continuumRobot_GVS(MP, model="Kirchhoff", batch=1)
    horizon = 1
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

    # t0 = time.time()
    q,u = MPC.solve_MPC(guess,qqq0,u0,p_d)
    # t1 = time.time()
    # print("MPC time: ",t1-t0)

    fig,ax = plt.subplots()
    t = np.linspace(0,horizon,horizon+1)*dt
    ax.plot(t, np.insert(u[:,0],0,u0[0],axis=0))
    ax.plot(t, np.insert(u[:,1],0,u0[1],axis=0))
    plt.show()

    

    p_traj = MPC.plant.g[:,:,0:3,3]

    TDCR = continuumRobot_GVS(MP, model="Kirchhoff")
    x = TDCR.static_solve(np.array([-5,0]),q[-1])
    print(q[-1,TDCR.nb:2*TDCR.nb])
    print(x[TDCR.nb:2*TDCR.nb])
    TDCR.forward_kinematics(x,np.zeros_like(x))
    p_static = TDCR.g[0,:,0:3,3]

    

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