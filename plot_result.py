import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from SE3 import *
from continuumRobot_GVS import continuumRobot_GVS
from contact import *

def main():
    num_step = 150 # 120
    dt = 15e-3
    t = np.linspace(0,num_step,num_step+1)*dt
    p_d = np.zeros((num_step,3))
    # p_d[:,2] = 1.0
    # p_d[:,2] = np.concatenate((np.linspace(1.0,0.92,25),np.ones(35)*0.92))
    # p_d[25:,0] = np.linspace(0.0,0.15,35)

    theta = np.linspace(0,np.pi,num_step+1)[1:]
    p_d[:,0] = 0.21*np.cos(theta)
    p_d[:,1] = 0.21*np.sin(theta)
    p_d[:,2] = 0.97*np.ones(num_step)

    # h6 = np.load('data/soft_h6_noreg.npz')
    # h6p_traj = h6['p_traj']
    # h6v_traj = h6['v_traj']
    # h6u = h6['u']
    # h6pos_err = h6['pos_err']
    # h6reg = np.load('data/soft_h6_reg.npz')
    # h6regp_traj = h6reg['p_traj']
    # h6regv_traj = h6reg['v_traj']
    # h6regu = h6reg['u']
    # h6regpos_err = h6reg['pos_err']
    # h1 = np.load('data/soft_h1_noreg.npz')
    # h1p_traj = h1['p_traj']
    # h1v_traj = h1['v_traj']
    # h1u = h1['u']
    # h1pos_err = h1['pos_err']
    # h1reg = np.load('data/soft_h1_reg.npz')
    # h1regp_traj = h1reg['p_traj']
    # h1regv_traj = h1reg['v_traj']
    # h1regu = h1reg['u']
    # h1regpos_err = h1reg['pos_err']
    # na = np.load('data/no_control_soft.npz')
    # na_traj = na['p_traj'][0,1:]

    h6 = np.load('data/stiff_h6_noreg.npz')
    h6p_traj = h6['p_traj']
    h6v_traj = h6['v_traj']
    h6u = h6['u']
    h6pos_err = h6['pos_err']
    h6reg = np.load('data/stiff_h6_reg.npz')
    h6regp_traj = h6reg['p_traj']
    h6regv_traj = h6reg['v_traj']
    h6regu = h6reg['u']
    h6regpos_err = h6reg['pos_err']
    h1 = np.load('data/stiff_h1_noreg.npz')
    h1p_traj = h1['p_traj']
    h1v_traj = h1['v_traj']
    h1u = h1['u']
    h1pos_err = h1['pos_err']
    h1reg = np.load('data/stiff_h1_reg.npz')
    h1regp_traj = h1reg['p_traj']
    h1regv_traj = h1reg['v_traj']
    h1regu = h1reg['u']
    h1regpos_err = h1reg['pos_err']
    na = np.load('data/stiff_h1_free.npz')
    na_traj = na['p_traj']#[0,1:]

    def plot_traj(p_traj,filename):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=39, azim=-117, roll=-107) #elev=16, azim=-103, roll=-94, elev=39, azim=-117, roll=-107
        num_step = np.shape(p_traj)[0]-1
        colors = cm.Blues(range(256))
        colors = colors[range(80,256,int((256-80)/num_step)),:]
        for frame in range(0,num_step+1,5):
            # ax.plot(p_static[:,0],p_static[:,1],p_static[:,2])
            # line = ax.plot(h6p_traj[frame,:,0],h6p_traj[frame,:,1],h6p_traj[frame,:,2],c='b')
            # points = ax.scatter(h6p_traj[frame,:,0],h6p_traj[frame,:,1],h6p_traj[frame,:,2], s=2, c='y')
            line = ax.plot(p_traj[frame,:,0],p_traj[frame,:,1],p_traj[frame,:,2],c=colors[frame])
            # points = ax.scatter(p_traj[frame,:,0],p_traj[frame,:,1],p_traj[frame,:,2], s=2, c='y')
        # ax.plot(p_traj[:frame+1,-1,0],p_traj[:frame+1,-1,1],p_traj[:frame+1,-1,2],c='g') # tip path
        ax.plot(p_traj[:,-1,0],p_traj[:,-1,1],p_traj[:,-1,2],c='g') # tip path
        ax.plot_surface(cylinder_x,cylinder_y,cylinder_z, alpha=0.5, color='r')
        ax.scatter(p_d[:,0],p_d[:,1],p_d[:,2], s=2, c='k')
        # forces = [ax.quiver(p_traj[0,1:,0],p_traj[0,1:,1],p_traj[0,1:,2],
        #            fc_traj[0,:,0],fc_traj[0,:,1],fc_traj[0,:,2], normalize=False)]
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.set_xlim3d([-0.1,1.0])
        ax.set_ylim3d([-0.2,0.2])
        ax.set_zlim3d([0,1])
        ax.axis('equal')
        plt.savefig('figures/'+filename, dpi=600, format='png')

    # plot_traj(na_traj,'stiff_free.png')
    # plot_traj(h1p_traj,'stiff_h1_noreg.png')
    # plot_traj(h1regp_traj,'stiff_h1_reg.png')
    # plot_traj(h6p_traj,'stiff_h6_noreg.png')
    # plot_traj(h6regp_traj,'stiff_h6_reg.png')
    # plt.show()

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig,ax = plt.subplots(figsize=(900*px, 300*px), tight_layout=True)
    ax.plot(t,h1u[:,0])
    ax.plot(t,h1regu[:,0])
    ax.plot(t,h6u[:,0])
    ax.plot(t,h6regu[:,0])
    ax.set_xlabel('t [s]')
    ax.set_ylabel('u_1 [N]')
    ax.legend(['horizon = 1, no regularization', 'horizon = 1, regularization', 'horizon = 6, no regularization', 'horizon = 6, regularization'])
    ax.grid()
    plt.savefig('figures/case2_u.png', dpi=600, format='png')

    fig,ax = plt.subplots(figsize=(900*px, 300*px), tight_layout=True)
    ax.plot(t,h1v_traj)
    ax.plot(t,h1regv_traj)
    ax.plot(t,h6v_traj)
    ax.plot(t,h6regv_traj)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('v [m/s]')
    ax.legend(['horizon = 1, no regularization', 'horizon = 1, regularization', 'horizon = 6, no regularization', 'horizon = 6, regularization'])
    ax.grid()
    plt.savefig('figures/case2_v.png', dpi=600, format='png')

    fig,ax = plt.subplots(figsize=(900*px, 300*px), tight_layout=True)
    ax.plot(t[1:],h1pos_err)
    ax.plot(t[1:],h1regpos_err)
    ax.plot(t[1:],h6pos_err)
    ax.plot(t[1:],h6regpos_err)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('position error [m]')
    ax.legend(['horizon = 1, no regularization','horizon = 1, regularization', 'horizon = 6, no regularization', 'horizon = 6, regularization'])
    ax.grid()
    plt.savefig('figures/case2_e.png', dpi=600, format='png')
    plt.show()

if __name__ == "__main__":
    main()