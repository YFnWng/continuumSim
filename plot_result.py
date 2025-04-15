import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from SE3 import *
from continuumRobot_GVS import continuumRobot_GVS
from contact import *

def main():
    num_step = 60
    dt = 15e-3
    t = np.linspace(0,num_step,num_step+1)*dt
    p_d = np.zeros((num_step,3))
    # p_d[:,2] = 1.0
    p_d[:,2] = np.concatenate((np.linspace(1.0,0.92,25),np.ones(35)*0.92))
    p_d[25:,0] = np.linspace(0.0,0.15,35)

    h6 = np.load('data/h6.npz')
    h6p_traj = h6['p_traj']
    h6v_traj = h6['v_traj']
    h6pos_err = h6['pos_err']
    h1 = np.load('data/h1.npz')
    h1p_traj = h1['p_traj']
    h1v_traj = h1['v_traj']
    h1pos_err = h1['pos_err']

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=16, azim=-103, roll=-94)
    for frame in range(0,num_step,5):
        # ax.plot(p_static[:,0],p_static[:,1],p_static[:,2])
        # line = ax.plot(h6p_traj[frame,:,0],h6p_traj[frame,:,1],h6p_traj[frame,:,2],c='b')
        # points = ax.scatter(h6p_traj[frame,:,0],h6p_traj[frame,:,1],h6p_traj[frame,:,2], s=2, c='y')
        line = ax.plot(h1p_traj[frame,:,0],h1p_traj[frame,:,1],h1p_traj[frame,:,2],c='b')
        points = ax.scatter(h1p_traj[frame,:,0],h1p_traj[frame,:,1],h1p_traj[frame,:,2], s=2, c='y')
    ax.plot_surface(cylinder_x,cylinder_y,cylinder_z, alpha=0.5, color='r')
    ax.scatter(p_d[:,0],p_d[:,1],p_d[:,2], s=2, c='k')
    # forces = [ax.quiver(p_traj[0,1:,0],p_traj[0,1:,1],p_traj[0,1:,2],
    #            fc_traj[0,:,0],fc_traj[0,:,1],fc_traj[0,:,2], normalize=False)]
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_zlim3d([-0.2,0.6])
    ax.set_ylim3d([-0.2,0.2])
    ax.set_zlim3d([0,1])
    ax.axis('equal')
    
    # def set_axes_equal(ax):
    #     '''Make axes of 3D plot have equal scale.'''
    #     x_limits = ax.get_xlim3d()
    #     y_limits = ax.get_ylim3d()
    #     z_limits = ax.get_zlim3d()

    #     x_range = abs(x_limits[1] - x_limits[0])
    #     x_middle = np.mean(x_limits)
    #     y_range = abs(y_limits[1] - y_limits[0])
    #     y_middle = np.mean(y_limits)
    #     z_range = abs(z_limits[1] - z_limits[0])
    #     z_middle = np.mean(z_limits)

    #     max_range = max(x_range, y_range, z_range)

    #     ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    #     ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    #     ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

    # set_axes_equal(ax)
    plt.show()

    

    fig,ax = plt.subplots()
    ax.plot(t,h1v_traj)
    ax.plot(t,h6v_traj)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('v [m/s]')
    ax.legend(['horizon = 1', 'horizon = 6'])
    plt.show()

    fig,ax = plt.subplots()
    ax.plot(t[1:],h1pos_err)
    ax.plot(t[1:],h6pos_err)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('position error [m]')
    ax.legend(['horizon = 1', 'horizon = 6'])
    plt.show()

if __name__ == "__main__":
    main()