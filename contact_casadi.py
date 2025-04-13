import numpy as np
import casadi as ca

def Ycylinder_SDF(p,center,radius):
    # a cylinder parallel to Y-axis
    vc = p - center # 3
    vc[1] = 0
    dc = ca.norm_2(vc)
    d = dc - radius
    n = vc / (dc+1e-6)
    return d, n

def contact_force(g,bpt,ds_end,k,alpha,mu,sigma,SDF):
    fc = []
    for i in range(len(g)):
        d,n = SDF(g[i][0:3,3])
        lambda_n = ca.if_else(d >= -alpha,
                   k*ca.fmin(d,0)**2/(2*alpha),
                   k*(-d - alpha/2))
        fc.append(lambda_n*n)

        bn = g[i][0:3,0:3].T@n
        vt = bpt[i][0:3] - ca.dot(bpt[i][0:3],bn) # tangent velocity in material frame
        vtnorm = ca.norm_2(vt)
        fc[i] -= ca.if_else(vtnorm >= sigma,
                              mu*lambda_n*vt/vtnorm,
                              0)
    
    FL = fc[-1]*ds_end

    return FL,fc[:-1]

def data_for_Ycylinder(center_x,center_z,radius,height):
    y = np.linspace(-height, height, 3)
    theta = np.linspace(0, 2*np.pi, 18)
    theta_grid, y_grid=np.meshgrid(theta, y)
    x_grid = radius*np.cos(theta_grid) + center_x
    z_grid = radius*np.sin(theta_grid) + center_z
    return x_grid,y_grid,z_grid

# environment setup
cylinder_c = [0.05,0,0.97]
cylinder_r = 0.05
SDF = lambda p : Ycylinder_SDF(p,center=cylinder_c,radius=cylinder_r)
contactForce = lambda g, bpt, ds: contact_force(g,bpt,ds,k=0,alpha=5e-6,mu=0.1,sigma=0.001,SDF=SDF)
cylinder_x,cylinder_y,cylinder_z = data_for_Ycylinder(cylinder_c[0],cylinder_c[2],radius=cylinder_r,height=0.1)