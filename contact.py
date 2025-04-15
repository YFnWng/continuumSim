import numpy as np

def Ycylinder_SDF(p,center,radius):
    # a cylinder parallel to Y-axis
    vc = p - center
    vc[...,1] = 0
    dc = np.linalg.norm(vc, ord=2, axis=-1, keepdims=True)
    d = dc - radius
    n = np.divide(vc, dc, out=np.zeros_like(vc), where=dc>0)
    return d, n

def contact_force(g,bpt,ds_end,k,alpha,mu,sigma,SDF):
    d,n = SDF(g[...,0:3,3])
    # no_contact = d >= 0
    shallow_contact = np.logical_and(d < 0, d >= -alpha)
    deep_contact = d < -alpha
    lambda_n = np.zeros_like(d)
    # fc = np.zeros_like(n)
    lambda_n[shallow_contact] = k*d[shallow_contact]**2/(2*alpha) # N/m
    lambda_n[deep_contact] = k*(-d[deep_contact] - alpha/2)
    fc = lambda_n*n

    bn = np.swapaxes(g[...,0:3,0:3],-1,-2)@n[...,None]
    vt = bpt - np.sum(bpt[...,None]*bn, axis=-2) # tangent velocity in material frame
    vtnorm = np.linalg.norm(vt, ord=2, axis=-1, keepdims=False)
    slip = vtnorm >= sigma
    t = vt[slip,:]/vtnorm[slip,None]
    fc[slip] -= mu*lambda_n[slip]*t
    FL = fc[...,-1,:]*ds_end

    return FL,fc[...,:-1,:]

def data_for_Ycylinder(center_x,center_z,radius,height):
    y = np.linspace(-height, height, 3)
    theta = np.linspace(0, 2*np.pi, 18)
    theta_grid, y_grid=np.meshgrid(theta, y)
    x_grid = radius*np.cos(theta_grid) + center_x
    z_grid = radius*np.sin(theta_grid) + center_z
    return x_grid,y_grid,z_grid

# environment setup
cylinder_c = [0.05,0,0.92]
cylinder_r = 0.05
SDF = lambda p : Ycylinder_SDF(p,center=cylinder_c,radius=cylinder_r)
contactForce = lambda g, bpt, ds: contact_force(g,bpt,ds,k=5000,alpha=5e-6,mu=0.1,sigma=0.001,SDF=SDF)
cylinder_x,cylinder_y,cylinder_z = data_for_Ycylinder(cylinder_c[0],cylinder_c[2],radius=cylinder_r,height=0.1)