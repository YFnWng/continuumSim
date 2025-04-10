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