import yaml
import math
import torch
import torchmin
from torchmin import minimize
import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
import scipy.io as sio
import time

def expSE3(Psi,device):
    # exponential map se3 to SE3, refer to Muller's review
    # Psi: N x 6 np array
    N = Psi.shape[0]
    # print(Psi._version)
    z11 = torch.zeros((N,1,1), device=device) # N x 1 x 1
    Psiw = Psi[:,0:3]#.clone() # N x 3
    Psiv = Psi[:,3:6]#.clone() # N x 3
    # print(Psiv._version)
    # theta = torch.sqrt(torch.sum(Psiw**2, axis=1))[:,None] # N x 1
    theta = torch.linalg.vector_norm(Psiw, dim=1, keepdim=True) # N x 1
    # print(theta)
    zeroidx = torch.flatten(theta != 0)
    n = torch.zeros_like(Psiw, device=device)
    n[zeroidx,:] = torch.divide(Psiw[zeroidx,:], theta[zeroidx,:]) # N x 3

    n_hat = torch.cat((torch.cat((z11, -n[:,None,None,2], n[:,None,None,1]),2),
                    torch.cat((n[:,None,None,2], z11, -n[:,None,None,0]),2),
                    torch.cat((-n[:,None,None,1], n[:,None,None,0], z11),2)),1) # N x 3 x 3
    n_hat2 = torch.matmul(n_hat, n_hat) # N x 3 x 3
    alpha = torch.zeros_like(theta, device=device)
    alpha[zeroidx,:] = torch.divide(torch.sin(theta[zeroidx,:]), theta[zeroidx,:]) # N x 1
    beta = torch.zeros_like(theta, device=device)
    beta[zeroidx,:] = torch.divide(1-torch.cos(theta[zeroidx,:]), theta[zeroidx,:]**2) # N x 1
    expPsiw = torch.eye(3,device=device)[None,:,:] + alpha[:,:,None]*n_hat*theta[:,:,None] + beta[:,:,None]*n_hat2*(theta**2)[:,:,None] # (2.6) N x 3 x 3
    dexpPsiw = torch.eye(3,device=device)[None,:,:] + beta[:,:,None]*n_hat*theta[:,:,None] + (1-alpha)[:,:,None]*n_hat2 # (2.13) N x 3 x 3
    expPsiv = torch.matmul(dexpPsiw, Psiv[:,:,None]) # (2.27) N x 3 x 1
    return torch.cat((torch.cat((expPsiw, expPsiv),2), torch.cat((torch.zeros((N,1,3),device=device), torch.ones((N,1,1),device=device)),2)),1) # N x 4 x 4

class splineCurve3D:
    def __init__(self, t, k, L, sites, device):
        # c: n x batch_size x 3
        # super().__init__(t, c, k, axis=0)
        self.device = device
        self.t = t
        self.c = None
        self.k = k
        self.L = L
        # self.n, self.N, _ = c.shape
        self.n = len(t) - k - 1
        self.N = 0
        self.s = sites # collocation sites
        self.h = sites[1:] - sites[:-1]
        self.tq = torch.tensor([1/2-torch.sqrt(torch.tensor(15))/10, 1/2, 1/2+torch.sqrt(torch.tensor(15))/10]) # Legendre zeros as quadrature points
        q = torch.reshape(sites[:-1,None] + self.h[:,None]*self.tq[None,:], (3*len(self.h),))
        V = torch.fliplr(torch.vander(self.tq-1/2)) # 3 x 3
        self.invV = torch.linalg.inv(V).to(self.device)
        self.hd = torch.tensor(self.h,device=device)
        self.T = None
        self.Tq = None
        # self.T = torch.zeros((self.N,len(self.s),4,4), device=self.device)#.to(self.device)
        # self.T[:,0,:,:] = torch.eye(4,device=self.device)[None,:,:]
        # self.dT = []

        # collocation
        self.S = torch.zeros((self.n,len(sites)), device=device)
        self.Q = torch.zeros((self.n,(len(sites)-1)*3), device=device)
        # self.D = torch.zeros_like(self.S,device=device)
        for i in range(self.n):
            b = BSpline.basis_element(t[i:i+k+2])
            # db = b.derivative(nu=1) # does not work due to duplicated knots
            active_s = torch.logical_and(sites>=t[i], sites<=t[i+k+1])
            active_q = torch.logical_and(q>=t[i], q<=t[i+k+1])
            self.S[i,active_s] = torch.from_numpy(b(sites[active_s])).float().to(self.device)
            self.Q[i,active_q] = torch.from_numpy(b(q[active_q])).float().to(self.device)
            # self.D[i,active_s] = torch.from_numpy(db(sites[active_s])).float().to(self.device)
        self.S[-1,-1] = 1
        # self.uc = torch.permute(c,(1,2,0))@self.S[None,:,:] # N x 3 x len(sites)
        D_mat = sio.loadmat('D.mat')
        # print(D_mat['D'].dtype)
        self.D = torch.from_numpy(D_mat['D']).float().to(device)
        self.uc = None
        self.t = self.t.to(device)

    def Magnus_expansion(self,Xq,h):
        # Xq: N x 3 x 6
        z1 = torch.zeros((self.N,3,1,1),device=self.device)
        z3 = torch.zeros((self.N,3,3,3),device=self.device)
        B = self.invV[None,:,:]@Xq*h # N x 3 x 6 self-adjoint basis
        w_hat = torch.cat((torch.cat((z1, -B[:,:,None,None,2], B[:,:,None,None,1]), 3),
            torch.cat((B[:,:,None,None,2], z1, -B[:,:,None,None,0]), 3),
            torch.cat((-B[:,:,None,None,1], B[:,:,None,None,0], z1), 3)), 2) # N x v x 3 x 3
        v_hat = torch.cat((torch.cat((z1, -B[:,:,None,None,5], B[:,:,None,None,4]), 3),
            torch.cat((B[:,:,None,None,5], z1, -B[:,:,None,None,3]), 3),
            torch.cat((-B[:,:,None,None,4], B[:,:,None,None,3], z1), 3)), 2) # N x v x 3 x 3
        adB = torch.cat((torch.cat((w_hat, z3),3), torch.cat((v_hat, w_hat),3)), 2) # N x v x 6 x 6

        B12 = adB[:,0,:,:]@B[:,1,:,None] # N x 6 x 1
        B23 = adB[:,1,:,:]@B[:,2,:,None]
        B13 = adB[:,0,:,:]@B[:,2,:,None]
        B113 = adB[:,0,:,:]@B13
        B212 = adB[:,1,:,:]@B12
        B112 = adB[:,0,:,:]@B12
        B1112 = adB[:,0,:,:]@B112

        Psi = B[:,0,:] + B[:,2,:]/12 + torch.squeeze(B12/12 - B23/240 + B113/360 - B212/240 - B1112/720) # N x 6
        
        return Psi

    def integrate_SE3(self,c,quadrature_pose=False):
        self.T = torch.tile(torch.eye(4,device=self.device),(self.N,len(self.s),1,1))
        h = self.h*self.L
        uq = c @ self.Q[None,:,:] # N x 3 x vnk, strain at quadrature points
        # Xq = torch.zeros((self.N,3,6,len(self.s)-1), device=self.device)
        # Psi = torch.zeros((self.N,6,len(self.s)-1), device=self.device)
        # expPsit = torch.zeros((self.N,4,4,len(self.s)-1), device=self.device)
        
        # zz = np.zeros((self.n,self.N,3,1,1))
        if quadrature_pose:
            self.Tq = torch.zeros((self.N,3*(len(self.s)-1),4,4),device=self.device)
            v = torch.cat((torch.zeros((self.N,2),device=self.device),torch.ones((self.N,1),device=self.device)),dim=1)

        for k in range(len(self.s)-1): # forward kinematics propagation, calculate SE3 transformation and derivatives
            # snapshots at quadrature point
            Xq = torch.cat((torch.permute(uq[:,:,k*3:k*3+3], (0,2,1)), torch.tile(torch.tensor([0, 0, 1],device=self.device),(self.N,3,1))), 2) # N x v x 6

            # Calculate Magnus expansion
            Psi = self.Magnus_expansion(Xq,h[k]) # N x 6

            # exponential map se3 to SE3
            # expPsit[...,k] = expSE3(Psi[...,k],self.device)

            # Integrate on SE3
            # a = self.T[:,k,:,:].clone()
            # b = expPsit[...,k].clone()
            self.T[:,k+1,:,:] = torch.matmul(self.T[:,k,:,:].clone(),expSE3(Psi.clone(),self.device)) #torch.rand((4,4),device=self.device)[None,:,:]
            # self.T = torch.cat((self.T,(self.T[:,k,:,:]@expPsit[...,k])[:,None,:,:]),dim=1)

            if quadrature_pose:
                # Psiq1 = (Xq(1,:)' + [Uc(:,k);0;0;1])/2*GV.hsp(k)*(1/2-sqrt(15)/10);
                # Psiq2 = (Xq(2,:)' + Xq(1,:)')/2*GV.hsp(k)*sqrt(15)/10;
                # Psiq3 = (Xq(3,:)' + [Uc(:,k+1);0;0;1])/2*GV.hsp(k)*(1/2-sqrt(15)/10);
                Psiq = torch.cat(((Xq[:,0,:] + torch.cat((self.uc[:,:,k],v),dim=1))/2*self.tq[0]*h[k], # N x 6
                                (Xq[:,0,:] + Xq[:,1,:])/2*(1/2-self.tq[0])*h[k],
                                -(Xq[:,2,:] + torch.cat((self.uc[:,:,k+1],v),dim=1))/2*self.tq[0]*h[k]), dim=0)
                expPsiq = expSE3(Psiq,self.device)
                self.Tq[:,k*3,:,:] = self.T[:,k,:,:].clone()@expPsiq[:self.N,:,:]
                self.Tq[:,k*3+1,:,:] = self.Tq[:,k*3-2,:,:].clone()@expPsiq[self.N:2*self.N,:,:]
                self.Tq[:,k*3+2,:,:] = self.T[:,k+1,:,:].clone()@expPsiq[2*self.N:,:,:]

    def get_position(self, config=None, sites=np.linspace(0,1,49)):
        # config: N x (nx2), no torsion for now
        if config is None and self.c is None:
            print('Coefficients not given.')
            return
        if config is not None:
            self.N = config.shape[0]
            self.c = torch.cat((torch.permute(torch.reshape(config, (self.N, self.n, 2)),(0,2,1)),torch.zeros((self.N,1,self.n),device=self.device)), dim=1) # N x 3 x n
            self.uc = self.c @ self.S[None,:,:] # N x 3 x len(sites)
            # self.T = None
            self.integrate_SE3(self.c)
        # if self.T is None:
        #     self.integrate_SE3()
        p = torch.zeros((self.N, len(sites), 3), device=self.device)
        seg = torch.cat((torch.tensor([0]), self.s[:-1] + self.h/2)) # segment the length such that collocation points are centered in intervals
        for k in range(len(sites)):
            idx = torch.nonzero(sites[k]>=seg)[-1][0] # index of reference collocation point
            ds = (sites[k] - self.s[idx])*self.L
            Psii = ds*torch.hstack((self.uc[:,:,idx], torch.zeros((self.N,2),device=self.device), torch.ones((self.N,1),device=self.device))) # N x 6
            # print('get_position')
            # expPsi = expSE3(Psi,self.device) # N x 4 x 4
            # print(Psi)
            # Tk = self.T[:,idx,:,:]@expSE3(Psi,self.device) # N x 4 x 4
            p[:,k,:]  = (self.T[:,idx,:,:]@expSE3(Psii,self.device))[:,0:3,3]
            # p[:,k,:]  = (self.T[:,idx,:,:])[:,0:3,3]
        return p
    
    def get_basis_idx(self, site_idx, sites=torch.linspace(0,1,49)):
        sites = sites.to(self.device)
        s = sites[site_idx]
        k_idx = torch.logical_and(torch.gt(s[:,None],self.t[None,:-1]),torch.le(s[:,None],self.t[None,1:])).nonzero()
        idx = torch.zeros(len(s),dtype=torch.long,device=self.device)
        idx[k_idx[:,0]] = k_idx[:,1]
        return idx

def zero_force(p):
    # p: ... x 3
    return 0*p

class TDCR(splineCurve3D):
    def __init__(self, spline_config, robot_config, device):
        # spline properties
        n = spline_config['n']
        k = spline_config['k']
        breaks = torch.linspace(0, 1, k+n+1-2*k)
        t = torch.cat((torch.zeros(k), breaks, torch.ones(k)))# knots
        sites = torch.cat((torch.tensor([0]), breaks[:-1] + (breaks[1:] - breaks[:-1])/2, torch.tensor([1])))
        super().__init__(t, k, spline_config['L'], sites, device)
        

        # robot properties
        PI = torch.tensor(math.pi,device=device)
        self.n_tendons = robot_config['num_tendons']
        beta = torch.arange(self.n_tendons,device=device)*2*PI/self.n_tendons
        self.r = torch.stack((torch.cos(beta),torch.sin(beta),torch.zeros_like(beta,device=device)),dim=1)*robot_config['tendon_offset'] # num x 3
        zz = torch.zeros((self.n_tendons,1,1),device=device)
        self.r_hat = torch.cat((torch.cat((zz,-self.r[:,2,None,None],self.r[:,1,None,None]),dim=2),
                                torch.cat((self.r[:,2,None,None],zz,-self.r[:,0,None,None]),dim=2),
                                torch.cat((-self.r[:,1,None,None],self.r[:,0,None,None],zz),dim=2)),dim=1)
        self.qub = robot_config['q_ub']
        self.qlb = robot_config['q_lb']
        # area = PI*(robot_config['r']**2)
        I = PI*(robot_config['r']**4)/4
        J = 2*I
        # GV.Kse = diag([shear_modulus*area, shear_modulus*area, E*area])
        self.Kbt = torch.diag(torch.tensor([robot_config['E']*I, robot_config['E']*I, robot_config['shear_modulus']*J],device=device))

        # variables
        self.tau = None


    def solve_Cosserat_model(self,q,init_guess=None, fe=zero_force):
        # Implementation based on cubic B-spline
        # https://www.math.ucdavis.edu/~bremer/classes/fall2018/MAT128a/lecture15.pdf
        # https://pytorch-minimize.readthedocs.io/en/latest/api/generated/torchmin.minimize.html
        # Nov 2024, Yifan Wang
        # Kirchhoff tendon-driven robot
        # q: N x num_tendons/2
        self.fe = fe
        self.N = q.shape[0]
        # self.tau = torch.zeros((self.N,self.n_tendons/2))
        self.tau = torch.cat((q,-q),dim=1) # push-pull
        # % hsp0 = GV.hsp; D0 = GV.D;
        # GV.hsp = GV.hsp*qa(3); % 1xn
        # GV.D = GV.D/qa(3);
        # GV.dUdotdc = GV.dUdotdc/qa(3);
        if init_guess is None:
            init_guess = torch.zeros((self.N,3,self.n),requires_grad=True, device=self.device)

        res = torchmin.bfgs._minimize_lbfgs(
            self.collocation_error, 
            init_guess, 
            line_search='strong-wolfe',
            # max_iter=1,
            xtol = 1e-6,
            disp=1
            )

        # res = minimize(
        #     self.collocation_error, 
        #     init_guess, 
        #     method='l-bfgs', 
        #     options=dict(line_search='strong-wolfe'),
        #     # max_iter=1000,
        #     # gtol = 1e-5,
        #     disp=2
        #     )
        self.c = res.x

    def collocation_error(self,c):
        #  c: N x 3 x n spline coefficients stacked
        self.uc = c @ self.S[None,:,:] # N x 3 x len(sites)

        # forward kinematics
        self.integrate_SE3(c,quadrature_pose=True)

        # Setup tendon linear system
        gx = torch.zeros((self.N,3,len(self.s)),device=self.device)
        
        # quadrature for integration of distributed external force
        fe = self.fe(self.Tq[:,:,0:3,3]) # N x 3(n-1) x 3
        # print('fe',fe)
        # print('self.Tq[:,:,0:3,3]',self.Tq[:,:,0:3,3])
        # print('self.T',self.T[:,:,0:3,3])
        intfe = torch.cat((self.L*self.hd[None,:,None]*(5*(fe[:,0::3,:] + fe[:,2::3,:]) + 8*fe[:,1::3,:])/18,\
                            torch.zeros((self.N,1,3),device=self.device)),dim=1)
        Intfe = torch.cumsum(intfe.flip(1),dim=1)

        # u = self.uc[...,i] # N x 3 x len(sites)
        
        # a = torch.zeros((self.N,len(self.s),3),device=self.device)
        # b = torch.zeros_like(a,device=self.device)
        # A = torch.zeros((self.N,len(self.s),3,3),device=self.device)
        # #G = zeros(3,3)
        # H = torch.zeros_like(A,device=self.device)
        # nbt = torch.zeros_like(a,device=self.device)
        e3 = torch.tensor([0.0,0.0,1.0],device=self.device).tile(self.N,1) # Nx3
        I3 = torch.eye(3,device=self.device)
        meL = torch.zeros(self.N,3,device=self.device)

        # zz = torch.zeros((self.N,1,1,len(self.s)))
        # u_hat = torch.cat((torch.cat((zz,-self.uc[:,2,None,:],self.uc[:,1,None,:]),dim=2),
        #                    torch.cat((self.uc[:,2,None,:],zz,-self.uc[:,0,None,:]),dim=2),
        #                    torch.cat((-self.uc[:,1,None,:],self.uc[:,0,None,:],zz),dim=2)),dim=1)

        for i in range(len(self.s)-1,-1,-1): # for each collocation point, backward
            
            # if i == len(self.s):
            #     meL = R'@Me
            a = torch.zeros((self.N,3),device=self.device)
            b = torch.zeros_like(a,device=self.device)
            A = torch.zeros((self.N,3,3),device=self.device)
            #G = zeros(3,3)
            H = torch.zeros_like(A,device=self.device)
            nbt = torch.zeros_like(a,device=self.device)

            for j in range(self.n_tendons): # these are all "local" variables
                pb_si = torch.cross(self.uc[:,:,i],self.r[j,:].expand(self.N,3),dim=1) + e3 # N x 3
                pb_s_norm = torch.linalg.vector_norm(pb_si,dim=1)
                Fb_j = -self.tau[:,j,None]*pb_si/pb_s_norm[:,None] # N x 3
                nbt -= Fb_j

                A_j = -(pb_si[:,:,None]@pb_si[:,None,:] - torch.square(pb_s_norm[:,None,None])*I3[None,...]) *\
                      (self.tau[:,j]/torch.pow(pb_s_norm,3))[:,None,None] # N x 3 x 3
                G_j = -A_j @ self.r_hat[j,:,:]
                a_j = torch.squeeze(A_j @ torch.cross(self.uc[:,:,i],pb_si,dim=1)[...,None],dim=-1)

                a = a + a_j
                b = b + torch.cross(self.r[None,j,:], a_j, dim=1)
                A = A + A_j
                #G = G + G_j;
                H = H + self.r_hat[j,:,:]@G_j

                if i == len(self.s)-1: # boundary condition
                    meL += torch.cross(self.r[None,j,:], Fb_j, dim=1)

            K = H + self.Kbt[None,:,:] # Nx3x3
            nb = -nbt + torch.squeeze(self.T[:,i,0:3,0:3].transpose(-2,-1)@Intfe[:,len(self.s)-i-1,:,None],dim=-1) # Nx3 not local
            mb = torch.squeeze(self.Kbt[None,:,:]@self.uc[:,:,i,None],dim=-1) # Nx3

            # Calculate ODE terms
            gx[:,:,i] = -torch.linalg.solve(K, torch.squeeze(torch.cross(self.uc[:,:,i],mb,dim=1) + torch.cross(e3,nb,dim=1) + b)) # Nx3

        # lambda = [mb;nb]
        
        # Assemble collocation and boundary errors
        bL = torch.linalg.solve(self.Kbt[None,:,:],meL[...,None]) # N x 3

        col_err = torch.cat((torch.div(c@self.D[None,:,:],self.L), self.uc[:,:,-1,None]), dim=2) - torch.cat((gx, bL), dim=2) # N x 3 x len(s)+1
        t1 = time.time()

        return torch.sum(torch.square(col_err)) # careful here


    # def collocation_error(self, c):
    #     # Compute u_c for all collocation points
    #     self.uc = c @ self.S[None, :, :]  # Shape: (N, 3, len(sites))

    #     # Forward kinematics
    #     self.integrate_SE3(c, quadrature_pose=True)

    #     # Compute distributed external forces
    #     fe = self.fe(self.Tq[:, :, 0:3, 3])  # Shape: (N, 3(n-1), 3)
    #     intfe = torch.cat((self.L * self.hd[None, :, None] *
    #                     (5 * (fe[:, 0::3, :] + fe[:, 2::3, :]) + 8 * fe[:, 1::3, :]) / 18,
    #                     torch.zeros((self.N, 1, 3), device=self.device)), dim=1)
    #     Intfe = torch.cumsum(intfe.flip(1), dim=1)  # Shape: (N, len(s), 3)

    #     # Precompute constants
    #     e3 = torch.tensor([0.0, 0.0, 1.0], device=self.device).tile(self.N, 1)  # Shape: (N, 3)
    #     I3 = torch.eye(3, device=self.device).unsqueeze(0)  # Shape: (1, 3, 3)

    #     # Compute per-tendon contributions in a vectorized manner
    #     pb_si = torch.cross(self.uc[:, :, :, None], self.r.T[None, :, :, None], dim=2) + e3[:, :, None, None]  # Shape: (N, 3, len(s), n_tendons)
    #     pb_s_norm = torch.linalg.vector_norm(pb_si, dim=1, keepdim=True)  # Shape: (N, 1, len(s), n_tendons)

    #     Fb_j = -self.tau[:, :, None] * pb_si / pb_s_norm  # Shape: (N, 3, len(s), n_tendons)
    #     A_j = -(pb_si.unsqueeze(-1) @ pb_si.unsqueeze(-2) -
    #             torch.square(pb_s_norm).unsqueeze(-1) * I3) * (self.tau / pb_s_norm ** 3).unsqueeze(-1)  # Shape: (N, len(s), n_tendons, 3, 3)

    #     a_j = (A_j @ torch.cross(self.uc[:, :, :, None], pb_si, dim=1).unsqueeze(-1)).squeeze(-1)  # Shape: (N, 3, len(s), n_tendons)

    #     # Aggregate all tendon contributions
    #     a = torch.sum(a_j, dim=-1)  # Shape: (N, 3, len(s))
    #     b = torch.sum(torch.cross(self.r[None, :, :, None], a_j, dim=1), dim=-1)  # Shape: (N, 3, len(s))
    #     A = torch.sum(A_j, dim=-1)  # Shape: (N, len(s), 3, 3)

    #     # Boundary conditions
    #     meL = torch.sum(torch.cross(self.r[None, :, :], Fb_j[:, :, -1, :], dim=1), dim=-1)  # Shape: (N, 3)
    #     bL = torch.linalg.solve(self.Kbt[None, :, :], meL.unsqueeze(-1)).squeeze(-1)  # Shape: (N, 3)

    #     # Compute K, nb, and gx in a batch
    #     H = torch.sum(self.r_hat @ (-A_j @ self.r_hat.T.unsqueeze(0)), dim=-1)  # Shape: (N, len(s), 3, 3)
    #     K = H + self.Kbt[None, :, :].unsqueeze(1)  # Shape: (N, len(s), 3, 3)

    #     nb = -torch.sum(Fb_j, dim=-1) + torch.cumsum(Intfe.flip(1), dim=1).flip(1)  # Shape: (N, 3, len(s))
    #     mb = torch.matmul(self.Kbt[None, :, :], self.uc).permute(0, 2, 1)  # Shape: (N, 3, len(s))

    #     gx = torch.linalg.solve(K, -(torch.cross(self.uc, mb, dim=1) + torch.cross(e3[:, None, :], nb, dim=1) + b))  # Shape: (N, 3, len(s))

    #     # Collocation and boundary errors
    #     col_err = torch.cat((torch.div(c @ self.D[None, :, :], self.L), self.uc[:, :, -1, None]), dim=2) - \
    #             torch.cat((gx, bL[:, :, None]), dim=2)  # Shape: (N, 3, len(s) + 1)

    #     return torch.sum(torch.square(col_err))



def main():
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    # print(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)
    robot = TDCR(config['spline'], config['robot'], device)
    q = torch.tensor([[40,0]],device=device)
    # q = torch.tensor([[20,10],[-10,10], [0,20],[3,5], [-8,2]],device=device)
    # init_guess1 = torch.cat((torch.zeros((1,1,12),device=device),7.64*torch.ones((1,1,12),device=device),torch.zeros((1,1,12),device=device)),dim=1)
    # init_guess2 = torch.cat((-7.64*torch.ones((1,1,12),device=device),torch.zeros((1,1,12),device=device),torch.zeros((1,1,12),device=device)),dim=1)
    # init_guess = torch.cat((init_guess1,init_guess2),dim=0)
    # robot.solve_Cosserat_model(q,init_guess=init_guess)

    robot.solve_Cosserat_model(q)
    print(robot.c)

#     k = 3 # degree
#     n = 12 # number of control points
#     breaks = torch.linspace(0, 1, k+n+1-2*k)
#     t = torch.cat((torch.zeros(k), breaks, torch.ones(k)))# knots
#     c1 = torch.vstack((torch.ones(n)*20, torch.zeros(n), torch.zeros(n)))
#     c2 = torch.vstack((torch.zeros(n), torch.ones(n)*20, torch.zeros(n)))
#     c = torch.stack((c1.T,c2.T), axis=1) # n x N x 3
#     L = 0.02
#     sites = torch.cat((torch.tensor([0]), breaks[:-1] + (breaks[1:] - breaks[:-1])/2, torch.tensor([1])))
#     spl = splineCurve3D(t, c, k, L, sites)
#     spl.integrate_SE3()
#     p = spl.get_position(sites=sites) # N x n x 3
#     # print(spl([0.2, 0.3])) # N x 3 x n_query
#     # print(p)
#     # print(spl.Q)
#     # print(spl.uc)
#     # print(spl.T[0,-1,:,:])
#     # print(spl.T[1,-1,:,:])

#     # b0 = BSpline.basis_element(t[0:0+k+2])
#     # bn = BSpline.basis_element(t[n-1:n-1+k+2])
#     # x0 = np.linspace(t[0], t[0+k+1], 100)
#     # xn = np.linspace(t[n-1], t[n-1+k+1], 100)
#     # fig, ax = plt.subplots()
#     # ax.plot(x0, b0(x0), 'g', lw=3)
#     # ax.plot(xn, bn(xn), 'r', lw=3)
#     # ax.grid(True)
#     # plt.show()

    p = robot.get_position().detach().cpu() # N x n x 3
    ax = plt.figure().add_subplot(projection='3d')
    for i in range(p.shape[0]):
        ax.plot(p[i,:,0].numpy(), p[i,:,1].numpy(), p[i,:,2].numpy())
    # i=1
    # ax.plot(p[i,:,0].numpy(), p[i,:,1].numpy(), p[i,:,2].numpy())
    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

if __name__ == "__main__":
    main()