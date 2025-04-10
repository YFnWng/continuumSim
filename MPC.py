import numpy as np
# from casadi import *
from scipy.interpolate import BSpline
# from scipy.spatial.transform import Rotation
# from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from SE3 import *
from continuumRobot_GVS import continuumRobot_GVS
from contactHandling import *

class MPC:
    def __init__(self, TDCR, p_d, weights, h=6):
        self.TDCR = TDCR
        self.p_d = p_d
        self.Q = weights["Q"]
        self.R = weights["R"]
        self.S = weights["S"]

    def cost(self,u):
        err = self.p_d - self.TDCR.g[-1,0:3,3]
        return self.Q*np.sum(err**2,axis=-1) + self.R*np.sum(u**2,axis=-1)

    def eq_constraint(self,q):
        return np.sum((self.TDCR.Newmark_residual(q))**2)

    def ineq_constraint(self):
        pass

    def solve_MPC(self):
        ineq_cons = {'type': 'ineq',
                     'fun': lambda u: np.array([-q[0] - q_init[0] - eps, -q[1] - q_init[1] - eps,
                                                q[0] + q_init[0] + 0.4 - eps, q[1] + q_init[1] + 0.3,
                                                q_init[1] - q_init[0] - eps - q[0] + q[1]]),
                     'jac': lambda q: np.array([[-1.0, 0, 0, 0],
                                                [0, -1.0, 0, 0],
                                                [1.0, 0.0, 0, 0],
                                                [0.0, 1.0, 0, 0],
                                                [-1.0, 1.0, 0, 0]])}

        res = minimize(self.cost, q, method='SLSQP', jac=self.jac,
                       constraints=[ineq_cons], options={'ftol': 0.75e-3})

        # print(res.x)
        return res.x