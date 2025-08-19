from simulator import Simulator
from operators import *

'''
Solves the equation u_t + cu_x = 0
'''
class OneWayWaveSimulator(Simulator):
    def define_space(self, a, b, N):
        super().define_space(a, b, N)
        self.ddx = Derivative(self.dx)

    def set_speed(self, c):
        #negative ensures correct direction
        self.speed = -c 

    def ddt(self, u, t):
        if not hasattr(self, 'speed'):
            self.speed = -1
        return self.speed * self.ddx.apply(u)
    
'''
Solves the equation u_t + 1/2(u^2)_x = 0
'''
class BurgerSimulator(Simulator):
    def define_space(self, a, b, N):
        super().define_space(a, b, N)
        self.ddx = Derivative(self.dx) 

    def ddt(self, u, t):
        return -0.5 * self.ddx.apply(u*u)
    
'''
Solves the equation u_t = H[u]
'''
class HilbertSimulator(Simulator):
    def define_space(self, a, b, N):
        super().define_space(a, b, N)
        self.hilbert = HilbertTransform(self.dx)

    def ddt(self, u, t):
        return self.hilbert.apply(u)

'''
Solves the equation iu_t = [-(1/2)d_xx + V(x)] u
'''
class SchrodingerSimulator(Simulator):
    def define_space(self, a, b, N):
        super().define_space(a, b, N)
        self.dxx = FilteredDerivative(self.dx,order = 2)

    def define_potential(self, fun):
        if not hasattr(self, 'x'):
            raise ValueError("Space not defined. Call define_space first.")
        self.potential : np.ndarray = fun(self.x)

    def ddt(self, u, t):
        if not hasattr(self, 'potential'):
            self.define_potential(lambda x: 0*x)
        return -1j * (-0.5 * self.dxx.apply(u) + self.potential * u)
    
'''
Solves the equation u_t = u_xx
'''
class HeatSimulator(Simulator):
    def define_space(self, a, b, N):
        super().define_space(a, b, N)
        self.dxx = Derivative(self.dx, order=2)

    def ddt(self, u, t):
        return self.dxx.apply(u)
    
'''
Solves the equation u_t = (Du_x)_x
'''
class DiffusionSimulator(Simulator):
    def define_space(self, a, b, N):
        super().define_space(a, b, N)
        self.ddx = FilteredDerivative(self.dx)

    def define_diffusion(self, fun):
        if not hasattr(self, 'x'):
            raise ValueError("Space not defined. Call define_space first.")
        self.diffusion : np.ndarray = fun(self.x)

    def ddt(self, u, t):
        if not hasattr(self, 'diffusion'):
            self.define_diffusion(lambda x: 1+0*x)
        u_x = self.ddx.apply(u)
        return self.ddx.apply(self.diffusion*u_x)
    
'''
Solves the equation u_t - 6uu_x + u_xxx = 0
'''
class KdVSimulator(Simulator):
    def define_space(self, a, b, N):
        super().define_space(a, b, N)
        self.ddx = Derivative(self.dx)
        self.dxxx = Derivative(self.dx,order=3)

    def ddt(self, u, t):
        u_xxx = self.dxxx.apply(u)
        uu_x = u * self.ddx.apply(u)
        return 6 * uu_x - u_xxx
    
'''
Solves the equation iu_t = -1/2u_xx + ku|u|^2
'''
class NLSSimulator(Simulator):
    def define_space(self, a, b, N):
        super().define_space(a, b, N)
        self.dxx = FilteredDerivative(self.dx,order = 2)

    def define_focusing(self, focusing):
        self.focusing = focusing

    def ddt(self, u, t):
        if not hasattr(self, 'focusing'):
            self.focusing = 0
        return -1j * (-0.5 * self.dxx.apply(u) + self.focusing * np.abs(u)**2 * u)