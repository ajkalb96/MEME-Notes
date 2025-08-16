from simulator import WaveSimulator
from operators import *

'''
Solves the equation u_t + cu_x = 0
'''
class OneWayWaveSimulation(WaveSimulator):
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
Solves the equation u_t + uu_x = 0
'''
class BurgerSimulation(WaveSimulator):
    def define_space(self, a, b, N):
        super().define_space(a, b, N)
        self.ddx = Derivative(self.dx) 

    def ddt(self, u, t):
        return -u * self.ddx.apply(u)
    
'''
Solves the equation u_t = H[u]
'''
class HilbertSimulation(WaveSimulator):
    def define_space(self, a, b, N):
        super().define_space(a, b, N)
        self.hilbert = HilbertTransform(self.dx)

    def ddt(self, u, t):
        return self.hilbert.apply(u)

'''
Solves the equation iu_t = [-(1/2)d_xx + V(x)] u
'''
class SchrodingerSimulation(WaveSimulator):
    def define_space(self, a, b, N):
        super().define_space(a, b, N)
        self.dxx = Derivative(self.dx,order = 2)

    def define_potential(self, fun):
        if not hasattr(self, 'x'):
            raise ValueError("Space not defined. Call define_space first.")
        self.potential : np.ndarray = fun(self.x)

    def ddt(self, u, t):
        if not hasattr(self, 'potential'):
            self.define_potential(lambda x: 0*x)
        return -1j * (-0.5 * self.dxx.apply(u) + self.potential * u)