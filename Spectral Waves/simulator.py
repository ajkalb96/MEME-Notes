import numpy as np
import scipy
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Simulator(ABC):

    def define_space(self,a,b,N):
        self.x = np.linspace(a, b, N, endpoint=False)
        self.dx = self.x[1] - self.x[0]

    def define_time(self,a,b,N):
        self.t = np.linspace(a, b, N)
        self.dt = self.t[1] - self.t[0]

    def define_initial_condition(self,f):
        if not hasattr(self,'x'):
            raise ValueError("Space not defined.")
        self.initial_condition : np.ndarray = np.asarray(f(self.x), dtype=complex)
    
    @abstractmethod
    def ddt(self, u: np.ndarray, t) -> np.ndarray:
        pass

    def simulate(self):
        if not hasattr(self, 'x'):
            raise ValueError("Space not defined. Call define_space first.")
        if not hasattr(self, 't'):
            raise ValueError("Time not defined. Call define_time first.")
        if not hasattr(self, 'initial_condition'):
            raise ValueError("Initial condition not defined. Call define initial_condition first.")
        
        y0_real = np.concatenate([self.initial_condition.real, self.initial_condition.imag])

        def rhs(t, y):
            n = y.size // 2
            u = y[:n] + 1j * y[n:]
            du_dt = self.ddt(u, t)
            return np.concatenate([du_dt.real, du_dt.imag])
        
        sol = scipy.integrate.solve_ivp(rhs, (self.t[0], self.t[-1]), y0_real)
        
        # Interpolate the solution to match self.t
        n = self.initial_condition.size
        interp = lambda x: scipy.interpolate.interp1d(
            sol.t, x, 
            axis=-1, kind='cubic',
            bounds_error=False, 
            fill_value="extrapolate"
            )(self.t)
        data = interp(sol.y[:n]) + 1j * interp(sol.y[n:])
        data = data.T


        for i, ti in enumerate(self.t):
            plt.figure(figsize=(8, 4))
            u = data[i]
            plt.plot(self.x, u.real, label="Real", color='red')
            plt.plot(self.x, u.imag, label="Imag", color='blue')
            plt.legend()

            plt.title(f"t = {ti:.2f}")
            plt.xlabel("x")
            plt.ylabel("u(x, t)")
            plt.ylim(np.min(self.initial_condition)-1, np.max(self.initial_condition)+1)
            plt.tight_layout()
            plt.savefig(f"video_tmp/file_{i:03d}.png")
            plt.close()
        self.generate_video()
        
    def generate_video(self):
        import os, subprocess, glob,uuid
        if not os.path.exists("video_tmp"):
            os.makedirs("video_tmp")
        self.filename = str(uuid.uuid4())
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', 'video_tmp/file_%03d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            f'{self.filename}.mp4'
        ])
        os.chdir("video_tmp")
        for file_name in glob.glob("*.png"):
            os.remove(file_name)
        os.chdir("..")

