import numpy as np
import matplotlib.pyplot as plt
from models import *

plt.style.use('dark_background')


sim = SchrodingerSimulator()
sim.define_space(-np.pi,np.pi,128)
sim.define_time(0,2*np.pi,128)
#sim.define_diffusion(lambda x: np.cos(x)**2)
sim.define_potential(lambda x: np.where(np.abs(x) <= 1, -1.0, 0.0))
sim.define_initial_condition(lambda x: np.exp(-x**2))
sim.simulate()