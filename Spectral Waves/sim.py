import numpy as np
import matplotlib.pyplot as plt
from models import *

plt.style.use('dark_background')


sim = KdVSimulator()
sim.define_space(-np.pi,np.pi,128)
sim.define_time(0,2*np.pi,128)
sim.define_initial_condition(lambda x: np.exp(-x**2))
sim.simulate()
