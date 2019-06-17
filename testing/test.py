from invisible_cities.io.mcinfo_io import load_mchits
from invisible_cities.reco.paolina_functions import voxelize_hits
import numpy as np
import matplotlib.pyplot as plt


Events = load_mchits("../data/Tl208-0000-ACTIVE.h5")

print(len(Events[0]))


#print(Events[0])
aa = voxelize_hits(Events[0], [5,5,5])
print(aa)
