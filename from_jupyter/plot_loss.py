import cebra
from cebra import CEBRA
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection

fig = plt.figure(figsize=(5,5))
ax = plt.subplot(111)
ax.plot(cebra_loc_model.state_dict_['loss'], c='deepskyblue', label = 'position+direction')
