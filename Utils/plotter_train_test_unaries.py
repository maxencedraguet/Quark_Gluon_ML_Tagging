import os
import sys
import json
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.markers as mmark


filename_gluon = "gluon_logger_info.txt"
filename_quark = "quark_logger_info.txt"

gsteps, gtrain_loss, gtest_loss = np.loadtxt(filename_gluon, delimiter=',', unpack=True, skiprows= 1)

plt.figure(figsize = (8, 8))
plt.plot(gsteps, gtrain_loss, 'b', label='Train Loss')
plt.plot(gsteps, gtest_loss, 'r', label='Test Loss')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Train and test loss for gluon model')
plt.legend(loc='upper right')
#plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.savefig('gluon_loss_curve.png', dpi=300, format='png', bbox_inches='tight')
plt.close()

qsteps, qtrain_loss, qtest_loss = np.loadtxt(filename_quark, delimiter=',', unpack=True, skiprows= 1)

plt.figure(figsize = (8, 8))
plt.plot(qsteps, qtrain_loss, 'b', label='Train Loss')
plt.plot(qsteps, qtest_loss, 'r', label='Test Loss')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Train and test loss for quark model')
plt.legend(loc='upper right')
#plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.savefig('quark_loss_curve.png', dpi=300, format='png', bbox_inches='tight')
plt.close()

