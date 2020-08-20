import os
import sys
import json
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.markers as mmark

"""
filename_gluon = "gluon_model/logger_info.txt"
filename_quark = "quark_model/logger_info.txt"
"""
save_path = "learning_curves_binary/"
os.makedirs(save_path, exist_ok=True)

steps, train_loss, test_loss, train_acc, test_acc, test_auc = np.loadtxt("logger_info.txt", delimiter=',', unpack=True, skiprows= 1)

plt.figure(figsize = (8, 8))
plt.plot(steps, train_loss, 'b', label='Train Loss')
plt.plot(steps, test_loss, 'r', label='Test Loss')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Train and test loss for gluon model')
plt.legend(loc='upper right')
#plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.savefig(save_path + 'loss_curve.png', dpi=300, format='png', bbox_inches='tight')
plt.close()

plt.figure(figsize = (8, 8))
plt.plot(steps, train_acc, 'b', label='Train accuracy')
plt.plot(steps, test_acc, 'r', label='Validation accuracy')
plt.plot(steps, test_auc, 'k', label='Validation AUC')

plt.xlabel('Steps')
plt.ylabel('Accuracy/AUC')
plt.title('Train and test accuracy for quark model')
plt.legend(loc='upper right')
#plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.savefig(save_path + 'acc_curve.png', dpi=300, format='png', bbox_inches='tight')
plt.close()
