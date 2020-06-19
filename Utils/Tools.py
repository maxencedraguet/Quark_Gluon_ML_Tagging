#############################################################################
#
# Tools.py
#
# A set of helpful method to analyse and produce results.
#
# Author -- Maxence Draguet (09/06/2020)
#
#############################################################################
import os
import itertools
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix given in cm.
    Aims to mimic that produced by sklearn plot_confusion_matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cmap_min, cmap_max = im.cmap(0), im.cmap(256)

    fmt = '.2f' if normalize else 'd'
    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        
        ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           ylabel="True label",
           xlabel="Predicted label")

    ax.set_ylim((n_classes - 0.5, -0.5))
    return fig

def get_dictionary_cross_section():
    """
    Read and convert to dictionary of cross section
    """
    data = np.loadtxt("Backgrounds.txt", usecols=(0, 2, 3))
    cross_section_dictionary = dict()
    for row in data:
        cross_section_dictionary[str(int(row[0]))] = (row[1], row[2])
    
    return cross_section_dictionary

def write_ROC_info(filename, test, pred):
    """
    Writes in text file filename the test label, prediction proba to be used in ROC curves
    """
    with open(filename, 'w') as f:
        for test_item, pred_item in zip(test, pred):
            f.write("{}, {}\n".format(test_item, pred_item))

def ROC_curve_plotter_from_files(list_of_files, save_path):
    """
    Given a list of files.txt in list_of_files, loads them and plots the signal ROC curve.
    Entries of list_of_files should be tuples of the forme (file_name, signal_name)
    """
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for file_name, signal_name in list_of_files:
        truth_pred, proba_pred = np.loadtxt(file_name, delimiter=',', unpack=True)
        false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(truth_pred, proba_pred)
        AUC_test = metrics.auc(false_pos_rate, true_pos_rate)
        plt.plot(false_pos_rate, true_pos_rate, label='{} (area = {:.4f})'.format(signal_name, AUC_test))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curves')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.savefig(os.path.join(save_path, 'ROC_curve.png'), dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

def ROC_curve_plotter_from_values(list_of_signal, save_path):
    """
        Given a list of signals in list_of_signal, plots the ROC curve.
        Entries of list_of_files should be tuples of the forme (signal_name, truth_pred, proba_pred), the last two being numpy arrays
        """
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for signal_name, truth_pred, proba_pred in list_of_signal:
        #print(signal_name, truth_pred, proba_pred)
        false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(truth_pred, proba_pred)
        AUC_test = metrics.auc(false_pos_rate, true_pos_rate)
        plt.plot(false_pos_rate, true_pos_rate, label='{} (area = {:.4f})'.format(signal_name, AUC_test))
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curves')
    #plt.legend(loc='best')
    #fig.legend(loc=7)
    #fig.tight_layout()
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.subplots_adjust(right=0.7)
    plt.savefig(os.path.join(save_path, 'ROC_curve.png'), dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()
