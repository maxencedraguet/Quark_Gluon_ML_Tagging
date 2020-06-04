import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix given in cm.
    Aims to mimic that produced by sklearn plot_confusion_matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cmap_min, cmap_max = im.cmap(0), im.cmap(256)

    plt.title(title)
    plt.colorbar()
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color =  = cmap_max if cm[i, j] < thresh else cmap_min
        
        ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.colorbar(self.im_, ax=ax)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=display_labels,
           yticklabels=display_labels,
           ylabel="True label",
           xlabel="Predicted label")

           ax.set_ylim((n_classes - 0.5, -0.5))
           plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

def write_ROC_info(filename, test, pred):
    """
    Writes in text file filename the test label, prediction proba to be used in ROC curves
    """
    with open(filename, 'w') as f:
        for test_item, pred_item in zip(test, pred):
            f.write("{}, {}\n".format(test_item, pred_item))

