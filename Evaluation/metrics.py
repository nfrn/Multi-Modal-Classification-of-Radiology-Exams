from labelbasedclassification import accuracyMacro, accuracyMicro, precisionMacro, precisionMicro, recallMacro, recallMicro, fbetaMacro, fbetaMicro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import metrics
import matplotlib.colors as colors
import math
THRESHOLD= 0.5
OPENI=True

if __name__ == '__main__':
    txts = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']


    filename = "ResNet_GrayScale_Predictions.npy"
    result = np.load(filename,allow_pickle=True)
    predictions = result[0]
    trueLabels = result[1]

    print(np.shape(predictions))
    print(np.shape(trueLabels))

    fpr, tpr, _ = metrics.roc_curve(trueLabels.ravel(), predictions.ravel())
    roc_auc = metrics.auc(fpr, tpr)
    print('\r MICRO val_roc_auc: %s' % (str(round(roc_auc, 4))), end=100 * ' ' + '\n')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(14):
        fpr[i], tpr[i], _ = metrics.roc_curve(trueLabels[:, i], predictions[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        if(math.isnan(roc_auc[i])):
            roc_auc[i]=0
        print(roc_auc[i])



    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(14)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    macro = sum(roc_auc.values())/13
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(14):
        fpr[i], tpr[i], _ = metrics.roc_curve(trueLabels[:, i], predictions[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(14)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(14):
        if(i==1 and OPENI):
            continue
        # print(interp(all_fpr, fpr[i], tpr[i]))
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    if OPENI:
        mean_tpr /= 13
    else:
        mean_tpr /= 14

    n_classes = 14

    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(trueLabels.ravel(), predictions.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average {0:0.3f}'
                   ''.format(roc_auc["micro"]),
             color='black', linestyle=':', linewidth=3)

    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average {0:0.3f}'
                   ''.format(roc_auc["macro"]),
             color='black', linestyle='-', linewidth=3)

    colors_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8',
                   '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#bcf60c', '#fabebe', '#008080', '#e6beff',
                   '#9a6324', '#fffac8', '#800000', '#aaffc3',
                   '#808000', '#ffd8b1', '#000075', '#808080',
                   '#ffffff', '#000000']
    print(colors_list)
    for i, color in zip(range(n_classes), colors_list):
        txt = txts[i]
        if not (math.isnan(roc_auc[i])):
            plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label='{0} {1:0.2f}'
                       ''.format(txt, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.00])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    print('\r MACRO val_roc_auc: %s' % (str(round(macro, 4))), end=100 * ' ' + '\n')

    value = metrics.coverage_error(trueLabels,predictions)
    print('\r MACRO CE: %s' % (str(round(value, 4))), end=100 * ' ' + '\n')
    value = metrics.label_ranking_average_precision_score(trueLabels,predictions)
    print('\r MACRO LRAP: %s' % (str(round(value, 4))), end=100 * ' ' + '\n')

    predictions = (predictions > THRESHOLD).astype(int)

    print("Accuracy Macro:" + str(accuracyMacro(trueLabels, predictions)))
    print("Accuracy Micro:" + str(accuracyMicro(trueLabels, predictions)))
    print("Precision Macro:" + str(precisionMacro(trueLabels, predictions)))
    print("Precision Micro:" + str(precisionMicro(trueLabels, predictions)))
    print("Recall Macro:" + str(recallMacro(trueLabels, predictions)))
    print("Recall Micro:" + str(recallMicro(trueLabels, predictions)))
    print("FBeta Macro:" + str(fbetaMacro(trueLabels, predictions, beta=1)))
    print("FBeta Micro:" + str(fbetaMicro(trueLabels, predictions, beta=1)))