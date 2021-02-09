from binary_classification import waf_calculation, labels_prediction_with_cuttof_baselines, predictions_baselines

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import metrics
from operator import itemgetter
import numpy as np


def curve_ROC (list_labels, list_SS, graphic_name, graphic_title, aspects):

    dict_SS_baselines= {}
    for aspect in aspects:
        dict_SS_baselines[aspect] = []

    for ss in list_SS:
        for i in range(len(aspects)):
            dict_SS_baselines[aspects[i]].append(ss[i])

    colors = list(cm.rainbow(np.linspace(0, 1, len(aspects))))

    plt.figure()
    for i in range(len(aspects)):
        FRP, TPR, _ = metrics.roc_curve(list_labels, dict_SS_baselines[aspects[i]])
        plt.plot(FRP, TPR, color=colors[i], lw=2, label=aspects[i])

    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.title(graphic_title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(graphic_name)
    plt.close('all')



def curve_fmeasure(list_labels, list_SS, graphic_name, graphic_title, aspects):
    cutoffs = list(np.arange(-1, 1, 0.01))

    dict_WAFs_baselines= {}
    for aspect in aspects:
        dict_WAFs_baselines[aspect] = []

    for cutoff in cutoffs:
        for i in range(len(aspects)):
            dict_WAFs_baselines[aspects[i]].append(waf_calculation(labels_prediction_with_cuttof_baselines(list_SS, cutoff, i)[0], list_labels))

    plt.figure()
    colors = list(cm.rainbow(np.linspace(0, 1, len(aspects))))

    for i in range(len(aspects)):
        plt.plot(cutoffs, dict_WAFs_baselines[aspects[i]], color=colors[i], lw=2, label=aspects[i])

    plt.xlim([-1.0, 1.05])
    plt.ylim([-1.0, 1.05])
    plt.title(graphic_title)
    plt.xlabel('Semantic Similarity Cutoff')
    plt.ylabel('Weighted Average F-measure')
    plt.legend()
    plt.savefig(graphic_name)
    plt.close('all')



def plots_baselines(list_labels, list_SS, path_graphic, dataset_name, SSM, Run, aspects):
    curve_ROC(list_labels, list_SS, path_graphic + dataset_name + '__' + SSM + '__ROC__Run' + str(Run) + '.png', SSM, aspects)
    curve_fmeasure(list_labels, list_SS,path_graphic + dataset_name + '__' + SSM + '__WAF__Run' + str(Run) + '.png', SSM, aspects)



def performance_baseline(X_train, X_test, y_train, y_test, list_SS, list_labels, filename_output, index_SS, cutoffs):

    WAFs_TrainingSet = {}
    for cutoff in cutoffs:
        WAFs_TrainingSet[cutoff] = waf_calculation(labels_prediction_with_cuttof_baselines(X_train, cutoff, index_SS)[0], y_train)
        
    max_cutoff_TrainingSet, max_waf_TrainingSet = max(WAFs_TrainingSet.items(), key=itemgetter(1))
    file_output = open(filename_output, 'a')
    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy, auc = predictions_baselines(y_test, X_test, max_cutoff_TrainingSet, index_SS)
 
    file_output.write(str(waf) + '\t' + str(fmeasure_noninteract) + '\t' + str(fmeasure_interact) + '\t' +
        str(precision) + '\t' + str(recall) + '\t' + str(accuracy) + '\t' + str(auc) + '\n')
    file_output.close()

    return waf, max_cutoff_TrainingSet



def performance_baselines(X_train, X_test, y_train, y_test, list_SS, list_labels, path_output, aspects):

    cutoffs = list(np.arange(-1, 1, 0.01))
    waf_baselines = []

    for i in range(len(aspects)):
        waf , max_cutoff_TrainingSet = performance_baseline(X_train, X_test, y_train, y_test, list_SS, list_labels, path_output + '__'+ str(aspects[i]) + '.txt', i , cutoffs)
        print('Maximum cutoff for '+ str(aspects[i]) + ' Baseline: ' + str(max_cutoff_TrainingSet))
        print('WAF in Test Set: ' + str(waf))
        waf_baselines.append(waf)

    return waf_baselines
