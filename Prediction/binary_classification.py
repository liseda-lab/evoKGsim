
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
import math

def rmse(y, y_pred):
    return np.sqrt(np.average((y_pred - y) ** 2))



def labels_prediction_with_cuttof_baselines(list_SS, cuttof, index_SS):
    predicted_labels = []
    predicted_values = []
    for SSs in list_SS:
        SS = SSs[index_SS]
        predicted_values.append(SS)
        if SS >= cuttof:
            prediction = 1
        else:
            prediction = 0
        predicted_labels.append(prediction)
    return predicted_labels, predicted_values



def labels_prediction_with_cuttof(list_predictions, cuttof):
    predicted_labels = []
    for prediction in list_predictions:
        if prediction >= cuttof:
            label_predict = 1
        else:
            label_predict = 0
        predicted_labels.append(label_predict)
    return predicted_labels



def waf_calculation(predicted_labels , list_labels):
    waf = metrics.f1_score(list_labels, predicted_labels, average='weighted')
    return waf



def predictions(predicted_labels, list_labels, predicted_values):
    waf = waf_calculation(predicted_labels , list_labels)
    fmeasure_noninteract, fmeasure_interact = metrics.f1_score(list_labels, predicted_labels, average=None)
    auc = metrics.roc_auc_score(list_labels, predicted_values, average='weighted')
    precision = metrics.precision_score(list_labels, predicted_labels)
    recall = metrics.recall_score(list_labels, predicted_labels)
    accuracy = metrics.accuracy_score(list_labels, predicted_labels)
    return [waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy, auc]



def predictions_baselines(list_labels, list_SS, cutoff , index_SS):
    predicted_labels, predicted_values = labels_prediction_with_cuttof_baselines(list_SS, cutoff, index_SS)
    return predictions(predicted_labels, list_labels, predicted_values)



def classification_report_summary (list_predictions, list_labels):
    predicted_labels = labels_prediction_with_cuttof(list_predictions, 0.5)
    return predictions(predicted_labels, list_labels, list_predictions)



def classification_report_with_predictions_summary(list_predictions, list_labels , filename_output):
    file_output = open(filename_output, 'w')
    file_output.write('Predicted_output' + '\t' + 'Expected_Output' + '\n')
    for i in range(len(list_predictions)):
        label = list_labels[i]
        prediction = list_predictions[i]
        file_output.write(str(prediction) + '\t' + str(label) + '\n')
    file_output.close()
    return classification_report_summary(list_predictions, list_labels)


