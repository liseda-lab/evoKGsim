
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import _Fitness

import binary_classification
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from sklearn import warnings

import numpy as np
import _pickle as pickle
import math
import time

from sklearn.utils.validation import check_array
from sklearn.model_selection import train_test_split


###################################
#####      GP Parameters      #####
###################################

population_size_value=500
generations_value=50
tournament_size_value=20
stopping_criteria_value=0.0
const_range_value=(-1, 1)
init_depth_value=(2, 6)
init_method_value='half and half'
function_set_value= ['add', 'sub', 'mul', 'max', 'min', 'div']
metric_value='rmse'
parsimony_coefficient_value=0.00001
p_crossover_value=0.9
p_subtree_mutation_value=0.01
p_hoist_mutation_value=0.01
p_point_mutation_value=0.01
p_point_replace_value=0.05
max_samples_value=1.0
warm_start_value=False
n_jobs_value=1
verbose_value=1
random_state_value=None
    


########################################
##   GENETIC PROGRAMMING ALGORITHM    ##
########################################

def genetic_programming_algorithm(X_train, X_test, y_train, y_test, list_ss, list_label, filename_output,
                                  filename_graphics, filename_predictions, fitness='all'):
    """
    Applies Genetic Programming Algorithm.
    :param X_train: the training input samples. The shape of the list is (n_samplesTrain, n_aspects);
    :param X_test: the testing input samples. The shape of the list is (n_samplesTest, n_aspects);
    :param y_train: the target values (proxy values) of the training set. The shape of the list is (n_samplesTrain);
    :param y_test: the target values (proxy values) of the test set. The shape of the list is (n_samplesTest);
    :param list_ss: the samples. The shape of the list is (n_samples, n_aspects);
    :param list_label: the target values (labels). The shape of the list is (n_samples);
    :param filename_predictions: path of predictions file;
    """
    file_output = open(filename_output, 'a')

    start_gp = time.time()
    # Genetic Programming
    
    if fitness == 'all':
        gp = SymbolicRegressor(population_size=population_size_value,
                                       generations=generations_value,
                                       tournament_size=tournament_size_value,
                                       stopping_criteria=stopping_criteria_value,
                                       const_range=const_range_value,
                                       init_depth=init_depth_value,
                                       init_method=init_method_value,
                                       function_set=function_set_value,
                                       metric=metric_value,
                                       parsimony_coefficient=parsimony_coefficient_value,
                                       p_crossover=p_crossover_value,
                                       p_subtree_mutation=p_subtree_mutation_value,
                                       p_hoist_mutation=p_hoist_mutation_value,
                                       p_point_mutation=p_point_mutation_value,
                                       p_point_replace=p_point_replace_value,
                                       max_samples=max_samples_value,
                                       warm_start=warm_start_value,
                                       n_jobs=n_jobs_value,
                                       verbose=verbose_value,
                                       random_state=random_state_value)


    else:

        if fitness == 'half':
            metric_function = _Fitness(_fitness_function_weighted_average_fmeasure_withhalfindividuals,
                                               greater_is_better=False)

        elif fitness == 'half':
            metric_function = _Fitness(_fitness_function_weighted_average_fmeasure_with25individuals,
                                               greater_is_better=False)

        elif fitness == 'tenth':
            metric_function = _Fitness(_fitness_function_weighted_average_fmeasure_with10individuals,
                                               greater_is_better=False)

        gp = SymbolicRegressor(population_size=population_size_value,
                                       generations=generations_value,
                                       tournament_size=tournament_size_value,
                                       stopping_criteria=stopping_criteria_value,
                                       const_range=const_range_value,
                                       init_depth=init_depth_value,
                                       init_method=init_method_value,
                                       function_set=function_set_value,
                                       metric=metric_function,
                                       parsimony_coefficient=parsimony_coefficient_value,
                                       p_crossover=p_crossover_value,
                                       p_subtree_mutation=p_subtree_mutation_value,
                                       p_hoist_mutation=p_hoist_mutation_value,
                                       p_point_mutation=p_point_mutation_value,
                                       p_point_replace=p_point_replace_value,
                                       max_samples=max_samples_value,
                                       warm_start=warm_start_value,
                                       n_jobs=n_jobs_value,
                                       verbose=verbose_value,
                                       random_state=random_state_value)

    parameters = gp.get_params()

    gp.fit(X_train, y_train)
    end_gp = time.time()


    print('\n')
    print('And the winner is:   ' + str(gp._program))
    print('\n')

    waf_train , waf_test = [], []
    precision_train, precision_test = [], []
    recall_train, recall_test = [], []
    rmse_train, rmse_test = [], []
    generations = []

    generation = 0
    for program in gp.best_individuals():
        list_ss = check_array(list_ss)
        _, gp.n_features = list_ss.shape
        predictions = program.execute(list_ss)

        X_test = check_array(X_test)
        _, gp.n_features = X_test.shape
        test_predictions = program.execute(X_test)

        X_train = check_array(X_train)
        _, gp.n_features = X_train.shape
        train_predictions = program.execute(X_train)

        classification_report_train = binary_classification.classification_report_summary(train_predictions, y_train)
        classification_report_test = binary_classification.classification_report_summary(test_predictions, y_test)
        classification_report = binary_classification.classification_report_summary(predictions, list_label)

        rmse_train_value = binary_classification.rmse(train_predictions, y_train)
        rmse_test_value = binary_classification.rmse(test_predictions, y_test)

        rmse_train.append(rmse_train_value)
        rmse_test.append(rmse_test_value)

        waf_train.append(classification_report_train[0])
        waf_test.append(classification_report_test[0])

        precision_train.append(classification_report_train[3])
        precision_test.append(classification_report_test[3])

        recall_train.append(classification_report_train[4])
        recall_test.append(classification_report_test[4])

        generations.append(generation)

        file_output.write(
            'Generation ' + str(generation) + '\t' + str(rmse_train_value) + '\t' + str(rmse_test_value) + '\t'  + str(classification_report_train[0]) + '\t' +  str(classification_report_test[0])+ '\t'
            +  str(classification_report[0]) + '\t' +  str(classification_report_train[1]) + '\t' +  str(classification_report_test[1])+ '\t' +  str(classification_report[1]) + '\t' +
            str(classification_report_train[2]) + '\t' +  str(classification_report_test[2])+ '\t' +  str(classification_report[2]) + '\t' +  str(classification_report_train[3]) + '\t' +
            str(classification_report_test[3])+ '\t' +  str(classification_report[3]) + '\t' +  str(classification_report_train[4]) + '\t' +  str(classification_report_test[4])+ '\t' +
            str(classification_report[4]) + '\t' +  str(classification_report_train[5]) + '\t' +  str(classification_report_test[5])+ '\t' +  str(classification_report[5]) + '\t' +
            str(classification_report_train[6]) + '\t' +  str(classification_report_test[6])+ '\t' +  str(classification_report[6]) + '\n')

        generation = generation + 1

    # Classification with the best program
    print('\n')
    print('Classification using the winner....')

    test_predictions = program.execute(X_test)
    train_predictions = program.execute(X_train)
    predictions = program.execute(list_ss)

    final_rmse_train_value = binary_classification.rmse(train_predictions, y_train)
    final_rmse_test_value = binary_classification.rmse(test_predictions, y_test)
    final_classification_report_train = binary_classification.classification_report_with_predictions_summary(
        train_predictions, y_train, filename_predictions + '_TrainSet')
    final_classification_report_test = binary_classification.classification_report_with_predictions_summary(
        test_predictions, y_test, filename_predictions + '_TestSet')
    final_classification_report = binary_classification.classification_report_with_predictions_summary(
        predictions, list_label, filename_predictions)

    file_output.write(
        'Final Classification' + '\t' + str(final_rmse_train_value) + '\t' + str(final_rmse_test_value) + '\t' + str(final_classification_report_train[0]) +
        '\t' + str(final_classification_report_test[0])+ '\t' + str(final_classification_report[0]) + '\t' + str(final_classification_report_train[1]) +
        '\t' + str(final_classification_report_test[1])+ '\t' + str(final_classification_report[1]) + '\t' + str(final_classification_report_train[2]) +
        '\t' + str(final_classification_report_test[2])+ '\t' + str(final_classification_report[2]) + '\t' + str(final_classification_report_train[3]) +
        '\t' + str(final_classification_report_test[3])+ '\t' + str(final_classification_report[3]) + '\t' + str(final_classification_report_train[4]) +
        '\t' + str(final_classification_report_test[4])+ '\t' + str(final_classification_report[4]) + '\t' + str(final_classification_report_train[5]) +
        '\t' + str(final_classification_report_test[5])+ '\t' + str(final_classification_report[5]) + '\t' + str(final_classification_report_train[6]) +
        '\t' + str(final_classification_report_test[6])+ '\t' + str(final_classification_report[6]) + '\t' + str(gp._program) + '\t' + str(parameters) + '\n')

    file_output.write('\n')
    file_output.close()

    # Grafico da WAF usando o melhor individuo de cada geracao para classificacao no test set e no training set (usando cutoff=0.5)
    plt.figure()
    plt.plot(generations, waf_train, color='darkblue', lw=2, label='WAF on Training Set with cutoff = 0.5')
    plt.plot(generations, waf_test, color='skyblue', lw=2, label='WAF on Test Set with cutoff = 0.5')
    plt.xlabel('Generations')
    plt.ylabel('WAF')
    plt.legend()
    plt.savefig(filename_graphics + '__Generation_vs_WAF.png')

    # Grafico da Precision usando o melhor individuo de cada geracao para classificacao no test set e no training set (usando cutoff=0.5)
    plt.figure()
    plt.plot(generations, precision_train, color='darkblue', lw=2, label='Precision on Training Set with cutoff = 0.5')
    plt.plot(generations, precision_test, color='skyblue', lw=2, label='Precision on Test Set with cutoff = 0.5')
    plt.xlabel('Generations')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(filename_graphics + '__Generation_vs_Precision.png')

    # Grafico da Recall usando o melhor individuo de cada geracao para classificacao no test set e no training set (usando cutoff=0.5)
    plt.figure()
    plt.plot(generations, recall_train, color='darkblue', lw=2, label='Recall on Training Set with cutoff = 0.5')
    plt.plot(generations, recall_test, color='skyblue', lw=2, label='Recall on Test Set with cutoff = 0.5')
    plt.xlabel('Generations')
    plt.ylabel('Recall')
    plt.legend()
    plt.savefig(filename_graphics + '__Generation_vs_Recall.png')

    # Grafico da RMSE usando o melhor individuo de cada geracao para classificacao no test set e no training set
    plt.figure()
    plt.plot(generations, rmse_train, color='darkblue', lw=2, label='RMSE on Training Set')
    plt.plot(generations, rmse_test, color='skyblue', lw=2, label='RMSE on Test Set')
    plt.xlabel('Generations')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(filename_graphics + '__Generation_vs_RMSE.png')

    plt.close('all')
    
    print('WAF in Test Set: ' + str(final_classification_report_test[0]))
    time_execution_gp = end_gp - start_gp

    return final_classification_report_test[0], time_execution_gp



############################
##   FITNESS FUNCTIONS    ##
############################

def _fitness_function_weighted_average_fmeasure_withhalfindividuals(y, y_pred, sample_weight):
    """
    Fitness function that trains with half of the individuals.
    :param y: input target y vector.
    :param y_pred: the predicted values from the genetic program.
    :param sample_weight: sample_weight vector.
    :return: floating point number corresponding to the rmse between the predicted values and the labels of half individuals.
    """
    train_y, test_y, train_y_pred, test_y_pred = train_test_split(y, y_pred, train_size=0.5)

    # Calculate the root mean square error
    rmse = np.sqrt(np.average((train_y_pred - train_y) ** 2))

    return rmse


def _fitness_function_weighted_average_fmeasure_with25individuals(y, y_pred, sample_weight):
    """
    Fitness function that trains with one quarter of the individuals.
    :param y: input target y vector.
    :param y_pred: the predicted values from the genetic program.
    :param sample_weight: sample_weight vector.
    :return: floating point number corresponding to the rmse between the predicted values and the labels of one quarter of the individuals.
    """

    train_y, test_y, train_y_pred, test_y_pred = train_test_split(y, y_pred, train_size=0.25)

    # Calculate the root mean square error
    rmse = np.sqrt(np.average((train_y_pred - train_y) ** 2))

    return rmse


def _fitness_function_weighted_average_fmeasure_with10individuals(y, y_pred, sample_weight):
    """
    Fitness function that trains with tenth of the individuals.
    :param y: input target y vector.
    :param y_pred: the predicted values from the genetic program.
    :param sample_weight: sample_weight vector.
    :return: floating point number corresponding to the rmse between the predicted values and the labels of tenth of the individuals.
    """

    train_y, test_y, train_y_pred, test_y_pred = train_test_split(y, y_pred, train_size=0.1)

    # Calculate the root mean square error
    rmse = np.sqrt(np.average((train_y_pred - train_y) ** 2))

    return rmse




