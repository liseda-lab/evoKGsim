from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from statistics import mean, median
import numpy as np
import time
import copy
import gc
import os
import sys

import genetic_programming
import baselines



#####################
##    Functions    ##
#####################

def ensure_dir(f):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param f: path-like object representing a file system path;
    """
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def process_SS_file(path_file_SS):
    """
    Process the similarity file and returns a dictionary with the similarity values for each pair of ebtities.
    :param path_file_SS: similarity file path. The format of each line of the similarity file is "Ent1  Ent2    Sim_SA1 Sim_SA2 Sim_SA3 ... Sim_SAn";
    :return: dict_SS is a dictionary where the keys are tuples of 2 entities and the values are the similarity values taking in consideration different semantic aspects.
    """
    file_SS = open(path_file_SS, 'r')
    dict_SS = {}

    for line in file_SS:
        line = line[:-1]
        split1 = line.split('\t')
        
        ent1 = split1[0].split('/')[-1]
        ent2 = split1[1].split('/')[-1]

        SS = split1[2:]
        SS_floats = [float(i) for i in SS]

        dict_SS[(ent1, ent2)] = SS_floats

    file_SS.close()
    return dict_SS


def process_dataset(path_dataset_file):
    """
    Process the dataset file and returns a list with the proxy value for each pair of entities.
    :param path_dataset_file: dataset file path. The format of each line of the dataset file is "Ent1  Ent2    Label";
    :return: list_proxies is a list where each element represents a list composed by [(ent1, ent2), label].
    """

    dataset = open(path_dataset_file, 'r')
    list_labels = []
    
    for line in dataset:
        split1 = line.split('\t')
        ent1, ent2 = split1[0], split1[1]
        label = int(split1[2][:-1])
        list_labels.append([(ent1, ent2), label])        

    dataset.close()
    return list_labels


def read_SS_dataset_file(path_file_SS, path_dataset_file):
    """
    Process the dataset file and the similarity file.
    :param path_file_SS: similarity file path. The format of each line of the similarity file is "Ent1  Ent2    Sim_SA1 Sim_SA2 Sim_SA3 ... Sim_SAn";
    :param path_dataset_file: dataset file path. The format of each line of the dataset file is "Ent1  Ent2    Proxy";
    :return: returns 4 lists.
    list_ents is a list of entity pairs in the dataset (each element of the list is a list [ent1, ent2]).
    list_SS is also a list of lists with the similarity values for each pair (each element of the list is a list [Sim_SA1,Sim_SA2,Sim_SA3,...,Sim_SAn]).
    list_SS_max_avg is a list of lists with the similarity values for each pair, including the average and the maximum (each element of the list is a list [Sim_SA1,Sim_SA2,Sim_SA3,...,Sim_SAn, Sim_AVG, Sim_MAX]).
    labels is a list of labels for each pair in the dataset.
        """
    list_SS = []
    list_SS_max_avg = []
    labels, list_ents = [], []

    dict_SS = process_SS_file(path_file_SS)
    list_labels = process_dataset(path_dataset_file)

    for (ent1, ent2), label in list_labels:

        SS_floats = dict_SS[(ent1, ent2)]

        max_SS = max(SS_floats)
        avg_SS = mean(SS_floats)

        list_ents.append([ent1, ent2])
        list_SS.append(SS_floats)
        
        new_SS_floats = copy.deepcopy(SS_floats)
        new_SS_floats.append(avg_SS)
        new_SS_floats.append(max_SS)
        list_SS_max_avg.append(new_SS_floats)
        labels.append(label)

    return list_ents, list_SS , list_SS_max_avg, labels


def process_indexes_partition(file_partition):
    """
    Process the partition file and returns a list of indexes.
    :param file_partition: partition file path (each line is a index);
    :return: list of indexes.
    """
    file_partitions = open(file_partition, 'r')
    indexes_partition = []

    for line in file_partitions:
        indexes_partition.append(int(line[:-1]))

    file_partitions.close()
    return indexes_partition



def run_cross_validation(path_file_SS, path_dataset_file, dataset_name, path_results, n_partition, path_partition, SSM, aspects, run_evolved = True, run_baselines = False):
    """
    Learn the best combination of semantic aspects.
    :param path_file_SS:  similarity file path. The format of each line of the similarity file is "Ent1  Ent2    Sim_SA1 Sim_SA2 Sim_SA3 ... Sim_SAn";
    :param path_dataset_file: ataset file path. The format of each line of the dataset file is "Ent1  Ent2    Label";
    :param dataset_name: name of the dataset;
    :param path_results: path where will be saved the results:
    :param n_partition: number of partitions;
    :param path_partition: the partition files path;
    :param SSM:  name of semantic similarity measure;
    :param aspects: list of semantic aspects;
    :param run_evolved: boolean. True for running the genetic programming and False otherwise. The default value is True;
    :param run_baselines: boolean. True for running the baselines and False otherwise. The default value is False;
    :return:
    """

    list_ents, list_ss_gp, list_ss_baselines, list_labels = read_SS_dataset_file(path_file_SS, path_dataset_file)

    if run_baselines:
        ensure_dir(path_results + '/' + SSM + "/ManualSelection/")
        aspects_baselines = aspects + ["Avg", "Max"]
        dict_baselines = {}
        for aspect in aspects_baselines:
            dict_baselines[aspect] = []

    if run_evolved:
        WAFs_gp= []
        ensure_dir(path_results + '/' + SSM + "/EvolvedCombinations/")

    n_pairs = len(list_labels)

    for Run in range(1, n_partition + 1):

        file_partition = path_partition + str(Run) + '.txt'
        test_index = process_indexes_partition(file_partition)
        train_index = list(set(range(0, n_pairs)) - set(test_index))
        
        print('###########################')
        print("######   RUN" + str(Run) + "       #######")
        print('###########################')

        list_labels = np.array(list_labels)
        y_train, y_test = list_labels[train_index], list_labels[test_index]
        y_train, y_test = list(y_train), list(y_test)

        list_ss_gp = np.array(list_ss_gp)
        list_ss_baselines = np.array(list_ss_baselines)

        X_train, X_test = list_ss_gp[train_index], list_ss_gp[test_index]
        X_train_baselines, X_test_baselines = list_ss_baselines[train_index], list_ss_baselines[test_index]
        X_train, X_test, X_train_baselines, X_test_baselines = list(X_train), list(X_test), list(X_train_baselines), list(X_test_baselines)

        
        print('\n')
        print('######################################' + SSM + '######################################')
        print('\n')
            
        if run_baselines:
            #############  MANUAL SELECTION BASELINES
            print('***************************************************')
            print('MANUAL SELECTION BASELINES .....')
            start_manualselection = time.time()

            path_graphic = path_results + '/' + SSM + "/ManualSelection/"
            path_output_baselines = path_results + '/' + SSM + "/ManualSelection/Measures__" + SSM + "__" + dataset_name + '__'
                       
            waf_baselines = baselines.performance_baselines(X_train_baselines, X_test_baselines, y_train, y_test, list_ss_baselines, list_labels, path_output_baselines, aspects_baselines)
            baselines.plots_baselines(y_train, X_train_baselines, path_graphic, dataset_name + '__TrainingSet', SSM, Run, aspects_baselines)
            baselines.plots_baselines(y_test, X_test_baselines, path_graphic, dataset_name + '__TestSet', SSM, Run, aspects_baselines)
            baselines.plots_baselines(list_labels, list_ss_baselines, path_graphic, dataset_name, SSM, Run, aspects_baselines)

            end_manualselection = time.time()
            print('Execution Time: ' + str(end_manualselection - start_manualselection))
            #############  END BASELINES

            for i in range(len(aspects_baselines)):
                dict_baselines[aspects_baselines[i]].append(waf_baselines[i])
  

        if run_evolved:

            #############  GENETIC PROGRAMMING
            print('***************************************************')
            print('GENETIC PROGRAMMING .....')

            filename_output_gp = path_results + '/' + SSM + "/EvolvedCombinations/Measures__" + SSM + "__" + dataset_name +  ".txt"
            filename_graphics_gp = path_results  + '/' + SSM + "/EvolvedCombinations/Plot__" + SSM + "__" + dataset_name +  "__Run" + str(Run)
            filename_predictions_gp = path_results  + '/' + SSM + "/EvolvedCombinations/Predictions__" + SSM + "__" + dataset_name +  "__Run" + str(Run)
            waf_gp, time_execution_gp = genetic_programming.genetic_programming_algorithm(X_train, X_test, y_train, y_test, list_ss_gp, list_labels, filename_output_gp, filename_graphics_gp, filename_predictions_gp, fitness='all')
            WAFs_gp.append(waf_gp)
            print('Execution Time: ' + str(time_execution_gp))
            #############  END GENETIC PROGRAMMING
    

    print('\n')
    print('###########################')
    print("######   RESULTS    #######")
    print('###########################')

    file_results = open(path_results + '/' + SSM + "/ResultsSummary.txt" , 'w') 
    if run_baselines:

        for i in range(len(aspects_baselines)):
        
            print('***************************************************')
            print("Median Baseline  "+ str(aspects_baselines[i]))
            print(str(dict_baselines[aspects_baselines[i]]))
            print(str(median(dict_baselines[aspects_baselines[i]])))
            file_results.write(str(aspects_baselines[i]) + '\t' + str(median(dict_baselines[aspects_baselines[i]])) + '\n')

    if run_evolved:
        print('***************************************************')
        print("Median GP: " )
        print(str(WAFs_gp))
        print(str(median(WAFs_gp)))
        file_results.write('GP' + '\t' + str(median(WAFs_gp)) + '\n')

    file_results.close()




#############################
##    Calling Functions    ##
#############################


if __name__ == "__main__":

    n_arguments = len(sys.argv)

    if n_arguments < 4:

        if n_arguments==3:
            if sys.argv[1] == 'False':
                run_evolved = False
            else:
                run_evolved = True

            if sys.argv[2] == 'False':
                run_baselines = False
            else:
                run_baselines = True

        if n_arguments==1:
            run_evolved = True
            run_baselines = True


        n_partition = 10
        aspects = ["biological_process","cellular_component", "molecular_function"]

        datasets = ["DIP_HS" , "STRING_DM" , "STRING_EC" , "STRING_HS" , "STRING_SC" , "GRIDHPRD_bal_HS" , "GRIDHPRD_unbal_HS" , "DIPMIPS_SC" , "BIND_SC"]
        embSSMs = ['TransE', 'TransH', 'TransD', 'rdf2vec_skip-gram_wl', 'distMult', 'ComplEx', 'TransR']
        SSMs = ["ResnikMax_ICResnik", "ResnikMax_ICSeco", "ResnikBMA_ICSeco", "ResnikBMA_ICResnik", "simGIC_ICSeco", "simGIC_ICResnik"]

        for dataset in datasets:
            path_dataset_file = 'Data/PPIdatasets/' + dataset + '/' + dataset + '.txt'
            path_results = 'Prediction/Results/' + dataset
            path_partition = 'Prediction/Results/' + dataset + '/Shuffle_Partitions/Indexes__crossvalidationTest__Run'
            
            for model_embedding in embSSMs:
                SSM = 'Embeddings_' + model_embedding
                path_file_SS = 'SS_Embedding_Calculation/Embeddings_SS_files/' + dataset + '/embedss_200_'+ model_embedding + '_' + dataset + '.txt'
                run_cross_validation(path_file_SS, path_dataset_file, dataset, path_results, n_partition, path_partition, SSM, aspects, run_evolved, run_baselines)
                gc.collect()

            for SSM in SSMs:
                path_file_SS = 'SS_Calculation/SS_files/' + dataset + '/ss_'+ SSM + '_' + dataset + '.txt'
                run_cross_validation(path_file_SS, path_dataset_file, dataset, path_results, n_partition, path_partition, SSM, aspects, run_evolved, run_baselines)
                gc.collect()
                
    else:
        dataset = sys.argv[1]
        SSM = sys.argv[2] 
        path_dataset_file = sys.argv[3]
        path_file_SS = sys.argv[4]
        n_partition = int(sys.argv[5])
        path_partition = sys.argv[6]
        path_results = sys.argv[7]
        n_aspects = int(sys.argv[8])

        aspects = []
        for i in range(n_aspects):
            aspect = sys.argv[9 + i]
            aspects.append(aspect)

        if sys.argv[-2] == 'False':
            run_evolved = False
        else:
            run_evolved = True

        if sys.argv[-1] == 'False':
            run_baselines = False
        else:
            run_baselines = True
        
        run_cross_validation(path_file_SS, path_dataset_file, dataset, path_results, n_partition, path_partition, SSM, aspects, run_evolved, run_baselines)

        

        






