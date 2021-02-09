import numpy
import os
import pandas as pd
from statistics import mean, median
from sklearn.model_selection import StratifiedKFold

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


def process_dataset_files(file_dataset_path):
    """
    Process the dataset file.
    :param file_dataset_path: dataset file path with the correspondent entity pairs. The format of each line of the dataset files is "Ent1 Ent2 Label";
    :return:
    - labels_list is a list of lists. Each list represents a pair of entities and its label;
    - prot_pairs is a list of tuples. Each tuple represents a pair of entities;
    - prot_labels is a list of labels;
    """
    dataset = open(file_dataset_path, 'r')
    labels_list = []
    prot_pairs, prot_labels = [], []
    
    for line in dataset:
        split1 = line.split('\t')
        prot1, prot2 = split1[0], split1[1]
        label = int(split1[-1][:-1])

        url_prot1 =  prot1
        url_prot2 =  prot2

        labels_list.append([(url_prot1, url_prot2),label])
        prot_pairs.append((url_prot1, url_prot2))
        prot_labels.append(label)
        
    dataset.close()
    return labels_list, prot_pairs, prot_labels



def run_partition(file_dataset_path, filename_output ):
    """
    Write partition files with indexes of each test partition.
    :param file_dataset_path: dataset file path with the correspondent entity pairs. The format of each line of the dataset files is "Ent1 Ent2 Label";
    :param filename_output: the partition files path;
    :return: partition files;
    """
    labels_list, prot_pairs, prot_labels  = process_dataset_files(file_dataset_path)
    n_pairs = len(prot_labels)

    index_partition = 1
    skf = StratifiedKFold(n_splits=10 , shuffle=True)

    ensure_dir(filename_output)

    for indexes_partition_train, indexes_partition_test in skf.split(prot_pairs, prot_labels):

        file_crossValidation_train = open(filename_output + 'Indexes__crossvalidationTrain__Run' + str(index_partition) + '.txt', 'w')
        file_crossValidation_test = open(filename_output + 'Indexes__crossvalidationTest__Run' + str(index_partition) + '.txt', 'w')
        for index in indexes_partition_train:
            file_crossValidation_train.write(str(index) + '\n')
        for index in indexes_partition_test:
            file_crossValidation_test.write(str(index) + '\n')
        file_crossValidation_train.close()
        file_crossValidation_test.close()

        index_partition = index_partition + 1



#############################
##    Calling Functions    ##
#############################

if __name__== '__main__':

    #datasets = ["STRING_DM", "STRING_EC", "STRING_SC", "DIPMIPS_SC", "BIND_SC", "DIP_HS", "STRING_HS", "GRIDHPRD_bal_HS", "GRIDHPRD_unbal_HS"]
    datasets = ["DIP_HS"]
    for dataset in datasets:
        file_dataset_path = 'Data/PPIdatasets/' + dataset + '/' + dataset + '.txt'
        filename_output = 'Prediction/Results/' + dataset + '/Shuffle_Partitions/'
        run_partition(file_dataset_path, filename_output)


