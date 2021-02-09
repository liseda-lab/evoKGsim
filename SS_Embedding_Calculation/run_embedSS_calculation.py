import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

#####################
##    Functions    ##
#####################

def ensure_dir(f):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param f: A path-like object representing a file system path.
    """
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        
def process_dataset(file_dataset_path):
    """
    Process the dataset file and returns a list with the proxy values for each pair of entities.
    :param file_dataset_path: dataset file path. The format of each line of the dataset files is "Ent1  Ent2   Proxy";
    :return: a list of lists. Each list represents a entity pair composed by 2 elements: a tuple (ent1,ent2) and the proxy value;
    """
    dataset = open(file_dataset_path, 'r')
    labels_list = []

    for line in dataset:
        split1 = line.split('\t')
        ent1, ent2 = split1[0], split1[1]
        label = float(split1[-1][:-1])

        url_ent1 = "http://" + ent1
        url_ent2 = "http://" + ent2
    
        labels_list.append([(url_ent1, url_ent2),label])

    dataset.close()
    return labels_list

def process_embedding_files(file_dataset_path, list_embeddings_files, output_file):
    """
    Compute cosine similarity between embeddings and write them.
    :param file_dataset_path: dataset file path with the entity pairs. The format of each line of the dataset files is "Ent1 Ent2 Proxy";
    :param list_embeddings_files: list of the embeddings files for each semantic aspect;
    :param output_file: new embedding similarity file path;
    :return: new similarity file;
    """
    list_labels_prots = process_dataset(file_dataset_path)

    list_dict = []
    for embedding_file in list_embeddings_files:
        dict_embeddings= eval(open(embedding_file, 'r').read())
        list_dict.append(dict_embeddings)
        
    o=open(output_file,"w")
    for pair, label in list_labels_prots:
        prot1=pair[0]
        prot2=pair[1]
        o.write(prot1+'\t'+prot2)
        
        for dict_embeddings in list_dict:
            prot1 = np.array(dict_embeddings[pair[0]])
            prot1 = prot1.reshape(1,len(prot1))

            prot2 = np.array(dict_embeddings[pair[1]])
            prot2 = prot2.reshape(1,len(prot2))

            sim = cosine_similarity(prot1, prot2)[0][0]
            
            o.write('\t' + str(sim))

        o.write('\n')
    o.close()


#############################
##    Calling Functions    ##
#############################


if __name__ == "__main__":

    n_arguments = len(sys.argv)
    if n_arguments == 1:

        datasets = ["DIP_HS" , "STRING_DM" , "STRING_EC" , "STRING_HS" , "STRING_SC" , "GRIDHPRD_bal_HS" , "GRIDHPRD_unbal_HS" , "DIPMIPS_SC" , "BIND_SC"]
        embSSMs = ['TransE', 'TransH', 'TransD', 'rdf2vec_skip-gram_wl', 'distMult', 'ComplEx', 'TransR']
        n_embeddings = 200
        
        for dataset in datasets:
            for model_embedding in embSSMs:
                file_dataset_path = 'Data/PPIdatasets/' + dataset + '/' + dataset + '.txt'
                path_embedding_file_bp = 'SS_Embedding_Calculation/Embeddings/' + dataset + '/biological_process/Embeddings_' + dataset + '_'+ model_embedding + '_biological_process.txt'
                path_embedding_file_cc = 'SS_Embedding_Calculation/Embeddings/' + dataset + '/cellular_component/Embeddings_' + dataset + '_'+ model_embedding + '_cellular_component.txt'
                path_embedding_file_mf = 'SS_Embedding_Calculation/Embeddings/' + dataset + '/molecular_function/Embeddings_' + dataset + '_'+ model_embedding + '_molecular_function.txt'
                list_embeddings_files = [path_embedding_file_bp,path_embedding_file_cc,path_embedding_file_mf]
                output_file = 'SS_Embedding_Calculation/Embeddings_SS_files/' + dataset + '/embedss_' + str(n_embeddings)+'_'+ model_embedding + '_'+dataset+'.txt'
                ensure_dir('SS_Embedding_Calculation/Embeddings_SS_files/' + dataset + '/')
                process_embedding_files(file_dataset_path, list_embeddings_files ,output_file)

        
    else:
        file_dataset_path = sys.argv[1]
        path_embedding_file_bp = sys.argv[2]
        path_embedding_file_cc = sys.argv[3]
        path_embedding_file_mf = sys.argv[4]
        output_file = sys.argv[5]
        ensure_dir(output_file)
        list_embeddings_files = [path_embedding_file_bp,path_embedding_file_cc,path_embedding_file_mf]
        process_embedding_files(file_dataset_path, list_embeddings_files ,output_file)

