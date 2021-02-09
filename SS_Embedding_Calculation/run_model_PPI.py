import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

import sys
import tensorflow as tf
import numpy as np
import os
import json
import rdflib

from OpenKE import config
from OpenKE import models

from process_KG_PPI import process_dataset, build_KG_domain, buildIds


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



def run_model(models_embeddings, n_embeddings, ontology_file_path, annotations_file_path, path_output, species, datasets , domains):
    """
    Calculate embeddings for each semantic aspect using KG and write them on a file.
    :param models_embeddings: list of embedding models;
    :param n_embeddings: dimension of embedding vectors;
    :param ontology_file_path: GO ontology file path in owl format;
    :param annotations_file_path: GOA annotations file path in GAF 2.1 version;
    :param path_output: OpenKE path;
    :param species: name of the species;
    :param datasets: list of datasets. Each dataset is a tuple (dataset, file_dataset_path, dataset_output). The "dataset" is the name of the dataset. The "file_dataset_path" is dataset file path with the protein pairs, the format of each line of the dataset files is "Ent1 Ent2 Proxy". The "dataset_output" is the embedding files path;
    :param domains: list of GO semantic aspects;
    :return: writes an embedding file with format "{ent1:[...], ent2:[...]}" for each GO semantic aspect.
    """
    for domain in domains:
        Graph_domain = build_KG_domain(ontology_file_path, annotations_file_path, domain)
        construct_embeddings(models_embeddings, n_embeddings, path_output, species, datasets, Graph_domain, domain)


def construct_embeddings(models_embeddings, n_embeddings, path_output, species, datasets, Graph, domain):
    """
    Calculate embedding using GO KG (GO + GO annotations) and write them on a file.
    :param models_embeddings: list of embedding models;
    :param n_embeddings: dimension of embedding vectors;
    :param path_output: OpenKE path;
    :param species: name of the species;
    :param datasets: list of datasets. Each dataset is a tuple (dataset, file_dataset_path, dataset_output). The "dataset" is the name of the dataset. The "file_dataset_path" is dataset file path with the protein pairs, the format of each line of the dataset files is "Ent1 Ent2 Proxy". The "dataset_output" is the embedding files path;
    :param Graph: knowledge graph;
    :param domain: semantic aspect;
    :return: writes an embedding file with format "{ent1:[...], ent2:[...]}".
    """
    dic_nodes, dic_relations, list_triples = buildIds(Graph)

    entity2id_file = open(path_output + "entity2id.txt", "w")
    entity2id_file.write(str(len(dic_nodes))+ '\n')

    for entity , id in dic_nodes.items():
        entity = entity.replace('\n' , ' ')
        entity = entity.replace(' ' , '__' )
        entity2id_file.write(str(entity) + '\t' + str(id)+ '\n')
    entity2id_file.close()

    relations2id_file = open(path_output + "relation2id.txt", "w")
    relations2id_file.write(str(len(dic_relations)) + '\n')
    for relation , id in dic_relations.items():
        relation  = relation.replace('\n' , ' ')
        relation = relation.replace(' ', '__')
        relations2id_file.write(str(relation) + '\t' + str(id) + '\n')
    relations2id_file.close()

    train2id_file = open(path_output + "train2id.txt", "w")
    train2id_file.write(str(len(list_triples)) + '\n')
    for triple in list_triples:
        train2id_file.write(str(triple[0]) + '\t' + str(triple[2]) + '\t' + str(triple[1]) + '\n')
    train2id_file.close()

    con = config.Config()
    #Input training files from data folder.
    con.set_in_path(path_output)
    con.set_dimension(n_embeddings)

    for model_embedding in models_embeddings:

        print('--------------------------------------------------------------------------------------------------------------------')
        print('MODEL: ' + model_embedding)

        # Models will be exported via tf.Saver() automatically.
        con.set_export_files(path_output + model_embedding + "/model_" + species + "_" + domain + ".vec.tf", 0)
        # Model parameters will be exported to json files automatically.
        con.set_out_files(path_output + model_embedding + "/embedding_" + species + "_" + domain + ".vec.json")

        if model_embedding == 'ComplEx':
            con.set_work_threads(8)
            con.set_train_times(1000)
            con.set_nbatches(100)
            con.set_alpha(0.5)
            con.set_lmbda(0.05)
            con.set_bern(1)
            con.set_ent_neg_rate(1)
            con.set_rel_neg_rate(0)
            con.set_opt_method("Adagrad")
            # Initialize experimental settings.
            con.init()
            #Set the knowledge embedding model
            con.set_model(models.ComplEx)

        elif model_embedding == 'distMult':
            con.set_work_threads(8)
            con.set_train_times(1000)
            con.set_nbatches(100)
            con.set_alpha(0.5)
            con.set_lmbda(0.05)
            con.set_bern(1)
            con.set_ent_neg_rate(1)
            con.set_rel_neg_rate(0)
            con.set_opt_method("Adagrad")
            # Initialize experimental settings.
            con.init()
            # Set the knowledge embedding model
            con.set_model(models.DistMult)


        elif model_embedding == 'HOLE':
            con.set_work_threads(4)
            con.set_train_times(500)
            con.set_nbatches(100)
            con.set_alpha(0.1)
            con.set_bern(0)
            con.set_margin(0.2)
            con.set_ent_neg_rate(1)
            con.set_rel_neg_rate(0)
            con.set_opt_method("Adagrad")
            # Initialize experimental settings.
            con.init()
            # Set the knowledge embedding model
            con.set_model(models.HolE)


        elif model_embedding == 'RESCAL':

            con.set_work_threads(4)
            con.set_train_times(500)
            con.set_nbatches(100)
            con.set_alpha(0.1)
            con.set_bern(0)
            con.set_margin(1)
            con.set_ent_neg_rate(1)
            con.set_rel_neg_rate(0)
            con.set_opt_method("Adagrad")
            # Initialize experimental settings.
            con.init()
            # Set the knowledge embedding model
            con.set_model(models.RESCAL)


        elif model_embedding == 'TransD':
            con.set_work_threads(8)
            con.set_train_times(1000)
            con.set_nbatches(100)
            con.set_alpha(1.0)
            con.set_margin(4.0)
            con.set_bern(1)
            con.set_ent_neg_rate(1)
            con.set_rel_neg_rate(0)
            con.set_opt_method("SGD")
            # Initialize experimental settings.
            con.init()
            # Set the knowledge embedding model
            con.set_model(models.TransD)

        elif model_embedding == 'TransE':
            con.set_work_threads(8)
            con.set_train_times(1000)
            con.set_nbatches(100)
            con.set_alpha(0.001)
            con.set_margin(1.0)
            con.set_bern(0)
            con.set_ent_neg_rate(1)
            con.set_rel_neg_rate(0)
            con.set_opt_method("SGD")
            # Initialize experimental settings.
            con.init()
            # Set the knowledge embedding model
            con.set_model(models.TransE)


        elif model_embedding == 'TransH':
            con.set_work_threads(8)
            con.set_train_times(1000)
            con.set_nbatches(100)
            con.set_alpha(0.001)
            con.set_margin(1.0)
            con.set_bern(0)
            con.set_ent_neg_rate(1)
            con.set_rel_neg_rate(0)
            con.set_opt_method("SGD")
            # Initialize experimental settings.
            con.init()
            # Set the knowledge embedding model
            con.set_model(models.TransH)


        elif model_embedding == 'TransR':

            con.set_work_threads(8)
            con.set_train_times(1000)
            con.set_nbatches(100)
            con.set_alpha(1.0)
            con.set_lmbda(4.0)
            con.set_margin(1)
            con.set_ent_neg_rate(1)
            con.set_rel_neg_rate(0)
            con.set_opt_method("SGD")
            # Initialize experimental settings.
            con.init()
            # Set the knowledge embedding model
            con.set_model(models.TransR)


        # Train the model.
        con.run()


        for dataset, file_dataset_path, dataset_output in datasets:

            dict_labels, prots = process_dataset(file_dataset_path)

            with open(path_output + model_embedding + "/embedding_" + species + "_" + domain + ".vec.json", 'r') as embeddings_file:
                data = embeddings_file.read()
            embeddings = json.loads(data)
            embeddings_file.close()

            ensure_dir(dataset_output +  domain + '/')
            with open(dataset_output +  domain + '/' + 'Embeddings_' + str(dataset) + '_' + str(model_embedding) + "_" + domain + '.txt', 'w') as file_output:
                file_output.write("{")
                first = False
                for i in range(len(prots)):
                    prot = prots[i]
                    if first:
                        if "ent_embeddings" in embeddings:
                            file_output.write(", '%s':%s" % (str(prot), str(embeddings["ent_embeddings"][dic_nodes[str(prot)]])))
                        else:
                            file_output.write(
                                ", '%s':%s" % (str(prot), str(embeddings["ent_re_embeddings"][dic_nodes[str(prot)]])))

                    else:
                        if "ent_embeddings" in embeddings:
                            file_output.write("'%s':%s" % (str(prot), str(embeddings["ent_embeddings"][dic_nodes[str(prot)]])))
                        else:
                            file_output.write(
                                "'%s':%s" % (str(prot), str(embeddings["ent_re_embeddings"][dic_nodes[str(prot)]])))
                        first = True
                file_output.write("}")
            file_output.close()



#############################
##    Calling Functions    ##
#############################


if __name__ == "__main__":

    n_arguments = len(sys.argv)

    if n_arguments == 1:

        domains = ["biological_process", "molecular_function", "cellular_component"]
        models_embeddings = ['distMult' , 'TransE', 'TransH', 'TransD', 'TransR', 'ComplEx']
        n_embeddings = 200
        ontology_file_path = './Data/GOdata/go.owl'
        path_output = './SS_Embedding_Calculation/OpenKE/'

        annot_files = [("fly", ["STRING_DM"] ), ( "ecoli", ["STRING_EC"]), ("yeast", ["STRING_SC", "DIPMIPS_SC", "BIND_SC"]), ("human", ["DIP_HS", "STRING_HS", "GRIDHPRD_bal_HS", "GRIDHPRD_unbal_HS"])]
        for species, name_datasets in annot_files:

            annotations_file_path = './Data/GOdata/goa_' + species + '_20.gaf'
            datasets = []
            for dataset in name_datasets:
                file_dataset_path = './Data/PPIdatasets/' + dataset + '/' + dataset + '.txt'
                path_output_embeddings = './SS_Embedding_Calculation/Embeddings/' + dataset + '/'
                datasets.append((dataset , file_dataset_path, path_output_embeddings))
            run_model(models_embeddings, n_embeddings, ontology_file_path, annotations_file_path, path_output, species, datasets, domains)

    else:

        domains = ["biological_process", "molecular_function", "cellular_component"]
        path_output = './SS_Embedding_Calculation/OpenKE/'

        ontology_file_path = sys.argv[1] 
        annotations_file_path =  sys.argv[2]
        species = sys.argv[3]
        dataset_name = sys.argv[4]
        file_dataset_path = sys.argv[5]
        path_output_embeddings = sys.argv[6]
        models_embeddings = [sys.argv[7]]
        n_embeddings =  int(sys.argv[8])
        
        datasets = [(dataset_name, file_dataset_path,  path_output_embeddings)]
        run_model(models_embeddings, n_embeddings, ontology_file_path, annotations_file_path, path_output, species, datasets, domains)








