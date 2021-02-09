import numpy
import os
import sys

import rdflib
from rdflib.namespace import RDF, OWL, RDFS
import json

from PyRDF2Vec.graph import *
from PyRDF2Vec.rdf2vec import RDF2VecTransformer

from process_KG_PPI import process_dataset, build_KG_domain

###################################
#####   RDF2Vec Parameters    #####
###################################

max_path_depth_value=8
max_tree_depth_value=2
window_value=5
n_jobs_value=4
max_iter_value=10
negative_value=25
min_count_value=1
wfl_iterations_value=4



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


def calculate_embeddings(g, prots, path_output, size, type_walk, type_word2vec, n_walks , domain, dataset):
    """
    Calculate embedding and write them on a file.
    :param g: knowledge graph;
    :param prots: list of entities for which embeddings will be calculated;
    :param path_output: embedding file path;
    :param size: dimension of embedding vectors;
    :param type_walk: indicates how to construct the sequences fed to the embedder (options are "wl" or "walk");
    :param type_word2vec: training algorithm (options are "skip-gram" or "CBOW");
    :param n_walks: maximum number of walks per entity;
    :param domain: semantic aspect;
    :param dataset: name of the dataset;
    :return: writes an embedding file with format "{prot1:[...], prot2:[...]}"
    """
    kg = rdflib_to_kg(g)
    graphs_prots = [extract_instance(kg, prot) for prot in prots]

    if type_word2vec == 'CBOW':
        sg = 0
    if type_word2vec == 'skip-gram':
        sg = 1

    print('----------------------------------------------------------------------------------------')
    print('Vector size: ' + str(size))
    print('Type Walk: ' + type_walk)
    print('Type Word2vec: ' + type_word2vec)

    transformer = RDF2VecTransformer(vector_size=size,
                                     max_path_depth=max_path_depth_value,
                                     max_tree_depth=max_tree_depth_value,
                                     _type=type_walk,
                                     walks_per_graph=n_walks,
                                     window=window_value,
                                     n_jobs=n_jobs_value,
                                     sg=sg,
                                     max_iter=max_iter_value,
                                     negative=negative_value,
                                     min_count=min_count_value,
                                     wfl_iterations=wfl_iterations_value)
    
    embeddings = transformer.fit_transform(graphs_prots)

    # Write embeddings
    print('Writing Embeddings ...')
    ensure_dir(path_output)
    with open(path_output + 'Embeddings_' + dataset +  '_rdf2vec_' + str(type_word2vec) + '_' + str(type_walk) + '_' + domain + '.txt', 'w') as file:
        file.write("{")
        first = False
        for i in range(len(prots)):
            if first:
                file.write(", '%s':%s" % (str(prots[i]), str(embeddings[i].tolist())))
            else:
                file.write("'%s':%s" % (str(prots[i]), str(embeddings[i].tolist())))
                first=True
            file.flush()
        file.write("}")
    print('Finished Embeddings Computation ...')
                

def run_RDF2Vec_GOembedddings(ontology_file_path, annotations_file_path, list_dataset, vector_sizes, types_walk, types_word2vec, n_walks, domains):
    """
    Calculate embedding using GO KG (GO + GO annotations) and write them on a file.
    :param ontology_file_path: GO ontology file path in owl format;
    :param annotations_file_path: GOA annotations file path in GAF 2.1 version;
    :param list_dataset: list of datasets. Each dataset is a tuple (dataset_name, path_dataset, path_output_embeddings). The "dataset_name" is the name of the dataset. The "path_dataset" is dataset file path with the protein pairs, the format of each line of the dataset files is "Ent1 Ent2 Proxy". The "path_output_embeddings" is the embedding files path;
    :param vector_sizes: dimension of embedding vectors;
    :param type_walk: indicates how to construct the sequences fed to the embedder (options are "wl" or "walk");
    :param type_word2vec: training algorithm (options are "skip-gram" or "CBOW");
    :param n_walks: maximum number of walks per entity;
    :param domains: list of semantic aspects (e.g., ["biological_process", "molecular_function", "cellular_component"]);
    :return: writes an embedding file for each semantic aspect with format "{ent1:[...], ent2:[...]}"
    """
    for domain in domains:
        g_domain = build_KG_domain(ontology_file_path, annotations_file_path, domain)
        for dataset_name, path_dataset, path_output_embeddings in list_dataset:
            dict_labels, prots = process_dataset(path_dataset)
            calculate_embeddings(g_domain, prots, path_output_embeddings + domain + '/', vector_sizes, types_walk, types_word2vec, n_walks, domain, dataset_name)



#############################
##    Calling Functions    ##
#############################

if __name__ == "__main__":

    n_arguments = len(sys.argv)

    if n_arguments == 1:
        domains = ["biological_process", "molecular_function", "cellular_component"]
        annot_files = [("fly", ["STRING_DM"] ), ( "ecoli", ["STRING_EC"]), ("yeast", ["STRING_SC", "DIPMIPS_SC", "BIND_SC"]), ("human", ["DIP_HS", "STRING_HS", "GRIDHPRD_bal_HS", "GRIDHPRD_unbal_HS"])]
        ontology_file_path = "Data/GOdata/go.owl"
        vector_sizes = 200
        n_walks = 500
        types_walk = "wl"
        types_word2vec = "skip-gram"

        for species, name_datasets in annot_files:
            annotations_file_path = 'Data/GOdata/goa_' + species + '_20.gaf'
            list_dataset = []
            for dataset_name in name_datasets:
                path_dataset = "Data/PPIdatasets/" + dataset_name + "/" + dataset_name + ".txt"
                path_output_embeddings = "SS_Embedding_Calculation/Embeddings/" + dataset_name + "/"
                list_dataset.append((dataset_name, path_dataset, path_output_embeddings))    
            run_RDF2Vec_GOembedddings(ontology_file_path, annotations_file_path, list_dataset, vector_sizes, types_walk, types_word2vec, n_walks, domains)

    else:
        domains = ["biological_process", "molecular_function", "cellular_component"]
        ontology_file_path = sys.argv[1]
        annotations_file_path = sys.argv[2]
        dataset_name = sys.argv[3]
        path_dataset = sys.argv[4]
        path_output_embeddings = sys.argv[5]
        vector_sizes = int(sys.argv[6])
        types_walk = sys.argv[7]
        types_word2vec = sys.argv[8]
        n_walks = int(sys.argv[9])
         
        list_dataset = [(dataset_name, path_dataset, path_output_embeddings)]
        run_RDF2Vec_GOembedddings(ontology_file_path, annotations_file_path, list_dataset, vector_sizes, types_walk, types_word2vec, n_walks, domains)
