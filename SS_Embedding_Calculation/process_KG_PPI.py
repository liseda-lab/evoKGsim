
import rdflib 
from rdflib.namespace import RDF, OWL


def process_dataset(file_dataset_path):
    """
    Process a dataset file.
    :param file_dataset_path: dataset file path with the correspondent entity pairs. The format of each line of the dataset files is "Ent1 Ent2 Proxy";
    :return: one dictionary and one list. "dict_labels" is a dictionary with entity pairs and respective similarity proxy. "prots" is a list of entities for which embeddings will be computed.
    """

    dataset = open(file_dataset_path, 'r')
    dict_labels = {}
    prots =[]

    for line in dataset:
        split1 = line.split('\t')
        prot1, prot2 = split1[0], split1[1]
        label = int(split1[-1][:-1])

        url_prot1 = "http://" + prot1
        url_prot2 = "http://" + prot2

        dict_labels[(url_prot1, url_prot2)] = label

        if url_prot1 not in prots:
            prots.append(url_prot1)

        if url_prot2 not in prots:
            prots.append(url_prot2)

    dataset.close()
    return dict_labels, prots



def build_KG_domain(ontology_file_path, annotations_file_path, domain):
    """
    Builds the KG for a GO semantic aspect.
    :param ontology_file_path: GO ontology file path in owl format;
    :param annotations_file_path: GOA annotations file path in GAF 2.1 version;
    :param domain: semantic aspect. Domain can be "molecular_function", "biological_process", "cellular_component";
    :return: a KG for a GO semantic aspect.
    """

    g = rdflib.Graph()
    g_domain = rdflib.Graph()

    # Process ontology file
    g.parse(ontology_file_path, format='xml')


    for (sub, pred, obj) in g.triples((None,None, None)):
        if g.__contains__((sub, rdflib.term.URIRef('http://www.geneontology.org/formats/oboInOwl#hasOBONamespace'), rdflib.term.Literal(domain, datatype=rdflib.term.URIRef('http://www.w3.org/2001/XMLSchema#string')))):
            g_domain.add(((sub, pred, obj)))

    file_annot = open(annotations_file_path , 'r')
    file_annot.readline()

    for annot in file_annot:
        list_annot = annot.split('\t')
        id_prot, GO_term = list_annot[1], list_annot[4]

        url_GO_term = "http://purl.obolibrary.org/obo/GO_" + GO_term.split(':')[1]
        url_prot = "http://" + id_prot

        if ((rdflib.term.URIRef(url_GO_term), None, None) in g_domain) or ((None, None, (rdflib.term.URIRef(url_GO_term)))in g_domain):
            g_domain.add((rdflib.term.URIRef(url_prot), rdflib.term.URIRef('http://www.geneontology.org/hasAnnotation') , rdflib.term.URIRef(url_GO_term)))
        else:
            g_domain.add((rdflib.term.URIRef(url_prot), RDF.type, rdflib.BNode()))

    file_annot.close()
    return g_domain



def buildIds(g):
    """
    Assigns ids to KG nodes and KG relations.
    :param g: knowledge graph;
    :return: 2 dictionaries and one list. "dic_nodes" is a dictionary with KG nodes and respective ids. "dic_relations" is a dictionary with type of relations in the KG and respective ids. "list_triples" is a list with triples of the KG.
    """

    dic_nodes = {}
    id_node = 0
    id_relation = 0
    dic_relations = {}
    list_triples = []
    for (subj, predicate , obj) in g:

        if str(subj) not in dic_nodes:
            dic_nodes[str(subj)] = id_node
            id_node = id_node + 1

        if str(obj) not in dic_nodes:
            dic_nodes[str(obj)] = id_node
            id_node = id_node + 1

        if str(predicate) not in dic_relations:
            dic_relations[str(predicate)] = id_relation
            id_relation = id_relation + 1

        list_triples.append([dic_nodes[str(subj)] , dic_relations[str(predicate)] , dic_nodes[str(obj)]])

    return dic_nodes , dic_relations , list_triples
    


