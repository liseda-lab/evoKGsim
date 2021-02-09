# evoKGsim

**SS**: Taxonomic Semantic Similarity; **ES**: Embedding Semantic Similarity; **SSM**: Semantic Similarity Measure; **GP**: Genetic Programming; **GO**: Gene Ontology; **PPI**: Protein-Protein Interaction.

This repository provides a implementationdescribed in the paper:
```
Evolving knowledge graph similarity for supervised learning in complex biomedical domains
Rita T. Sousa, Sara Silva, and Catia Pesquita
BMC bioinformatics 2020 21(1):1-19. doi: 10.1186/s12859-019-3296-1
```


## Pre-requesites
* install python 3.6.8;
* install java JDK 11.0.4;
* install python libraries by running the following command:  ```pip install -r req.txt```.



## 1. Benchmark Datasets
For the program to work, provide a text file with the protein pairs and respective labels (interact or non-interact). 
This tab-delimited text file have 3 columns: 
* 1st column - Protein1 UniProt Identifier;	 
* 2nd column - Protein2 UniProt Identifier;
* 3rd column - Label (the options are 1 for interacting pairs and 0 for non-interacting pairs). 

In this work, we used 9 Benchmark datasets (STRING-SC, STRING-HS, STRING-EC, STRING-DM, DIP-HS, BIND-SC, DIP/MIPS-SC, GRID/HPRD-bal-HS, and GRID/HPRD-unbal-HS) of different species for evaluation. The data is in [Data/PPIdatasets](https://github.com/ritatsousa/evoKGsim/tree/master/Data/PPIdatasets) folder.



## 2. Taxonomic Semantic Similarity Computation

For taxonomic semantic similarity calculation, provide:
* Dataset file with the previously described format;
* Ontology file in OWL format;
* Annotations file in 2.0. or 2.1. GAF format (http://geneontology.org/docs/go-annotation-file-gaf-format-2.0/). GAFs are tab-delimited plain text files, where each line in the file represents a single association between a gene product and a GO term. 

To support SS calculations, SML was employed:
```
The Semantic Measures Library and Toolkit: fast computation of semantic similarity and relatedness using biomedical ontologies
Sébastien Harispe*, Sylvie Ranwez, Stefan Janaqi and Jacky Montmain
Bioinformatics 2014 30(5): 740-742. doi: 10.1093/bioinformatics/btt581
```
The software is available on GitHub (https://github.com/sharispe/slib/tree/dev/slib-sml) under a CeCILL License.

In Linux, compile the command:
```
javac -cp ".:./SS_Calculation/jar_files/*" ./SS_Calculation/Run_SS_calculation.java
```
and then run
```
java -cp ".:./SS_Calculation/jar_files/*" SS_Calculation/Run_SS_calculation
```

This command will create, for each dataset, **SS files** (one for each SSM) with the SS between each pair of proteins for each semantic aspect (biological process, cellular component and molecular function) using six different SSMs (ResnikMax_ICSeco, ResnikMax_ICResnik, ResnikBMA_ICSeco, ResnikBMA_ICResnik, simGIC_ICSeco, simGIC_ICResnik). The description of this text file is in [SS_Calculation/SS_files/SS_file_ format.txt](https://github.com/ritatsousa/evoKGsim/blob/master/SS_Calculation/SS_files/SS_file_%20format.txt) file. 
The new SS files are placed in [SS_Calculation/SS_files/datasetname](https://github.com/ritatsousa/evoKGsim/tree/master/SS_Calculation/SS_files) folder. 



## 3. Embedding Semantic Similarity Computation


### 3.1. Compute RDF2Vec Embeddings for each GO semantic aspect

An RDF2Vec python implementation was used to calculate graph embedding.  
```
RDF2Vec: RDF graph embeddings for data mining
Petara Ristoski and Heiko Paulheim
International Semantic Web Conference, Springer, Cham, 2016 (pp. 498-514)
```
The implementation is available on GitHub https://github.com/IBCNServices/pyRDF2Vec.

In RDF2Vec, a set of sequences was generated from Weisfeiler-Lehman subtree kernels.
For the Weisfeiler-Lehman algorithm, we use walks with depth 8, and we extracted a limited number of 500 random walks for each protein. The corpora of sequences were used to build a Skip-Gram model with the following parameters: window size=5; number of iterations=10; entity vector size=200.

Run the command to calculate the embeddings for each protein using rdf2vec implementation:
```
python3 SS_Embedding_Calculation/run_RDF2VecEmbeddings_PPI.py
```
For each dataset:
* This command creates **3 embedding files** (one for each GO semantic apect: biological_process, cellular_component aspect, molecular_function) and place them in [SS_Embedding_Calculation/Embeddings/datasetname/aspect] (https://github.com/ritatsousa/evoKGsim/tree/master/SS_Embedding_Calculation/Embeddings) folder.
The filename is in the format “Embeddings_datasetname_skig-gram_wl_aspect.txt”. 
The description of this text file is in [SS_Embedding_Calculation/Embeddings/Embeddings_format.txt](https://github.com/ritatsousa/evoKGsim/blob/master/SS_Embedding_Calculation/Embeddings/Embeddings_format.txt) file.


### 3.2. Compute OpenKE Embeddings for each GO semantic aspect 

OpenKE is only implemented for the Linux system.

Run the command to calculate the embeddings for each protein using OpenKE implementation for 6 embedding methods (TransE, TransH, TransD, TransR, distMult, ComplEx):
```
python3 SS_Embedding_Calculation/run_model_PPI.py
```
For each dataset:
* For each embedding method, this command creates **3 embedding files** (one for each GO semantic apect: biological_process, cellular_component aspect, molecular_function) and place them in [SS_Embedding_Calculation/Embeddings/datasetname/aspect] (https://github.com/ritatsousa/evoKGsim/tree/master/SS_Embedding_Calculation/Embeddings) folder.
The filename is in the format “Embeddings_datasetname_method_aspect.txt”. 
The description of this text file is in [SS_Embedding_Calculation/Embeddings/Embeddings_format.txt](https://github.com/ritatsousa/evoKGsim/blob/master/SS_Embedding_Calculation/Embeddings/Embeddings_format.txt) file.


### 3.3. Compute the Embedding Semantic Similarity for each pair

After generating embeddings for each semantic aspect and then calculated the cosine similarity for each pair
in datasets.
Run the command for calculating embedding similarity for each semantic aspect (biological process ES_BP, cellular component ES_CC and molecular function ES_MF):
```
python3 SS_Embedding_Calculation/run_embedSS_calculation.py
```
For each dataset:
* This command creates **1 embedding similarity file** and places it in [SS_Embedding_Calculation/Embeddings_SS_files] (https://github.com/ritatsousa/evoKGsim/tree/master/SS_Embedding_Calculation/Embeddings_SS_files) folder.
The filename is in the format "embedss_200_model_datasetname.txt". 
The format of each line of embedding similarity file is "Prot1  Prot2	ES_BP	ES_CC	ES_MF"; 



## 4. Evolve combinations of semantic aspects
For 10-cross-validation purposes, run the command to split each dataset into ten partitions:
```
python3 Prediction/run_make_shuffle_partitions.py
```
This command will create, for each dataset, **10 Partitions files** and place them in [Prediction/Results/Datasetname/Shuffle_Partitions](https://github.com/ritatsousa/evoKGsim/tree/master/Prediction/Results) folder. Each line of these files is an index (corresponding to a protein pair) of the dataset. This folder is already created, so you do not have to change any folder path.

With semantic similarities, run the command for PPI prediction using evolved combinations:
```
python3 Prediction/run_withPartitions_evoKGsim.py
```
The parameters we have set are listed in the next Table. All others were used with the default values of the gplearn software. 

| Parameter   |  Value  |
| ------------------- | ------------------- |
|  Number of generations |  50 |
|  Size of population | 500 |
|  Function set |  +,-,/,x,max,min |
|  Fitness function |  RMSE |
|  Parsimony coeffcient |  0.00001 |

For running the baselines (static combinations of semantic aspects), run the command:
```
python3 Prediction/run_withPartitions_evoKGsim_SS.py False True
```
