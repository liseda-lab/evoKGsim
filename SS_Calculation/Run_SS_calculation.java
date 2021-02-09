package SS_Calculation;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;

import org.openrdf.model.URI;
import slib.graph.algo.extraction.rvf.instances.InstancesAccessor;
import slib.graph.algo.extraction.rvf.instances.impl.InstanceAccessor_RDF_TYPE;
import slib.graph.algo.utils.GAction;
import slib.graph.algo.utils.GActionType;
import slib.graph.algo.utils.GraphActionExecutor;
import slib.graph.io.conf.GDataConf;
import slib.graph.io.loader.GraphLoaderGeneric;
import slib.graph.io.util.GFormat;
import slib.graph.model.graph.G;
import slib.graph.model.impl.graph.memory.GraphMemory;
import slib.graph.model.impl.repo.URIFactoryMemory;
import slib.graph.model.repo.URIFactory;
import slib.sml.sm.core.engine.SM_Engine;
import slib.sml.sm.core.metrics.ic.utils.IC_Conf_Corpus;
import slib.sml.sm.core.metrics.ic.utils.IC_Conf_Topo;
import slib.sml.sm.core.metrics.ic.utils.ICconf;
import slib.sml.sm.core.utils.SMConstants;
import slib.sml.sm.core.utils.SMconf;
import slib.utils.ex.SLIB_Ex_Critic;
import slib.utils.ex.SLIB_Exception;
import slib.utils.impl.Timer;

/*
This class is responsible for convertng GAF 2.1 to GAF 2.0, the format accepted by SML
 */
class Convert_GAF_versions {

    // 2 arguments: annot and new_annot
    // annot is the file annotations path in GAF.2.1 version
    // new_annot is the new file annotations path in GAF 2.0 version
    public String annot;
    public String new_annot;

    public Convert_GAF_versions(String arg1, String arg2){
        annot = arg1;
        new_annot = arg2;
    }


    public void run() throws FileNotFoundException , IOException {

        PrintWriter new_file = new PrintWriter(new_annot);
        new_file.println("!gaf-version: 2.0");

        // Open the file with annotations
        FileInputStream file_annot = new FileInputStream(annot);
        BufferedReader br = new BufferedReader(new InputStreamReader(file_annot));

        String strLine;


        // Read file line by line
        while ((strLine = br.readLine()) != null){

            if (!strLine.startsWith("!") || !strLine.isEmpty() || strLine != null  || strLine!= ""){

                ArrayList<String> fields = new ArrayList<String>(Arrays.asList(strLine.split("\t")));

                if (fields.size()>12){ // verify if the annotation have taxon
                    fields.set(7 , fields.get(7).replace("," , "|"));
                    String newLine = String.join("\t" , fields);
                    new_file.println(newLine);
                }
            }
        }

        new_file.close();
        file_annot.close();


    }

}

/*
This class is responsible for calculating SSM for a list of protein pairs files of the same species (using same GAF file)
 */
class Calculate_sim_prot {

    // 4 arguments: path_file_goOBO , annot , SSM_files and dataset_files.
    // path_file_goOBO is the ontology file path
    // annot is the annotations file path in GAF 2.0 version
    // datasets_files is the list of dataset files path with the pairs of proteins. The format of each line of the dataset files is "Prot1  Prot2   Label"
    // SSM_files is the list of semantic similarity files paths for each element of the dasets_files
    public String path_file_goOBO;
    public String annot;
    public String[] SSM_files;
    public String[] dataset_files;

    public Calculate_sim_prot(String arg1, String arg2, String[] arg3, String[] arg4) {
        path_file_goOBO = arg1;
        annot = arg2;
        SSM_files = arg3;
        dataset_files = arg4;
    }

    public void run() throws SLIB_Exception, FileNotFoundException, IOException {

        Timer t = new Timer();
        t.start();

        URIFactory factory = URIFactoryMemory.getSingleton();
        URI graph_uri = factory.getURI("http://");

        // define a prefix in order to build valid uris from ids such as GO:XXXXX
        // (the URI associated to GO:XXXXX will be http://go/XXXXX)
        factory.loadNamespacePrefix("GO", graph_uri.toString());

        G graph_BP = new GraphMemory(graph_uri);
        G graph_CC = new GraphMemory(graph_uri);
        G graph_MF = new GraphMemory(graph_uri);

        GDataConf goConf = new GDataConf(GFormat.OBO, path_file_goOBO);
        GDataConf annotConf = new GDataConf(GFormat.GAF2, annot);

        GraphLoaderGeneric.populate(goConf, graph_BP);
        GraphLoaderGeneric.populate(goConf, graph_CC);
        GraphLoaderGeneric.populate(goConf, graph_MF);

        GraphLoaderGeneric.populate(annotConf, graph_BP);
        GraphLoaderGeneric.populate(annotConf, graph_CC);
        GraphLoaderGeneric.populate(annotConf, graph_MF);

        URI bpGOTerm = factory.getURI("http://0008150");
        GAction reduction_bp = new GAction(GActionType.VERTICES_REDUCTION);
        reduction_bp.addParameter("root_uri", bpGOTerm.stringValue());
        GraphActionExecutor.applyAction(factory, reduction_bp, graph_BP);

        URI ccGOTerm = factory.getURI("http://0005575");
        GAction reduction_cc = new GAction(GActionType.VERTICES_REDUCTION);
        reduction_cc.addParameter("root_uri", ccGOTerm.stringValue());
        GraphActionExecutor.applyAction(factory, reduction_cc, graph_CC);

        URI mfGOTerm = factory.getURI("http://0003674");
        GAction reduction_mf = new GAction(GActionType.VERTICES_REDUCTION);
        reduction_mf.addParameter("root_uri", mfGOTerm.stringValue());
        GraphActionExecutor.applyAction(factory, reduction_mf, graph_MF);

        int i = 0;
        for (String dataset_filename : dataset_files) {
            ArrayList<String> pair_prots = get_proteins_dataset(dataset_filename);
            semantic_measures_2prots(graph_BP, graph_CC, graph_MF, factory, pair_prots, SSM_files[i]);
            i++;
        }


        t.stop();
        t.elapsedTime();
    }

    private ArrayList<String> get_proteins_dataset(String dataset_filename) throws IOException {

        FileInputStream file_dataset = new FileInputStream(dataset_filename);
        BufferedReader br = new BufferedReader(new InputStreamReader(file_dataset));

        ArrayList<String> pairs_prots = new ArrayList<>();
        String strLine;

        // Read file line by line
        while ((strLine = br.readLine()) != null) {
            strLine = strLine.substring(0, strLine.length() - 1);
            pairs_prots.add(strLine);
        }

        file_dataset.close();
        return pairs_prots;


    }

    private void groupwise_measure_file(G graph_BP, G graph_CC, G graph_MF, SM_Engine engine_bp, SM_Engine engine_cc, SM_Engine engine_mf, URIFactory factory, ArrayList<String> pairs_prots, String SSM_file, SMconf ssm) throws SLIB_Ex_Critic, FileNotFoundException {

        double sim_BP, sim_CC, sim_MF;

        PrintWriter file = new PrintWriter(SSM_file);
        //file.print("prot1   prot2   sim_BP_" + SSM + "   sim_CC_" + SSM + "   sim_MF_" + SSM +  "\n");

        for (String pair : pairs_prots) {
            ArrayList<String> proteins = new ArrayList<String>(Arrays.asList(pair.split("\t")));
            String uri_prot1 = "http://" + proteins.get(0);
            String uri_prot2 = "http://" + proteins.get(1);

            URI instance1 = factory.getURI(uri_prot1);
            URI instance2 = factory.getURI(uri_prot2);

            if (((graph_BP.containsVertex(instance1)) || (graph_CC.containsVertex(instance1)) || ((graph_MF.containsVertex(instance1)))) &&
                    ((graph_BP.containsVertex(instance2)) || (graph_CC.containsVertex(instance2)) || (graph_MF.containsVertex(instance2)))) {
                InstancesAccessor gene_instance1_acessor_bp = new InstanceAccessor_RDF_TYPE(graph_BP);
                Set<URI> annotations_instance1_BP = gene_instance1_acessor_bp.getDirectClass(instance1);

                InstancesAccessor gene_instance1_acessor_cc = new InstanceAccessor_RDF_TYPE(graph_CC);
                Set<URI> annotations_instance1_CC = gene_instance1_acessor_cc.getDirectClass(instance1);

                InstancesAccessor gene_instance1_acessor_mf = new InstanceAccessor_RDF_TYPE(graph_MF);
                Set<URI> annotations_instance1_MF = gene_instance1_acessor_mf.getDirectClass(instance1);


                InstancesAccessor gene_instance2_acessor_bp = new InstanceAccessor_RDF_TYPE(graph_BP);
                Set<URI> annotations_instance2_BP = gene_instance2_acessor_bp.getDirectClass(instance2);

                InstancesAccessor gene_instance2_acessor_cc = new InstanceAccessor_RDF_TYPE(graph_CC);
                Set<URI> annotations_instance2_CC = gene_instance2_acessor_cc.getDirectClass(instance2);

                InstancesAccessor gene_instance2_acessor_mf = new InstanceAccessor_RDF_TYPE(graph_MF);
                Set<URI> annotations_instance2_MF = gene_instance2_acessor_mf.getDirectClass(instance2);

                if (instance1.equals(instance2)) {
                    sim_BP = sim_CC = sim_MF = 1;


                } else {
                    if (annotations_instance1_BP.isEmpty() || annotations_instance2_BP.isEmpty()) {
                        sim_BP = 0;
                    } else {
                        sim_BP = engine_bp.compare(ssm, annotations_instance1_BP, annotations_instance2_BP);
                    }

                    if (annotations_instance1_CC.isEmpty() || annotations_instance2_CC.isEmpty()) {
                        sim_CC = 0;
                    } else {
                        sim_CC = engine_cc.compare(ssm, annotations_instance1_CC, annotations_instance2_CC);
                    }

                    if (annotations_instance1_MF.isEmpty() || annotations_instance2_MF.isEmpty()) {
                        sim_MF = 0;
                    } else {
                        sim_MF = engine_mf.compare(ssm, annotations_instance1_MF, annotations_instance2_MF);
                    }
                }

                file.print(instance1 + "\t" + instance2 + "\t" + sim_BP + "\t" + sim_CC + "\t" + sim_MF + "\n");
            }
        }

        file.close();

    }

    private void pairwise_measure_file(G graph_BP, G graph_CC, G graph_MF, SM_Engine engine_bp, SM_Engine engine_cc, SM_Engine engine_mf, URIFactory factory, ArrayList<String> pairs_prots, String SSM_file, SMconf ssm, SMconf aggregation) throws SLIB_Ex_Critic, FileNotFoundException {

        double sim_BP, sim_CC, sim_MF;

        PrintWriter file = new PrintWriter(SSM_file);
        //file.print("prot1   prot2   sim_BP_" + SSM + "   sim_CC_" + SSM + "   sim_MF_" + SSM +  "\n");

        for (String pair : pairs_prots) {
            ArrayList<String> proteins = new ArrayList<String>(Arrays.asList(pair.split("\t")));
            String uri_prot1 = "http://" + proteins.get(0);
            String uri_prot2 = "http://" + proteins.get(1);

            URI instance1 = factory.getURI(uri_prot1);
            URI instance2 = factory.getURI(uri_prot2);

            if (((graph_BP.containsVertex(instance1)) || (graph_CC.containsVertex(instance1)) || ((graph_MF.containsVertex(instance1)))) &&
                    ((graph_BP.containsVertex(instance2)) || (graph_CC.containsVertex(instance2)) || (graph_MF.containsVertex(instance2)))) {
                InstancesAccessor gene_instance1_acessor_bp = new InstanceAccessor_RDF_TYPE(graph_BP);
                Set<URI> annotations_instance1_BP = gene_instance1_acessor_bp.getDirectClass(instance1);

                InstancesAccessor gene_instance1_acessor_cc = new InstanceAccessor_RDF_TYPE(graph_CC);
                Set<URI> annotations_instance1_CC = gene_instance1_acessor_cc.getDirectClass(instance1);

                InstancesAccessor gene_instance1_acessor_mf = new InstanceAccessor_RDF_TYPE(graph_MF);
                Set<URI> annotations_instance1_MF = gene_instance1_acessor_mf.getDirectClass(instance1);

                InstancesAccessor gene_instance2_acessor_bp = new InstanceAccessor_RDF_TYPE(graph_BP);
                Set<URI> annotations_instance2_BP = gene_instance2_acessor_bp.getDirectClass(instance2);

                InstancesAccessor gene_instance2_acessor_cc = new InstanceAccessor_RDF_TYPE(graph_CC);
                Set<URI> annotations_instance2_CC = gene_instance2_acessor_cc.getDirectClass(instance2);

                InstancesAccessor gene_instance2_acessor_mf = new InstanceAccessor_RDF_TYPE(graph_MF);
                Set<URI> annotations_instance2_MF = gene_instance2_acessor_mf.getDirectClass(instance2);

                if (instance1.equals(instance2)) {
                    sim_BP = sim_CC = sim_MF = 1;


                } else {
                    if (annotations_instance1_BP.isEmpty() || annotations_instance2_BP.isEmpty()) {
                        sim_BP =  0;
                    } else {
                        sim_BP = engine_bp.compare(aggregation, ssm, annotations_instance1_BP, annotations_instance2_BP);
                    }

                    if (annotations_instance1_CC.isEmpty() || annotations_instance2_CC.isEmpty()) {
                        sim_CC = 0;
                    } else {
                        sim_CC = engine_cc.compare(aggregation, ssm, annotations_instance1_CC, annotations_instance2_CC);
                    }

                    if (annotations_instance1_MF.isEmpty() || annotations_instance2_MF.isEmpty()) {
                        sim_MF =  0;
                    } else {
                        sim_MF = engine_mf.compare(aggregation, ssm, annotations_instance1_MF, annotations_instance2_MF);
                    }
                }

                file.print(instance1 + "\t" + instance2 + "\t" + sim_BP + "\t" + sim_CC + "\t" + sim_MF + "\n");
            }
        }

        file.close();

    }

    private void semantic_measures_2prots(G graph_BP, G graph_CC, G graph_MF, URIFactory factory, ArrayList<String> pairs_prots, String SSM_file) throws SLIB_Ex_Critic, FileNotFoundException {

        ICconf ic_Seco = new IC_Conf_Topo("Seco", SMConstants.FLAG_ICI_SECO_2004);
        ICconf ic_Resnik = new IC_Conf_Corpus("resnik", SMConstants.FLAG_IC_ANNOT_RESNIK_1995_NORMALIZED);

        SMconf SimGIC_icSeco = new SMconf("gic", SMConstants.FLAG_SIM_GROUPWISE_DAG_GIC);
        SimGIC_icSeco.setICconf(ic_Seco);

        SMconf SimGIC_icResnik = new SMconf("gic", SMConstants.FLAG_SIM_GROUPWISE_DAG_GIC);
        SimGIC_icResnik.setICconf(ic_Resnik);

        SMconf Resnik_icSeco = new SMconf("resnik", SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_RESNIK_1995);
        Resnik_icSeco.setICconf(ic_Seco);

        SMconf Resnik_icResnik = new SMconf("resnik", SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_RESNIK_1995);
        Resnik_icResnik.setICconf(ic_Resnik);

        SMconf max = new SMconf("max", SMConstants.FLAG_SIM_GROUPWISE_MAX);
        SMconf bma = new SMconf("bma", SMConstants.FLAG_SIM_GROUPWISE_BMA);

        SM_Engine engine_bp = new SM_Engine(graph_BP);
        SM_Engine engine_cc = new SM_Engine(graph_CC);
        SM_Engine engine_mf = new SM_Engine(graph_MF);


        ArrayList<String> pre_filename = new ArrayList<String>(Arrays.asList(SSM_file.split("/ss_")));
        File theDir = new File(pre_filename.get(0));
        if (!theDir.exists()) {
            theDir.mkdirs();
        }

        String filename_SimGIC_icSeco = pre_filename.get(0) + "/ss_simGIC_ICSeco" + "_" + pre_filename.get(1);
        groupwise_measure_file(graph_BP, graph_CC, graph_MF, engine_bp, engine_cc, engine_mf, factory, pairs_prots, filename_SimGIC_icSeco, SimGIC_icSeco);
        String filename_SimGIC_icResnik = pre_filename.get(0) + "/ss_simGIC_ICResnik" + "_" + pre_filename.get(1);
        groupwise_measure_file(graph_BP, graph_CC, graph_MF, engine_bp, engine_cc, engine_mf, factory, pairs_prots, filename_SimGIC_icResnik, SimGIC_icResnik);

        String filename_ResnikMax_icSeco = pre_filename.get(0) + "/ss_ResnikMax_ICSeco" + "_" + pre_filename.get(1);
        pairwise_measure_file(graph_BP, graph_CC, graph_MF, engine_bp, engine_cc, engine_mf, factory, pairs_prots, filename_ResnikMax_icSeco, Resnik_icSeco, max);
        String filename_ResnikMax_icResnik = pre_filename.get(0) + "/ss_ResnikMax_ICResnik" + "_" + pre_filename.get(1);
        pairwise_measure_file(graph_BP, graph_CC, graph_MF, engine_bp, engine_cc, engine_mf, factory, pairs_prots, filename_ResnikMax_icResnik, Resnik_icResnik, max);

        String filename_ResnikBMA_icSeco = pre_filename.get(0) + "/ss_ResnikBMA_ICSeco" + "_" + pre_filename.get(1);
        pairwise_measure_file(graph_BP, graph_CC, graph_MF, engine_bp, engine_cc, engine_mf, factory, pairs_prots, filename_ResnikBMA_icSeco, Resnik_icSeco, bma);
        String filename_ResnikBMA_icResnik = pre_filename.get(0) + "/ss_ResnikBMA_ICResnik" + "_" + pre_filename.get(1);
        pairwise_measure_file(graph_BP, graph_CC, graph_MF, engine_bp, engine_cc, engine_mf, factory, pairs_prots, filename_ResnikBMA_icResnik, Resnik_icResnik, bma);

    }
}

public class Run_SS_calculation {

    public static void main(String[] args) throws Exception {

        // The implementation of SML requires a annotation file in GAF 2.0. Since the most recent GO annotation file is in GAF 2.1 format, it was converted to the older format specifications.
        Convert_GAF_versions human_annot = new Convert_GAF_versions("Data/GOdata/goa_human.gaf", "Data/GOdata/goa_human_20.gaf");
        human_annot.run();
        // Calculate the SS for human datasets
        Calculate_sim_prot human_datasets = new Calculate_sim_prot("Data/GOdata/go.obo", "Data/GOdata/goa_human_20.gaf",
                new String[]{"SS_Calculation/SS_files/DIP_HS/ss_DIP_HS.txt",
                        "SS_Calculation/SS_files/STRING_HS/ss_STRING_HS.txt",
                        "SS_Calculation/SS_files/GRIDHPRD_bal_HS/ss_GRIDHPRD_bal_HS.txt",
                        "SS_Calculation/SS_files/GRIDHPRD_unbal_HS/ss_GRIDHPRD_unbal_HS.txt"},
                new String[]{"Data/PPIdatasets/DIP_HS/DIP_HS.txt",
                        "Data/PPIdatasets/STRING_HS/STRING_HS.txt",
                        "Data/PPIdatasets/GRIDHPRD_bal_HS/GRIDHPRD_bal_HS.txt",
                        "Data/PPIdatasets/GRIDHPRD_unbal_HS/GRIDHPRD_unbal_HS.txt"});
        human_datasets.run();
        System.out.println("---------------------------------------------------------------");
        System.out.println("SSM for human completed.");
        System.out.println("---------------------------------------------------------------");

        // Calculate the SS for E.coli dataset
        Convert_GAF_versions ecoli_annot = new Convert_GAF_versions("Data/GOdata/ecocyc.gaf", "Data/GOdata/goa_ecoli_20.gaf");
        ecoli_annot.run();
        Calculate_sim_prot ecoli_datasets = new Calculate_sim_prot("Data/GOdata/go.obo", "Data/GOdata/goa_ecoli_20.gaf",
                new String[]{"SS_Calculation/SS_files/STRING_EC/ss_STRING_EC.txt"},
                new String[]{"Data/PPIdatasets/STRING_EC/STRING_EC.txt"});
        ecoli_datasets.run();
        System.out.println("---------------------------------------------------------------");
        System.out.println("SSM for E. coli completed.");
        System.out.println("---------------------------------------------------------------");

        // Calculate the SS for fly dataset
        Convert_GAF_versions fly_annot = new Convert_GAF_versions("Data/GOdata/goa_fly.gaf", "Data/GOdata/goa_fly_20.gaf");
        fly_annot.run();
        Calculate_sim_prot fly_datasets = new Calculate_sim_prot("Data/GOdata/go.obo", "Data/GOdata/goa_fly_20.gaf",
                new String[]{"SS_Calculation/SS_files/STRING_DM/ss_STRING_DM.txt"},
                new String[]{"Data/PPIdatasets/STRING_DM/STRING_DM.txt"});
        fly_datasets.run();
        System.out.println("---------------------------------------------------------------");
        System.out.println("SSM for D. melanogaster completed.");
        System.out.println("---------------------------------------------------------------");

        // Calculate the SS for yeast datasets
        Convert_GAF_versions yeast_annot = new Convert_GAF_versions("Data/GOdata/goa_yeast.gaf", "Data/GOdata/goa_yeast_20.gaf");
        yeast_annot.run();
        Calculate_sim_prot yeast_datasets = new Calculate_sim_prot("Data/GOdata/go.obo", "Data/GOdata/goa_yeast_20.gaf",
                new String[]{"SS_Calculation/SS_files/STRING_SC/ss_STRING_SC.txt",
                        "SS_Calculation/SS_files/DIPMIPS_SC/ss_DIPMIPS_SC.txt",
                        "SS_Calculation/SS_files/BIND_SC/ss_BIND_SC.txt"},
                new String[]{"Data/PPIdatasets/STRING_SC/STRING_SC.txt",
                        "Data/PPIdatasets/DIPMIPS_SC/DIPMIPS_SC.txt",
                        "Data/PPIdatasets/BIND_SC/BIND_SC.txt"});
        yeast_datasets.run();
        System.out.println("---------------------------------------------------------------");
        System.out.println("SSM for S. cerevisiae completed.");
        System.out.println("---------------------------------------------------------------");

    }

}
