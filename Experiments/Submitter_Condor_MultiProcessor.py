#############################################################################
#
# Submitter_Condor_MultiProcessor.py
#
# Program to be run by the Submitter_Condor_Bash_MP to run the executor several time in the right environment.
#
# Author -- Maxence Draguet (26/05/2020) heavily influenced by Aaron O'Neill.
#
#############################################################################
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import multiprocessing as mp
from DataLoaders import make_df, event_cleaning, run_uproot, save_junipr_data_to_json
from Utils import Specific_Set4_Parameters

if __name__ == "__main__":
    """
    # An example of files
    list_files = ["/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/small_set_test_mp/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_JETM6.e6337_e5984_s3126_r10201_r10210_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/small_set_test_mp/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhadVERSION2.deriv.DAOD_JETM6.e6337_e5984_s3126_r10201_r10210_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/small_set_test_mp/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhadVERSION3.deriv.DAOD_JETM6.e6337_e5984_s3126_r10201_r10210_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/small_set_test_mp/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhadVERSION4.deriv.DAOD_JETM6.e6337_e5984_s3126_r10201_r10210_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/small_set_test_mp/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhadVERSION5.deriv.DAOD_JETM6.e6337_e5984_s3126_r10201_r10210_p4128.root"]
                  
    # The following two are used to process the ttbar part II in parallel
    list_files = ["/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_part2_quark_ESUB1gev/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH1.deriv.DAOD_JETM6.e6337_e5984_s3126_r9364_r9315_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_part2_quark_ESUB1gev/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH2.deriv.DAOD_JETM6.e6337_e5984_s3126_r9364_r9315_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_part2_quark_ESUB1gev/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_BATCH3.deriv.DAOD_JETM6.e6337_e5984_s3126_r9364_r9315_p4128.root"]


    list_save_path = ["/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1",
                      "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1",
                      "/data/atlas/atlasdata3/mdraguet/Set4_ttbar_Part2_esub1gev/batch1"]
    """
    # This is to process the dijet in //.
    list_files = ["/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_gluon_ESUB1gev/mc16_13TeV.364700.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_gluon_ESUB1gev/mc16_13TeV.364701.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_gluon_ESUB1gev/mc16_13TeV.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_gluon_ESUB1gev/mc16_13TeV.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_gluon_ESUB1gev/mc16_13TeV.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_gluon_ESUB1gev/mc16_13TeV.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_gluon_ESUB1gev/mc16_13TeV.364706.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_gluon_ESUB1gev/mc16_13TeV.364707.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_gluon_ESUB1gev/mc16_13TeV.364709.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_gluon_ESUB1gev/mc16_13TeV.364710.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_gluon_ESUB1gev/mc16_13TeV.364711.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128.root",
                  "/data/atlas/atlasdata3/mdraguet/RawProcessedGranularData/junipr_gluon_ESUB1gev/mc16_13TeV.364712.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW.deriv.DAOD_JETM6.e7142_e5984_s3126_r9364_r9315_p4128.root"]
                  
    list_save_path = ["/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/",
                      "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/",
                      "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/",
                      "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/",
                      "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/",
                      "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/",
                      "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/",
                      "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/",
                      "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/",
                      "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/",
                      "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/",
                      "/data/atlas/atlasdata3/mdraguet/Set4_dijet_esub1gev/",
                      ]
    
    print("Number of CPU's available : ", mp.cpu_count())
    """
    print("Going parallel with POOL")
    starttime = time.time()
    pool = mp.Pool()
    pool.starmap(run_uproot, zip(list_files, list_save_path))
    pool.close()
    pool.join()
    print('Time taken = {} seconds'.format(time.time() - starttime))
    """
    """
    print("Going sequential")
    starttime = time.time()
    run_uproot(list_files[0], list_save_path[0])
    run_uproot(list_files[1], list_save_path[1])
    run_uproot(list_files[2], list_save_path[2])
    run_uproot(list_files[3], list_save_path[3])
    run_uproot(list_files[4], list_save_path[4])
    print('Time taken = {} seconds'.format(time.time() - starttime))
    """
    
    print("Going parallel with PROCESS")
    starttime = time.time()
    processes = []
    for i in range(len(list_files)):
        p = mp.Process(target=run_uproot, args=(list_files[i], list_save_path[i]))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print('Time taken = {} seconds'.format(time.time() - starttime))

