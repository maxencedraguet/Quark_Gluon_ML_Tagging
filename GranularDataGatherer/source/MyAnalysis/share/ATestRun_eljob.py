#!/usr/bin/env python

# Read the submission directory as a command line argument. You can
# extend the list of arguments with your private ones later on.
import optparse
parser = optparse.OptionParser()
parser.add_option( '-s', '--submission-dir', dest = 'submission_dir',
                   action = 'store', type = 'string', default = 'submitDir',
                   help = 'Submission directory for EventLoop' )
( options, args ) = parser.parse_args()

# Set Data type (add this as an automatic function or argument)
# [choose from Data=0, FullSim=1 or AtlfastII=2]
dataType = 0

# Set up (Py)ROOT.
import ROOT
ROOT.xAOD.Init().ignore()

# Set up the sample handler object. See comments from the C++ macro
# for the details about these lines.
import os
sh = ROOT.SH.SampleHandler()
sh.setMetaString( 'nc_tree', 'CollectionTree' )
inputFilePath = "/data/atlas/atlasdata3/oneill/DAOD_JETM6/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_JETM6.e6337_e5984_s3126_r10201_r10210_p4128"
#ROOT.SH.ScanDir().filePattern( 'DAOD_JETM6.20933895._001089.pool.root.1' ).scan( sh, inputFilePath )

#inputFilePath = "/data/atlas/atlasdata3/oneill/DAOD_JETM6/data16_13TeV.periodF.physics_Main.PhysCont.DAOD_JETM6.grp16_v01_p4129"
ROOT.SH.ScanDir().filePattern( '*' ).scan( sh, inputFilePath )
sh.printContent()

# Create an EventLoop job.
job = ROOT.EL.Job()
job.sampleHandler( sh )
job.options().setDouble( ROOT.EL.Job.optMaxEvents, 10 )
job.options().setString( ROOT.EL.Job.optSubmitDirMode, 'unique-link')
job.outputAdd (ROOT.EL.OutputStream ('ANALYSIS'))
# Create the algorithm's configuration.
from AnaAlgorithm.DualUseConfig import createAlgorithm
alg = createAlgorithm ( 'MyxAODAnalysis', 'AnalysisAlg' )

# Add and configure the GRL tool.
from AnaAlgorithm.DualUseConfig import addPrivateTool

# add the GRL tool to the algorithm
addPrivateTool( alg, 'grlTool', 'GoodRunsListSelectionTool' )

# configure the properties of the GRL tool
### TODO add year configurations here (short term just sent all GRL files to tool).
fullGRLFilePath = "GoodRunsLists/data16_13TeV/20180129/data16_13TeV.periodAllYear_DetStatus-v89-pro21-01_DQDefects-00-02-04_PHYS_StandardGRL_All_Good_25ns.xml"
alg.grlTool.GoodRunsListVec = [ fullGRLFilePath ]
alg.grlTool.PassThrough = 0 # if true (default) will ignore result of GRL and will just pass all events

#Add SUSY tools to the algorithm
addPrivateTool( alg, 'SUSYTools', 'ST::SUSYObjDef_xAOD' )
#Set jet type to EMTopo for now (add option to change this).
alg.SUSYTools.JetInputType = 1 # 1 EMTopo and 9 PFlow.
alg.SUSYTools.DataSource = dataType

#Need to pass the PRW lumi calc files and configure the prw tool with STs

# Add our algorithm to the job
job.algsAdd( alg )

# Run the job using the direct driver.
driver = ROOT.EL.DirectDriver()
driver.submit( job, options.submission_dir )
