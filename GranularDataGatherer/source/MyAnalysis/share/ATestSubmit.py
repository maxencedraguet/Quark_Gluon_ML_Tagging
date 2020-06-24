#!/usr/bin/env python
#include <SampleHandler/ToolsDiscovery.h>

# Read the submission directory as a command line argument. You can
# extend the list of arguments with your private ones later on.
import optparse
parser = optparse.OptionParser()
parser.add_option( '-s', '--submission-dir', dest = 'submission_dir',
                  action = 'store', type = 'string', default = 'submitDir',
                  help = 'Submission directory for EventLoop' )
( options, args ) = parser.parse_args()

# Set up (Py)ROOT.
import ROOT
ROOT.xAOD.Init().ignore()

# Set up the sample handler object. See comments from the C++ macro
# for the details about these lines.
import os
sh = ROOT.SH.SampleHandler()
sh.setMetaString( 'nc_tree', 'CollectionTree' )
#inputFilePath = "/data/atlas/atlasdata3/oneill/DAOD_JETM6/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_JETM6.e6337_e5984_s3126_r10201_r10210_p4128"
#ROOT.SH.ScanDir().filePattern( 'DAOD_JETM6.20933895._001089.pool.root.1' ).scan( sh, inputFilePath )
ROOT.SH.scanRucio (sh, "mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_JETM6.e6337_e5984_s3126_r10201_r10210_p4128")
sh.printContent()

# Create an EventLoop job.
job = ROOT.EL.Job()
job.sampleHandler( sh )
job.options().setDouble( ROOT.EL.Job.optMaxEvents, 10 )
job.options().setString( ROOT.EL.Job.optSubmitDirMode, 'unique-link')

#This is for TTreeCache
#job.options().setDouble (EL::Job::optCacheSize, 10*1024*1024);

job.outputAdd (ROOT.EL.OutputStream ('ANALYSIS'))
# Create the algorithm's configuration.
from AnaAlgorithm.DualUseConfig import createAlgorithm
alg = createAlgorithm ( 'MyxAODAnalysis', 'AnalysisAlg' )

# later on we'll add some configuration options for our algorithm that go here

# Add our algorithm to the job
job.algsAdd( alg )

# Run the job using the direct driver.
#driver = ROOT.EL.DirectDriver()
driver = ROOT.EL.PrunDriver()
driver.options().setString("nc_outputSampleName", "user.mdraguet.testLatest2.%in:name[2]%.%in:name[6]%");
driver.submit( job, options.submission_dir )
