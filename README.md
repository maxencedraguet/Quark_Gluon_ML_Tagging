# Quark-Gluon Tagging with Machine Learning - ATLAS Experiment
## Meetings
### Recent progress: 
* Regarding the meeting with Alan's group on Monday: I pushed a draft of the slides to the repo (Short_Presentation_3rd_August_Draguet.pdf)

* A week of many computational challenges. We (with Vip) still have not found what does not work with Condor. I shared the problem on Monday and Ynir (a student of Aaron) kindly had a look at the submit files but could not find any issue. I shared the problem with Prof. Bortoletto (Physics) and she advised me to contact some of her students. 

* Concerning data: I am <b>still</b> doing the JuniprProcessing for the dijet sample. The ttbar sample (miraculously) worked (giving 3.5 GB of jets, about 700k of them). Thanks to Vip, I have received access to a specific worker node until Monday and I am running the job there (there is also a Condor one running but I would not bet on it).

* I have in the meantime stumbled upon a very weird error in JuniprProcessing for dijet. Somehow, for only (as far as I've seen) one of the jet, the seed momentum (the very first mother) finds itself in a loop where it is asked what its mother is, which returns a None and crashes the whole program. I have checked the loop and do not understand how this happen (it is in fact extremely rare: 1 on the 600,000 jets already processed). I am wondering if this is not a bug emanting from PyJet (the library I use to cluster them). I have added some security checks to catch this and just discard such jets (given how rare it is). I also have had several instances of constituents of a <b>single</b> EMTopoJet (reclustered with anti-kT of radius 0.4) reclustered into <b>two</b> jets (with C/A and radius 0.5). I am not sure I understand how this can possibly happen though I don't think it's impossible (one of the particle can be left aside and never clustered because of its distance to the rest). This is more frequent than the other error but still extremely rare (10 on the 500,000 jets already processed) and I also discard such occurences.
    
* I have (laboriously) "finished" training two JUNIPR models. I ran on 300k jets of dijet and 300k jets of ttbar for 25 epochs of 200 batchsize (with the junipr architecture shown in base_config). This process was really slow (I think I've improved things in the meantime, see later): it took easily 20 hours each.  The training visibly did not reach a plateau (both training and validation loss were still going down). I however moved on to a binary JUNIPR training to have a look at the final result. I will restart training these jets (on the same dataset, the full dijet still missing). 

* While training the binary JUNIPR for the first time, I was surprised by how slow it went. Compared to the unary case, it went a 100 times slower (and the unary case was far from being fast to start with): 
    * For the unary case, the "time" (it's CPU time, not real seconds) it took to run a step (process 1 batch) was:
        * Time to reach loss step 0.5172271630000012
        * Time to compute loss step 5.534175025999986
        * Time to backprop loss step 6.994377635999996
        * Finished step | Time for whole step 13.047642465999985
    * For the binary case, the "time" it took to run a step (process 1 batch) was:
        * Time to reach loss step 5.966914106999866
        * Time to compute loss step 619.6969968519998
        * Time to finish backprop loss step 871.8608963879997
        * Finished step | Time for whole step 1497.5664760149994
* I went in debug mode and tried a few things to improve performance. None of the following worked:
    * Lighten the loss function: return a tensor with loss for each of the batch item along a dimension (instead of list of tuples, from which I didn’t use the second item and the first one was the same information).
    * Tried doubling the batch size 200 -> 400. Observations: the forward pass goes from 6 to 9, the time to compute the loss doubles (from 600 to 1300) and backprop takes (slightly) more than twice as long (2200 instead of 900). 
    * Lighten the samples of the training (a lot of information was not necessary for training but only for the junipr tree plotting). 
    * discard junipr tree plotting in training (I had a system sometimes drawing a tree and adding the image to TensorBoard ... looked good and actually showed the weights changing for a tree, or a randomly selected one, but took too long).
* This did not solve the issue, which was clearly coming from the loss and backprop. Indeed, I observed that the binary loss took 300 seconds to perform the exact same function that took 6 seconds in unary case (needing to do it twice, it explains the 600 for computing the loss). I was about to turn the whole loss in batch form when I found the issue:
* in PyTorch, you have to be <b>very</b> <i>very</i> very careful of the require_grad flag of tensors (whether the tensor has to be placed in the computational graph and takes part in the gradient of the loss). Forcing tensors that shouldn't contribute to have no gradient and mentioning those that have to explicitely, I obtained the following binary time case:
        * Time to reach loss step 0.21103984199999104
        * Time to compute loss step 7.794431438999709
        * Time to finish backprop loss step 8.696363398999893
        * Finished step |Time for whole step 16.706329296999684
    * This is as expected only about a factor 2 slower than the unary case (I suspect some of the modifications applied will also speed up the unary case). 

* I hope that this issue is now solved. I have been running (on pplxint 10 and 11 terminal with nohup) the training of two binary JUNIPR models using the 25-epoch-trained JUNIPR models. Given that I need truth labels, and being short of large datasets, I kept every samples and assigned as labels to the jets having -1 a label of 0 or 1 depending on whether they came from the dijet or ttbar datasets respectively (this is not optimal for training obviously ... imagine a jet that really is a quark but had a -1 label from a dijet sample. If the network predicts a quark, the correct true label, it's going to be seen as false since its -1 label made him get a 0 label from the dijet). I am not sure what to do for the binary training:
    * the objective is, for quark jets (their true label) to maximise logP(jet | quark model) - log  logP(jet | gluon model) and the opposite for gluon jet. I turned this into minimising -logP(jet | quark model) + log  logP(jet | gluon model) which is practical since these -logP() is exactly what the unary loss computed. 
    * Should I add a sigmoid layer to this (to do minimise sigmoid[-logP(jet | quark model) + log  logP(jet | gluon model)] for quark jets and a flip of signs inside the sigmoid for gluon jets) ? I feel like the difference between the two logs can easily become quite large and this risk saturating in some case the sigmoid (which is almost linear near 0 but quickly flattens) and therefore the gradient will be very small. 
    * Should I average the loss on the batch size or keep the full summed value? 
    * Should I still attempt to turn the loss module in a batch-friendly way for training (could still speed up some things).
    
* I am training one model with a sigmoid and the other one without it. Both seems to (very slowly ... of course) display the expected behaviour. 
    * The one <b>without</b> sigmoid: the optimisation push make the difference of log probabilities more and more negative (reason why I'm scared of the sigmoid, I think it will quickly saturate for the values at play). What follow is the same batch shown at the start and after one epoch (IMPORTANT: the validation loss shown here have been spuriously multiplied by a factor of 200 ... this has no major consequence so I have interrupted training):
        * Training epoch 0| batch loss :   -2,647.51147461 | accuracy : 0.575
        * Training epoch 1| batch loss : -10,872.37109375 | accuracy : 0.595
        * Validation epoch 0 | loss =   -467,525.63 | accuracy = 0.595966330527544
        * Validation epoch 1 | loss = -3,881,725.25 |  accuracy = 0.6167597299774982
        * Clearly things are improving: the loss is lower (more negative) and the accuracy increased slightly. I am however afraid that this suffers some dramatic volatily. A batch is only but 200 elements taken at random in a dataset containing as many quark as gluon from an unspecified phase-space region. This leads to some very distinct batch data and this is visible in how the whole values fluctuate. For example, in the same epoch, here are some training losses ( 1 step = processing 1 batch of 200 items):
            * Training 1500 step | batch loss :  -5,166.497 | accuracy : 0.585
            * Training 1550 step | batch loss :  -8,041.684 | accuracy : 0.65
            * Training 1600 step | batch loss :  -5,926.687 | accuracy : 0.565
            * Training 1650 step | batch loss :  -6,324.645 | accuracy : 0.59
            * Training 1700 step | batch loss :  -6,280.829 | accuracy : 0.605
            * Training 1750 step | batch loss : -12,999.625 | accuracy : 0.655
              
        * Validation 1500 step |  loss = -1,285,013.87 |  accuracy = 0.6047170597549796
        * Validation 1550 step |  loss = -1,324,978.75 |  accuracy = 0.6035086257188099
        * Validation 1600 step |  loss = -1,277,915.87 |  accuracy = 0.6032586048837403
        * Validation 1650 step |  loss = -1,460,842.75 |  accuracy = 0.5960080006667222
        * Validation 1700 step |  loss = -1,521,012.62 |  accuracy = 0.6046753896158014
        * Validation 1750 step |  loss = -1,614,105.25 |  accuracy = 0.6056754729560797
    * The one <b>with</b> sigmoid: the optimisation push makes the value go down (the value is now limited between 0 and 1). What follow is the same batch shown at the start and after one epoch (same batch as above in fact):
        * Training epoch 0 | batch loss : 0.47359 | accuracy : 0.525
        * Training epoch 1| batch loss : 0.45004| accuracy : 0.55
        * Validation epoch 0 |  loss = 87.37873 |  accuracy = 0.563
        * Validation epoch 1 |  loss = 82.28725 |  accuracy = 0.588
        * These results go in the same direction as suggested above (remember that validation loss is a factor of 200 too large). However all is not so easy: some segment of training had a negative impact on the validation with the accuracy going down or the loss going up.

* Should I change this ? Accumulate gradients on several batches and then propagate ? I should also perhaps train far more the initial JUNIPR models. In this respect, it is actually paramount that they are trained a not too distinct amount. Indeed, if a model is trained far less than anotehr one, the probability it output is much smaller (hasn't really gone into maximising the likelihood of obersving its jet). This can actually introduce large differences which, with the sigmoid, can neutralise training. So sticking to similarly sized dataset and same number of epochs is paramount. Funny note: the model without sigmoid takes longer to train:one step costs on average about 2210 steps versus 1600 for the sigmoid-endowed one.

* I had a look at the diagnostic plots of the dijet samples gathered. All seems normal except for a strange peak in jet energy and pT. For all pairs, ttbar is left and dijet right.
* constituentDeltaRtoJet:
<p float="center">
<img src="Readme_Result/ttbar/constituentDeltaRtoJet.png" width="350" />
<img src="Readme_Result/dijet/constituentDeltaRtoJet.png" width="350" />
</p>

* constituentE_log:
<p float="center">
<img src="Readme_Result/ttbar/constituentE_log.png" width="350" />
<img src="Readme_Result/dijet/constituentE_log.png" width="350" />
</p>

* differencePx_log:
<p float="center">
<img src="Readme_Result/ttbar/differencePx_log.png" width="350" />
<img src="Readme_Result/dijet/differencePx_log.png" width="350" />
</p>

* isTruthQuark:
<p float="center">
<img src="Readme_Result/ttbar/isTruthQuark.png" width="350" />
<img src="Readme_Result/dijet/isTruthQuark.png" width="350" />
</p>

* isNotPVJet (this is before cuts are applied:
<p float="center">
<img src="Readme_Result/ttbar/isNotPVJet_Before_cuts.png" width="350" />
<img src="Readme_Result/dijet/isNotPVJet_Before_cuts.png" width="350" />
</p>

* jetE:
<p float="center">
<img src="Readme_Result/ttbar/jetE.png" width="350" />
<img src="Readme_Result/dijet/jetE.png" width="350" />
</p>

* jetPt:
<p float="center">
<img src="Readme_Result/ttbar/jetPt.png" width="350" />
<img src="Readme_Result/dijet/jetPt.png" width="350" />
</p>

* jetNumberConstituent_log:
<p float="center">
<img src="Readme_Result/ttbar/jetNumberConstituent_log.png" width="350" />
<img src="Readme_Result/dijet/jetNumberConstituent_log.png" width="350" />
</p>

* jetWidth:
<p float="center">
<img src="Readme_Result/ttbar/jetWidth.png" width="350" />
<img src="Readme_Result/dijet/jetWidth.png" width="350" />
</p>



[Notes on meetings.](https://docs.google.com/document/d/1mPCNGwLqUHwPWRzEXwxDVAvANspSMXEBrSzKO49E8Ds/edit?usp=sharing)

## Readings
[Temporary bibliography.](https://docs.google.com/document/d/1T0P84bvZvcEdx9cvs6z_uXsKWNDNlzjyWbvqWfU1s5I/edit)

[Note on Readings.](https://docs.google.com/document/d/1u7orIhStgtNy6GY1Ix_eOC2UjRiMTey7CkkDW5u7Oxg/edit?usp=sharing)

## Work
[Notes on Work Progress.](https://docs.google.com/document/d/1REFWLDmTNmnLVJMIwqeWt13o8EeNrBTAoQybtgy6I2A/edit?usp=sharing)

[Experiment Log.](https://docs.google.com/spreadsheets/d/1Yu8Fxa3OA3b5M0SDpXkCFffr_e0Qvg-HA2QqpyZvl-I/edit?usp=sharing)

PyTorch should be appropriate to implement all considered network implementations and exploit GPU's. In particular:
* Convolutional Neural Network ([CNN](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html))
* Recurrent Neural Network ([RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html))
* Variational Autoencoders ([VAE](https://pyro.ai/examples/vae.html))
* Generative Adversarial Networks ([GAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html))

A larger list of tutorials for [PyTorch](https://pytorch.org/tutorials/). 

A general tutorial on EventLoop is accessible [here](https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/EventLoop#Grid_Driver) but requires a CERN account.

A tutorial on how to use Athena and the ATLAS codebase to analyse the xAOD files can be found [here](https://atlassoftwaredocs.web.cern.ch/ABtutorial/alg_basic_intro/).

An ATLAS dataset browser is available here [AMI](https://ami.in2p3.fr) and requries a CERN account as well as some certificates. 

A short explanation on variables is available [here](https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/Run2JetMoments) but requires a TWIKI access. 

The JUNIPR framework is implemented on [this github page](https://github.com/andersjohanandreassen/JUNIPR)
