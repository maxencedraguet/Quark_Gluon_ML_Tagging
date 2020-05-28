# Quark-Gluon Tagging with Machine Learning - ATLAS Experiment
## Meetings
### Recent progress: 
* Worked on the dataloader, processing and formatting for the entire database. Had to trim a 2.2 TB to about 17 GB. This is implemented in the /DataLoader/UpRootTransformer.py file of the repo.
* Adapted to processing steps from Aaron’s code and slightly-optimised it. A lot of iteration were required to account for many bugs (empty datasets, cuts having the wrong effect, etc.).
* In order to achieve the data processing, we had no choice but to move to HDCondor. This required a lot of effort to make the entire code run. We managed to find a surprisingly succinct way of performing this. The big advantage is that the method is general enough to run any variant of the code. Implemented in the job_submitter.submit and Submitter_Condor.py files of /Experiments/
* Discussion on file store format. Requiring long-term, stable and robust storage for large data, we took the decision to opt for HDF5, which is particularly well suited for pandas and numpy. We eliminated ROOT as this would require a lot of processing to store and load. CSV was not optimal as size of storage and speed are bad. The data is stored on the ATLAS disk at /data/atlas/atlasdata3/mdraguet/Set2/HF
* The data processing generated some diagnostic plots of the datasets (such as the number of true quark versus true gluon: isTruthQuark, …).
* Implemented a dataloader for this new data with further care taken to stabilise the occurence of each particle. Problem: neither kera nor sklearn make Adaboost decision tree compatible with incremental learning. 
* IN PROGRESS Ran the BDT on the whole training dataset with grid search. Optimising the result. 
*  WAITING FOR PREVIOUS STEP: ran the BDT model found optimal on the whole dataset. Displayed some result statistics. 
* Got access to the ATLAS resources (thank you Alan for your help).
* The finer granularity dataset in light format for the next steps should be available, according to Aaron, early next week. If I have time, I will implement a NN model to classify. 

[Notes on meetings.](https://docs.google.com/document/d/1mPCNGwLqUHwPWRzEXwxDVAvANspSMXEBrSzKO49E8Ds/edit?usp=sharing)

## Readings
[Temporary bibliography.](https://docs.google.com/document/d/1T0P84bvZvcEdx9cvs6z_uXsKWNDNlzjyWbvqWfU1s5I/edit)

[Note on Readings.](https://docs.google.com/document/d/1u7orIhStgtNy6GY1Ix_eOC2UjRiMTey7CkkDW5u7Oxg/edit?usp=sharing)

## Work
[Notes on Work Progress.](https://docs.google.com/document/d/1REFWLDmTNmnLVJMIwqeWt13o8EeNrBTAoQybtgy6I2A/edit?usp=sharing)

PyTorch should be appropriate to implement all considered network implementations and exploit GPU's. In particular:
* Convolutional Neural Network ([CNN](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html))
* Recurrent Neural Network ([RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
* Variational Autoencoders ([VAE](https://pyro.ai/examples/vae.html))
* Generative Adversarial Networks ([GAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html))

A larger list of tutorials for [PyTorch](https://pytorch.org/tutorials/). 
