#############################################################################
#
# Main.py
#
# Using Set 1 to train a BDT to tag jets as quark
# or gluon initiated.
#
# Author -- Maxence Draguet (19/05/2020)
#
#############################################################################

from Context import Models, Utils, DataLoaders

import argparse
import time
import datetime
import os
import yaml

parser = argparse.ArgumentParser()

parser.add_argument('-config', type=str, help='path to configuration file', default='base_config.yaml')
parser.add_argument('--experiment_name', type=str, default=None)

args = parser.parse_args()

if __name__ == "__main__":
    main_file_path = os.path.dirname(os.path.realpath(__file__))
    print(main_file_path)
    # Read and store base configuration parameters
    base_config_full_path = os.path.join(main_file_path, args.config)
    with open(base_config_full_path, 'r') as yaml_file:
        parameters = yaml.load(yaml_file, yaml.SafeLoader)
    experiment_parameters = Utils.Parameters.MainParameters(parameters)

    # If need be, update the parameters from parser values
    if args.experiment_name:
        experiment_parameters._config["experiment_name"] = args.experiment_name
    
    # Establish experiment archiving.
    exp_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    experiment_name = experiment_parameters.get("experiment_name")
    log_path = os.path.join(main_file_path, 'Results', exp_timestamp, experiment_name)
    experiment_parameters.set_property("log_path", log_path)
    experiment_parameters.set_property("experiment_timestamp", exp_timestamp)
    experiment_parameters.set_property("df_log_path", os.path.join(log_path, 'data_logger.csv'))

    # Set a seed value to packages with non-deterministic initialisaiton for reproducibility
    import random
    import numpy as np
    import torch
    seed_value = experiment_parameters.get("seed")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    experiment_parameters._config["seed"] = seed_value

    if experiment_parameters.get(["experiment_type"]) == "BDT":
        experiment_parameters.save_configuration(log_path)
        runner = Models.BDTRunner(config=experiment_parameters)

    elif experiment_parameters.get(["experiment_type"]) == "NN":
        experiment_parameters.save_configuration(log_path)
        runner = Models.NNRunner(config=experiment_parameters)

    elif experiment_parameters.get(["experiment_type"]) == "Multi_model":
        runner =  Models.MultimodalRunner(config=experiment_parameters)

    elif experiment_parameters.get(["experiment_type"]) == "UpRootTransformer":
        runner = DataLoaders.UpRootTransformer(config=experiment_parameters)

    elif experiment_parameters.get(["experiment_type"]) == "Train_Test_Separator":
        runner = DataLoaders.Train_Test_Separator(config=experiment_parameters)
