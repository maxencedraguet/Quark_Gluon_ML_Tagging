#############################################################################
#
# models_to_load.py
#
# This lists the models to load in the specific format expected.
#
#############################################################################

models_to_load_list = [("BDT", "/home/draguet/QGdis/Code/Experiments/store_model2/BDT_300_estimators_depth_3_lr_0.15/saved_model.joblib", "None", "BDT_300_estimators_depth_3_lr_0.15"),
                       
                       ("NN", "/home/draguet/QGdis/Code/Experiments/store_model2/NN_20_20/saved_NN_weights.pt",
                        "/home/draguet/QGdis/Code/Experiments/store_model2/NN_20_20/config.yaml", "NN_2_20_-unit_hidden_layers"),
                       
                       ("NN", "/home/draguet/QGdis/Code/Experiments/store_model2/NN_20_20_dropout_0.1/saved_NN_weights.pt",
                        "/home/draguet/QGdis/Code/Experiments/store_model2/NN_20_20_dropout_0.1/config.yaml", "NN_2_20-unit_hidden_layers_(dropout=0.1)"),
                       
                       ("NN", "/home/draguet/QGdis/Code/Experiments/store_model2/NN_20_20_dropout_0.2/saved_NN_weights.pt",
                        "/home/draguet/QGdis/Code/Experiments/store_model2/NN_20_20_dropout_0.2/config.yaml", "NN_2_20-unit_hidden_layers_(dropout=0.2)" ),
                       
                       ("NN", "/home/draguet/QGdis/Code/Experiments/store_model2/NN_32_32_weight_decay_000001/saved_NN_weights.pt",
                        "/home/draguet/QGdis/Code/Experiments/store_model2/NN_32_32_weight_decay_000001/config.yaml", "NN_2_32-unit_hidden_layers_(wd=0.000001_and_dropout=0.1)" ),
                       
                       ("NN", "/home/draguet/QGdis/Code/Experiments/store_model2/NN_32_32_weight_decay_00001/saved_NN_weights.pt",
                        "/home/draguet/QGdis/Code/Experiments/store_model2/NN_32_32_weight_decay_00001/config.yaml", "NN_2_32-unit_hidden_layers_(wd=0.00001_and_dropout=0.1)" ),
                       
                       ("NN", "/home/draguet/QGdis/Code/Experiments/store_model2/NN_32_32_weight_decay_0001/saved_NN_weights.pt",
                        "/home/draguet/QGdis/Code/Experiments/store_model2/NN_32_32_weight_decay_0001/config.yaml", "NN_2_32-unit_hidden_layers_(weight_decay=0.0001)" ),
                       
                       ("NN", "/home/draguet/QGdis/Code/Experiments/store_model2/NN_32_32_weight_decay_001/saved_NN_weights.pt",
                        "/home/draguet/QGdis/Code/Experiments/store_model2/NN_32_32_weight_decay_001/config.yaml", "NN_2_32-unit_hidden_layers_(weight_decay=0.001)" ),
                       
                       ("NN", "/home/draguet/QGdis/Code/Experiments/store_model2/NN_32_32_weight_decay_01/saved_NN_weights.pt",
                        "/home/draguet/QGdis/Code/Experiments/store_model2/NN_32_32_weight_decay_01/config.yaml", "NN_2_32-unit_hidden_layers_(weight_decay=0.01)" )]
                       
