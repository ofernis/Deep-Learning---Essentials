from hw2.experiments import cnn_experiment

for K, L in [([32], [8,16,32]),([64,128,256], [2,4,8])]:
    for l in L:
        cnn_experiment(
            'exp1_4',
            seed=None,
            # Training params
            bs_train=128,
            bs_test=None,
            batches=100,
            epochs=60,
            early_stopping=3,
            lr=1e-3,
            reg=1e-3,
            # Model Params
            filters_per_layer=K,
            layers_per_block=l,
            pool_every=l//2+1,
            batchnorm=True,
            dropout=0.4,
            hidden_dims=[1024],
            pooling_type='avg',
            model_type='resnet',
        )