from hw2.experiments import cnn_experiment

for K, L, P in [([32], [8,16,32], [4,8,16]),([64,128,256], [2,4,8], [2,3,5])]:
    for l, p in zip(L, P):
        cnn_experiment(
            'exp1_4',
            seed=None,
            # Training params
            bs_train=128,
            bs_test=None,
            batches=100,
            epochs=100,
            early_stopping=3,
            lr=1e-3,
            reg=1e-3,
            # Model Params
            filters_per_layer=K,
            layers_per_block=l,
            pool_every=l,
            batchnorm=True,
            dropout=0.3,
            hidden_dims=[1024],
            model_type='resnet',
        )