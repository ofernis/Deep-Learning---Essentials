from hw2.experiments import cnn_experiment

for k in [32, 64]:
    for l in [2, 4, 8, 16]:
        cnn_experiment(
            'exp1_1',
            seed=None,
            # Training params
            bs_train=128,
            bs_test=None,
            batches=100,
            epochs=60,
            early_stopping=3,
            lr=7e-4,
            reg=1e-3,
            # Model Params
            filters_per_layer=[k],
            layers_per_block=l,
            pool_every=min(l//2, 3),
            hidden_dims=[1024],
            model_type='cnn', 
        )