from hw2.experiments import cnn_experiment

for l, p in zip([2, 4, 8],[2, 3, 5]):
    for k in [32, 64, 128]:
        cnn_experiment(
            'exp1_2',
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
            filters_per_layer=[k],
            layers_per_block=l,
            pool_every=p,
            hidden_dims=[1024],
            model_type='cnn', 
        )