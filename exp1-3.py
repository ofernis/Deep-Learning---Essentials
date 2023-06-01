from hw2.experiments import cnn_experiment

K = [64, 128]
for l in [2, 3, 4]:
    cnn_experiment(
        'exp1_3',
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
        pool_every=2,
        hidden_dims=[1024],
        model_type='cnn', 
    )