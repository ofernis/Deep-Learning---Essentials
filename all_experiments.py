from hw2.experiments import cnn_experiment

# experiment 1.1
# for k in [32, 64]:
#     for l in [2, 4, 8, 16]:
#         cnn_experiment(
#             'exp1_1',
#             seed=None,
#             # Training params
#             bs_train=128,
#             bs_test=None,
#             batches=100,
#             epochs=60,
#             early_stopping=3,
#             lr=7e-4,
#             reg=1e-3,
#             # Model Params
#             filters_per_layer=[k],
#             layers_per_block=l,
#             pool_every=min(l//2, 3),
#             hidden_dims=[1024],
#             model_type='cnn', 
#         )
        
        
# experiment 1.2
# for l in [2, 4, 8]:
#     for k in [32, 64, 128]:
#         cnn_experiment(
#             'exp1_2',
#             seed=None,
#             # Training params
#             bs_train=128,
#             bs_test=None,
#             batches=100,
#             epochs=60,
#             early_stopping=3,
#             lr=7e-4,
#             reg=1e-3,
#             # Model Params
#             filters_per_layer=[k],
#             layers_per_block=l,
#             pool_every=min(l//2, 3),
#             hidden_dims=[1024],
#             model_type='cnn', 
#         )
       
        
# experiment 1.3
# K = [64, 128]
# for l in [2, 3, 4]:
#     cnn_experiment(
#         'exp1_3',
#         seed=None,
#         # Training params
#         bs_train=128,
#         bs_test=None,
#         batches=100,
#         epochs=60,
#         early_stopping=3,
#         lr=7e-4,
#         reg=1e-3,
#         # Model Params
#         filters_per_layer=K,
#         layers_per_block=l,
#         pool_every=min(l//2, 3),
#         hidden_dims=[1024],
#         model_type='cnn', 
#     )
    

# experiment 1.4
for K, L, P in [([32], [8,16,32], [4,8,16]),([64,128,256], [2,4,8], [2,3,5])]:
    for l, p in zip(L, P):
        cnn_experiment(
            'exp1_4',
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
            filters_per_layer=K,
            layers_per_block=l,
            pool_every=p,
            hidden_dims=[1024],
            model_type='resnet', 
        )


# num_of_conv = len(K) * l, pool_every = min(l//2, math.ceil(num_of_conv/log2(img_size)))