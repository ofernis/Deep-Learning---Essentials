r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

1. For the Jacobian tensor $\pderiv{\mat{Y}}{\mat{X}}$:
    1. The shape of this tensor will be (64, 512, 64, 1024).
    2. This Jacobian is sparse. We're differentiating each output sample (in Y) w.r.t each input sample (in X) and we know
    that each output sample derivative depends only on its corresponding input sample. 
    Hence, the only elements that are not zero correspond to indices of the form (i,j,i,k) where $1<=i<=64$,
    meaning that this is the derivative of the i-th output sample w.r.t its corresponding input sample (same i-th index).
    3. No, we don't need to materialize the Jacobian tensor $\pderiv{\mat{Y}}{\mat{X}}$ in order to calculate the downstream gradient $\delta\mat{X}$
    
        Instead, we can leverage the chain rule to compute the gradient efficiently:
        $$
        \delta\mat{X}=\pderiv{L}{\mat{X}} = \pderiv{L}{\mat{Y}} \pderiv{\mat{Y}}{\mat{X}}
        $$
        - $\pderiv{L}{\mat{Y}}$ is given to us.
        - $\pderiv{\mat{Y}}{\mat{X}}$ is $\mattr{W}$ (because $\mat{Y}=\mat{X} \mattr{W} + \vec{b}$ - Using matrix differentiation rules).  
    
    As we saw in the previous clause, the only non-zero elements of the Jacobian tensor $\pderiv{\mat{Y}}{\mat{X}}$ have the index form of (i,j,i,k),
    meaning that only those elements in the Jacobian tensor are needed
    in order to compute the gradient (all other elements in the product matrix will be set to zero).  
    So eventually we get: 
    $$
    \delta\mat{X}=\pderiv{L}{\mat{Y}}\mattr{W}=\delta\mat{Y}\mattr{W}
    $$
        
2. For the Jacobian tensor $\pderiv{\mat{Y}}{\mat{W}}$:
    1. The shape of this tensor will be (64, 512, 512, 1024).
    2. This Jacobian is sparse again, for the same reason as earlier - We're differentiating each output feature (in Y) w.r.t each input feature (in W), 
    such that the only non-zero elements correspond to indices of the form (i,j,i,k) where $1<=i<=64$,
    meaning that this is the derivative of the i-th output feature w.r.t its corresponding input feature (same i-th index).
    3. No, given the gradient of the output $\pderiv{\mat{L}}{\mat{Y}}$, we can again use the chain rule to compute the gradient efficiently:
        $$
        \delta\mat{W}=\pderiv{L}{\mat{W}} = \pderiv{L}{\mat{Y}} \pderiv{\mat{Y}}{\mat{W}}
        $$
        - $\pderiv{L}{\mat{Y}}$ is given to us.
        - $\pderiv{\mat{Y}}{\mat{W}}$ is $\mattr{X}$ (because $\mat{Y}=\mat{X} \mattr{W} + \vec{b}$ - Using matrix differentiation rules).  
    
    As we saw in the previous clause, the only non-zero elements of the Jacobian tensor $\pderiv{\mat{Y}}{\mat{W}}$ have the index form of (i,j,i,k),
    meaning that only those elements in the Jacobian tensor are needed
    in order to compute the gradient (all other elements in the product matrix will be set to zero).
    So eventually we get: 
    $$
    \delta\mat{W}=\pderiv{L}{\mat{Y}}\mattr{W}=\delta\mat{Y}\mattr{X}
    $$
    
# TODO
"""

part1_q2 = r"""
**Your answer:**

**Yes**, for any practical purposes, backpropagation is required in order to train neural networks.

This is because without backpropagation, it would be difficult and computationally expensive (and even infeasible) to calculate these gradients manually.
Backpropagation automates the process, making training feasible and efficient.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.05
    lr = 0.02
    reg = 0.00001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.05
    lr_vanilla = 0.021
    lr_momentum = 0.0022
    lr_rmsprop = 0.00025
    reg = 0.001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.05
    lr = 0.0022
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

**1.1**: Our baseline is no dropout setting, with pre-tuned hyper-parameters for an overfitted model (overfitting can be inferred from the graphs, where the train accuracy is much higher than the test accuracy).
We see that the training accuracy is higher without dropout in comparison to using dropout, but the test accuracy is lower for the same dropout settings comparison.
This fits what we expected, as the dropout method "omits" training information and thus decrease the training accuracy, but due to its randomness, it makes the network less sensitive to the influence of single
neurons and therefore demonstrate better generalization. Looking at our graphs, it is worth mention that as the dropout level increases, the training accuracy decreases (due to the reason we mentioned earlier).
An opposite effect is seen in the test accuracy graphs, where the lower accuracy corresponds to higher dropout level (0.8) rather than the lower dropout level (0.4).

**1.2**: As we see in the graphs, the high-dropout setting had lower accuracy than the low-dropout setting, indicating a lower generalization competence for this setting.
This might stem from the fact that too many neurons activations are being dropped on this setting (80% percent on average), leading to insufficient optimization in the training phase.
As we expect, the best performance of the optimization is related to a medium level of dropout, which in our case is 0.4, in comparison to 0 (no dropout) and to 0.8 (high-dropout).

"""
part2_q2 = r"""
**Your answer:**


When training a model with the cross-entropy loss function, it is possible for the test loss to **increase** for a few epochs while the test accuracy also **increases**.

We can look at an example for a single epoch:
- We have 3 classes and 10 examples.
- Our model gave probability $\frac{1}{3} + \epsilon$ for the correct class, and a little less for each other class.
- Thus, our model's accuracy is very high (in this example it's 100%).
- And on the other hand, our model's loss is high because the CE loss is proportional to the probability given to the correct class.
$$
L_{CE}(\hat{y},y)
=
-\sum_{k=1}^K\mathbb{I}\left[ y=k \right]\log{\frac{e^{p_k}}{\sum^K_{j=1}e^{p_j}}}
\propto
p_y
$$

To conclude, if this happens for each epoch with decreasing $\epsilon$, then the behavior will be as described.

"""

part2_q3 = r"""
**Your answer:**


### Answer 1

In essence, gradient descent is the optimization process, while backpropagation is the algorithm used within that process to compute the gradients.

**Gradient descent**:
- Gradient descent is an optimization algorithm used to **minimize a loss function**.
- It iteratively adjusts the model parameters by moving in the direction of the steepest descent of the loss function to find the minimum.
- It can use the backpropagation algorithm to compute the gradients of the model parameters more efficiently. 

**Backpropagation**:
- Backpropagation is an algorithm used to **compute the gradients of the model parameters**, usually in a neural network.
- It propagates the error gradients backward through the network, allowing for efficient calculation of gradients needed for gradient descent.
- For large neural networks, the gradient calculations can be infeasible without the backpropagation algorithm.

### Answer 2

1. **Batch updates**: In GD, the entire training dataset is used to compute the gradient of the cost function before updating the model parameters.
In SGD, only a single training example (or a mini-batch) is randomly selected and used to compute the gradient and update the parameters.

2. **Efficiency**: As GD considers the entire dataset for each parameter update, it can be computationally expensive, especially for large datasets.
SGD is more efficient compared to GD as it uses a single example for the parameter update.

3. **Convergence**: GD converges to the global minimum if the loss function is convex, the convergence is generally slower as it takes more steps to reach the minimum.
SGD might not converge to the global minimum due to the randomness introduced by selecting the examples.
While it might not reach the optimal solution, it often finds a good enough solution in practice and does that faster than GD.

4. **Noise and Regularization**: GD uses the entire dataset, resulting a smoother parameter updates.
SGD introduces randomness to the parameter updates, making it more noisy, but allowing it to escape local minima sometimes, making it an implicit form of regularization.

### Answer 3
SGD is commonly used in the practice of deep learning for several reasons:
1. **Efficiency with large datasets**: Deep learning often involves working with massive datasets that may not fit entirely into memory.
SGD's ability to update parameters based on individual examples (or mini-batches) makes it more efficient.

2. **Faster convergence**: While SGD's updates tends to be noisier than GD's, it can actually lead to faster convergence in practice.

3. **Generalization**: SGD's inherent stochastically during training acts as a form of regularization, preventing overfitting and promoting better generalization.

4. **Avoiding local minima**: Deep learning models often have complex, high-dimensional loss landscapes with numerous local minima. GD relies on the entire dataset and may get trapped in suboptimal local minima.
SGD's stochastic nature provides a mechanism for escaping these suboptimal solutions.

###  Answer 4
1. **Yes**, this method will produce a gradient equivalent to GD. We remember that:
$$
L(\vec{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \ell(\vec{y}^i, \hat{\vec{y}^i}) + R(\vec{\theta})
$$  
We can re-write the above using $K$ disjoint batches (assume we keep the same order for the $i$ 's):
$$
L(\vec{\theta}) 
= \frac{1}{N} \sum_{j=1}^{K} \sum_{i \in B_j} \ell(\vec{y}^i, \hat{\vec{y}^i}) + R(\vec{\theta})
$$
Thus, the forward calculation is equivalent.

Now, we have $L(\vec{\theta})$ and wish to calculate $\frac{\partial L}{\partial \vec{x}}$, we don't need the dataset anymore for this calculation, so it is equivalent to GD's.

2. The most probable thing that happened is that during each forward pass, we accumulate the activation's values and store them in the memory.
Thus, after multiple forward passes, the memory could be filled with the values we accumulated.

"""

part2_q4 = r"""
**Your answer:**

1. Using the **chain rule** we get:
$$
\frac{\partial f}{\partial x_0} = \frac{\partial f}{\partial f_n} \frac{\partial f_n}{\partial f_{n-1}} \cdots \frac{\partial f_1}{\partial x_0}
$$

**In forward mode:**

We can calculate the derivative with the following equations:
$$
\begin{align}
\frac{\partial f_1}{\partial x_0} &\\
\frac{\partial f_2}{\partial x_0} &= \frac{\partial f_2}{\partial f_1} \frac{\partial f_1}{\partial x_0} \\
& \vdots \\
\frac{\partial f_n}{\partial x_0} &= \frac{\partial f_{n-1}}{\partial f_{n-2}} \frac{\partial f_{n-2}}{\partial x_0} \\
\frac{\partial f}{\partial x_0} &= \frac{\partial f}{\partial f_n} \frac{\partial f_n}{\partial x_0} \\
\end{align}
$$

We can compute the above using only 2 variables to store the calculations - accumulated derivative $\delta$ and current derivative $d$:
$$
\begin{align*}
&1.~ \text{Init:} \\
& \quad d_1 \leftarrow \frac{\partial f_1}{\partial x_0} \\
& \quad \delta_1 \leftarrow d \\
&2.~ \text{for}~ i=2,...,n:\\
& \quad d_i \leftarrow \frac{\partial f_i}{\partial f_{i-1}} \\
& \quad \delta_i \leftarrow \delta_{i-1} \cdot d_i \\
&3.~ d_f \leftarrow \frac{\partial f}{\partial f_n} \\
&4.~ \text{Return}~ \delta_n \cdot d_f \\
\end{align*}
$$

This way, we reduced the memory complexity to $\mathcal{O}(1)$ while keeping time complexity at $\mathcal{O}(n)$.

**In backward mode:**

This time, we can calculate the derivative with the following equations:
$$
\begin{align}
\frac{\partial f}{\partial f_n} &\\
\frac{\partial f}{\partial f_{n-1}} &= \frac{\partial f}{\partial f_n} \frac{\partial f_n}{\partial f_{n-1}} \\
& \vdots \\
\frac{\partial f}{\partial f_1} &= \frac{\partial f}{\partial f_2} \frac{\partial f_2}{\partial f_1} \\
\frac{\partial f}{\partial x_0} &= \frac{\partial f}{\partial f_1} \frac{\partial f_1}{\partial x_0} \\
\end{align}
$$
We can compute the above using the following algorithm;
$$
\begin{align*}
&1.~ \text{Init:} \\
& \quad d_n \leftarrow \frac{\partial f}{\partial f_n} \\
& \quad \delta_1 \leftarrow d \\
&2.~ \text{for}~ i=n,...,2:\\
& \quad d_i \leftarrow \frac{\partial f_i}{\partial f_{i-1}} \\
& \quad \delta_i \leftarrow \delta_{i+1} \cdot d_i \\
&3.~ d_f \leftarrow \frac{\partial f_1}{\partial x_0} \\
&4.~ \text{Return}~ \delta_n \cdot d_f \\
\end{align*}
$$

Again, we reduced the memory complexity to $\mathcal{O}(1)$ while keeping time complexity at $\mathcal{O}(n)$.

2. **Yes**, this technique can be generalized for any arbitrary computational graphs.  
The change that we need to make is computing the gradient for each input (in forward mode) 
or for each output (in backward mode).
This will increase the memory usage to $\mathcal{O} \left( \text{\# \{ inputs or outputs \}} \right)$.

3. In deep network architectures, it is common that the number of outputs is smaller than the number of inputs 
(i.e., the input is a 32x32x3 image - 3072 inputs, while the output is a vector of size $C$ - the number of classes, which is much smaller).  
Thus, the the backprop algorithm with the backward mode will benefit from these techniques because the memory cost will be much lower.
"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 3
    hidden_dims = 10
    activation = 'tanh'
    out_activation = 'none'
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.012
    weight_decay = 0.015
    momentum = 0.91
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**

1. Optimization error - **low**  
This error is between the best predictor we can get from the training set, $\theta^*_{\text{training}}$
and the best predictor we can get by the optimization $\theta^*_{\text{optimization}}$.  
The signs for high optimization error are: slow convergence, instability, or erratic behavior **during training**.  
As we can see in the graph, the training proccess gives us a nice, smooth convergence to a high accuracy,
so this error is probably low.

2. Generalization error - **high**  
This error is between the ideal predictor for the architecture we chose, $\theta^*_{\text{world}}$
the best predictor we can get from the training set, $\theta^*_{\text{training}}$.  
The signs for high generalization error are: overfitting (high training accuracy but low validation/test accuracy, and poor performance on new data).  
As we can see in the graph, we get high training accuracy but our validation accuracy is erratic and low.
So we can conclude that our generalization error is high.

3. Approximation error - **low**  
This error is between the ideal predictor (overall) $\theta^*$ and
the ideal predictor for the architecture we chose, $\theta^*_{\text{world}}$  
The signs for high approximation error are: underfitting (high training and validation errors, low accuracy, and poor predictive performance).  
As we can see in the graph, we get high training accuracy (and low error) so our approximation error is probably low.
"""

part3_q2 = r"""
**Your answer:**

In the data generation process, we know that the validation data has more noise than the training data (0.25 in comparison to 0.2).
Both datasets were rotated in a different angle, a manipulation that adds more noise to the data. Nevertheless, due to the symmetry of the "moons"
shapes - it does not change the proportion of FPR to FNR. therefore, understand the both FPR and FNR become higher in the validation set, 
with relatively same values, according to the symmetry we mentioned above.

"""


part3_q3 = r"""
**Your answer:**


You wish to screen as many people as possible at the lowest possible cost and loss of life.
Would you still choose the same "optimal" point on the ROC curve as above?
If not, how would you choose it?
Answer these questions for two possible scenarios:

1. A person with the disease will develop non-lethal symptoms that immediately confirm the diagnosis and can then be treated.
2. A person with the disease shows no clear symptoms and may die with high probability if not diagnosed early enough, either by your model or by the expensive test.

**Scenario 1 - non-lethal symptoms**  
*   Because the symptoms are non-lethal, and confirm the diagnosis.
*   So, we would choose a threshold with more weight towards lower FPR.
*   This means that less patients will falsely get diagnosed positive for the illness and get the expensive tests to confirm the diagnosis.
*   By that, we reduced the cost while maintaining the low loss of life.

**Scenario 2 - lethal illness and no clear symptoms**  
*   The illness is lethal and the symptoms are not clear, so we must allow more false-negatives to be able to save lives.
*   Thus, we should choose the threshold with more weight towards the low FNR.
*   This means that more patients will be falsely diagnosed for the illness, and start the treatment to save their lives
and that the expensive test will give us the true results later on.
*   By that, we reduced the loss of life while increasing the cost by a little (more expensive tests).

"""


part3_q4 = r"""
**Your answer:**


Analyze your results from the Architecture Experiment.

1. For fixed `depth` and varying `width`, we can see that the performance increases as we increase the width, but at some point the high width tends to create overfit,
we can see that by the drop in the accuracies at the bottom model of each column.
2. For fixed `width` and varying `height`, we can see the same behavior as in the first answer above. Increasing the depth makes the model more expressive, 
but at some point it creates overfit.
3. The two models, although having the same number of parameters, have huge difference in performance.
The `depth=1, width=32` has a very poor performance, the low depth enforces a low expressivity, making the model not much different from a regulat SVM model (linear decision boundary).  
On the other hand, the `depth=4, width=8` is the best performing model in terms of validation and test accuracy. The depth and width are in a good balance and making the model both very expressive, and very generalized for the data.
4. The threshold selection method was performed the validation set and not on the train set, meaning that the model was able to find its optimal hyperparameter and optimize its performance according to new samples (no data leakage).
This aided to provide a better generalization and a better accuracy on the test set.

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.011
    weight_decay = 0.015
    momentum = 0.87
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**  
1. The number of parameters a  convolution has is calculated by: $\left( K \times K \times C_{\text{in}} + 1 \right) \cdot C_{\text{out}}$.  
Where $K$ is the `kernel_size` and $C_{\text{in}}, C_{\text{out}}$ are the number of `channels`, each filter also has +1 for the `bias`.  
    -   The regular block have 2 convolution layers:  
    The first layer have $\left( 3 \times 3 \times 256 + 1 \right) \cdot 64 = 147520$ parameters.  
    The second layer have $\left( 3 \times 3 \times 64 + 1 \right) \cdot 256 = 147712$.  
    Thus, the regular block have a total of $295232$ parameters.  
    -   The bottleneck block have 3 layers: The first have $\left( 1 \times 1 \times 256 + 1 \right) \cdot 64 = 16448$.  
    The second have $\left( 3 \times 3 \times 64 + 1 \right) \cdot 64 = 36928$.  
    The third have $\left( 1 \times 1 \times 64 + 1 \right) \cdot 256 = 16640$.
    Giving a total of $70016$ parameters (~25% of the regular block!).  
2. The number of FLOPs done in a convolution layer is calculated by: $\# \text{FLOPs} = \left[ \left( K^2 \times C_{\text{in}} + 1 \right) \cdot C_{\text{out}} \right] \cdot \left( H \times W \right)
= \# \text{parameters} \times H \times W $.  
Thus, for a $32 \times 32$ images, the regular block have $302317568 \approx 300M$ FLOPs 
and the bottleneck block have $71696384 \approx 70M$ FLOPs.  
3. Well explain the two requested aspects:  
    -   Spatial Combination within Feature Maps: The bottleneck block will have a lower spatial ability than the regular block, because it reduces and restores the dimensions of the data to save calculations. 
    -   Combination across Feature Maps: The bottleneck reduction and restoration should not affect this ability directly. However, the spatial ability could lead to a secondary affect on this ability.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

### Question 1 

Analyze your results from experiment 1.1. In particular,
1. **More depth doesn't mean better accuracy**  
We can see that the lower depths `L=2` and `L=4` performed better than `L=8` and `L=16`.
For `K=32` the depths `L=8,16` were untrainable and for `K=64` the depth `L=16` was untrainable.
This shows than increasing the depth helps until some upper bound, then after that the performance begin to decrease.
1. We can see that both `L=8` and `L=16` were untrainable.  
A probable cause for that is the vanishing/exploding gradients effect that usually happens in deep networks.  
A solution to that problem can be using a different activation function, using batch normalization or initializing the weights differently.

"""

part5_q2 = r"""
**Your answer:**

We can see that again, **more width doesn't mean better accuracy.**  
*   For `L=4` the higher `K` gave us better performance (by a little), but on the other hand for `L=2`, the best performing `K` was `K=32`.
For `L=2` the results were very close, while for `L=8` only the `K=64` model was trainable.
*   Both experiment gave us a similar conclusion, there is a trade-off between high and low depth/width, and we should look for the middle "sweet spot".
"""

part5_q3 = r"""
**Your answer:**

*   We can see that the varying number of filters `K` made the model a little better in terms of accuracy.
*   The model with `L=4` was untrainable, probably due to the same vanishing/exploding gradients problem.
*   This experiment shows us that increasing `K` throughout the depth of the network, helps increasing the model's performance.
*   This gives us a good intuition for a good architecture for these CNNs.

"""

part5_q4 = r"""
**Your answer:**

*   First of all, We can see that the ResNet model gave us much higher score compared to the regular CNN model.
*   With `L=2` and `K=[64, 128, 256]` we got the best results - around 90%.
This is consistent with the idea of increasing the number of channels the deeper we go
*   In comparrison to experiments 1.1 and 1.3, the ResNet models were much more resilient to the higher depth of the network.  
We saw that the regular CNN models failed to train with a high `L` value, but the ResNet models were able to perform very well with them.

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**

1. The model managed to detect the boundaries of the objects in the pictures generally well, but its classifications were mostly wrong. 
It can be seen in the first picture, where two dolphins were classified as persons and the tail of the third one was classified as a surfboard.
In the second picture, two dogs were classified as cats and the cat was not detected at all.
We'll also notice that the dog that was classified correctly had a score of 0.51, which implies a significant level of uncertainty.
All in all, the model's performance was not good for these 2 pictures.  
2. We'll examine numerous reasons for the model failures and suggest compatible solutions for the issues:  
- A possible reason for the model failure is lack of light/dark images, which makes objects' features less noticeable and produces difficulties in the models classification.
This can be resolved by adding brightness to the images, changing their contrast rate or using similar lighting techniques.  
- Another possible reason for the model failures can be a missing class, on which the model was not trained - like in the first picture, where it's seems like there is no dolphin class exists.
The solution for that issue is to train the model to classify dolphins in addition to the existing classes.  
- Furthermore, a bias that is related to image setup in the training phase could lead to a misclassification of objects. 
For example, in the first image there is a background of sea surface, sun and sky, which might cause the model to classify the dolphins as humans - 
in case it was trained on pictures of humans with beach background and on pictures of dolphins with underwater background. 
To solve this problem, more pictures of each class can be provided for the training, containing different backgrounds in order to prevent this kind of classification bias.  
- Looking at the second picture, the overlapping of the objects (occlusion) might cause misclassification or missing detection of an object.
A solution for that might involve some spatial manipulations like rotation, cropping or resizing some objects in the pictures.
We'll mention that the YOLO model's algorithm divides the image into a grid of boxes, and tries to detect+classify an object in each box according to its center.
Adjusting the number of boxes in the grid (equivalent to changing the boxes' sizes) could improve the model's performance in this situation.
"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**

The model detected the objects in the pictures very badly (again the detection of boundaries was good, but a total misclassification was demonstrated):

- In the occlusion demonstration picture, a child's body was hidden behind a tree, such that only her head was appearing.
The child was classified as a bird instead of a person, but with a relatively low confidence level - 0.29.

- In the model bias demonstration picture, a cell phone was caught in different views (front, side, back and angular). 
The front and the side views were not detected at all, whereas the back view was classified as a remote. Only the angular view was classified correctly (with high confidence level!).

- In the blurring demonstration picture, a running dog was caught without a proper focus - thus the obtained picture was blurred. 
The dog was classified as a horse with a high confidence level - 0.77.

These all imply some limitations in the model's abilities, leading to poor performance in the given situations.
"""

part6_bonus = r"""
**Your answer:**


Trying to improve the model's performance on the previous pictures, we made the following manipulations:

- In the occlusion demonstration picture, At first we cropped the tree from the image - yielding another misclassification of the child - dog class with confidence level of 0.25.
The change of classification indicated that the occlusion with the tree had some influence on the detection.
We assumed that in addition to the occlusion, model bias is involved due to the rotated face of the child (parallel to the ground), so in order to fix that - we now also performed
a counter-rotation to the image, resulted with a correct classification of the child - person class with confidence level of 0.50.

- In the model bias demonstration picture, Seeing that only the angular view was classified correctly - We rotated by 45 degrees the front and the back views of the cell phones.
The result was a correct classification of the back view - cell phone class with confidence level of 0.42 (which was a bit low for what we expected).
Unfortunately, the rotated front view was still not detected at all, a result that is probably related to training bias - where images of cell phones were taken mainly on angular views (not only rotated).

- In the blurring demonstration picture, we tried to sharpen the image and clarify it, using some common filters.
The result was still noisy and blurred, though it became a bit clearer, hence the model again misclassified it - this time it detected two objects, horse (same box as earlier, this time with lower confidence level of 0.68)
and a person (over the dog's tail area). From what we understand, the sharpening manipulation might had amplified some noise in this region of the image, which became now more noticable and mislead the model to detect another object. 

In summery, the manipulations we've performed indeed improved the model's performance over the poorly recognized images.
It involved some spatial manipulations like rotation and cropping, along with fine-tuning filters like the sharpening, resulting in a correct classification for the majority of the images.
It is worth to mention that for some situations (like the blurred dog image), a more sophisticated manipulation is required to correctly classify the required objects.
"""