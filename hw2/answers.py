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
    1. This Jacobian is sparse. The only elements that are not zero, correspond to the an index (i,j,i,k),
        meaning that this is the derivative of the sample i.
    1. No, we don't need to materialize the Jacobian tensor $\pderiv{\mat{Y}}{\mat{X}}$ in order to calculate the downstream gradient $\delta\mat{X}$\\
        Instead, we can use the chain rule to compute the gradient:\\
        $\pderiv{L}{\mat{X}} = \pderiv{L}{\mat{Y}} \pderiv{\mat{Y}}{\mat{X}}$ \\
        $\pderiv{L}{\mat{Y}}$ is given to use and:\\
        

1. For the Jacobian tensor $\pderiv{\mat{Y}}{\mat{W}}$:
    1. The shape of this tensor will be (64, 512, 1024, 512).
    1. This Jacobian is/isnot sparse. why and which elements?
    1. Given the gradient of the output 
"""

part1_q2 = r"""
**Your answer:**



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
    wstd = 0.1
    lr_vanilla = 0.025
    lr_momentum = 0.015
    lr_rmsprop = 0.0003
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
    lr = 0.02
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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

To conclude, if this happends for each epoch with decreasing $\epsilon$, then the behavior will be as described.

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
In SGD, only a single training example (or a mini-batch) is randomlly selected and used to compute the gradient and update the parameters.

2. **Efficiency**: As GD considers the entire dataset for each parameter update, it can be computationally expensive, especially for large datasets.
SGD is more efficient compared to GD as it uses a single example for the parameter update.

3. **Convergence**: GD converges to the global minimum if the loss function is convex, the convergence is generally slower as it takes more steps to reach the minimum.
SGD might not converge to the global minimum due to the randomness introduced by selecting the examples.
While it might not reach the optimal solution, it often finds a good enough solution in practice and does that faster than GD.

4. **Noise and Regularization**: GD uses the entire dataset, resulting a smoother parameter updates.
SGD intoduces randomness to the parameter updates, making it more noisy, but allowing it to escape local minima sometimes, making it an implicit form of regularization.

### Answer 3
SGD is commonly used in the practice of deep learning for several reasons:
1. **Efficiency with large datasets**: Deep learning often involves working with massive datasets that may not fit entirely into memory.
SGD's ability to update parameters based on individual examples (or mini-batches) makes it more efficient.

2. **Faster convergence**: While SGD's updates tends to be noisier than GD's, it can actually lead to faster convergence in practice.

3. **Generalization*: SGD's inherent stochasticity during training acts as a form of regularization, preventing overfitting and promoting better generalization.

4. **Avoiding local minima**: Deep learning models often have complex, high-dimensional loss landscapes with numerous local minima. GD relies on the entire dataset and may get trapped in suboptimal local minima.
SGD's stochastic nature provides a mechanism for escaping these suboptimal solutions.

"""

part2_q4 = r"""
**Your answer:**

HI


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
    raise NotImplementedError()
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
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""