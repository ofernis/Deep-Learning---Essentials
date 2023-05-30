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
    1. No, we don't need to materialize the Jacobian tensor $\pderiv{\mat{Y}}{\mat{X}}$ in order to calculate the downstream gradient $\delta\mat{X}$
    
        Instead, we can leverage the chain rule to compute the gradient efficiently:
        $$
        \pderiv{L}{\mat{X}} = \pderiv{L}{\mat{Y}} \pderiv{\mat{Y}}{\mat{X}}
        $$
        - $\pderiv{L}{\mat{Y}}$ is given to us.
        - $\pderiv{\mat{Y}}{\mat{X}}$ is W^T (because \mat{Y}=\mat{X} \mattr{W} + \vec{b}).
        

1. For the Jacobian tensor $\pderiv{\mat{Y}}{\mat{W}}$:
    1. The shape of this tensor will be (64, 512, 1024, 512).
    1. This Jacobian is/isnot sparse. why and which elements?
    1. Given the gradient of the output 
    
# TODO
"""

part1_q2 = r"""
**Your answer:**

**Yes**, backpropagation is required in order to train neural networks.

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

**1.1**: Our baseline is no dropout setting, with pre-tuned hyperparameters for an overfitted model (overfitting can be inferred from the graphs, where the train accuracy is much higher than the test accuracy).
We see that the training accuracy is higher without dropout in comparison to using dropout, but the test accuracy is lower for the same dropout settings comparison.
This fits what we expected, as the dropout method "omits" training information and thus decrease the training accuracy, but due to its randomness, it makes the network less sensitive to the influence of single
neurons and therefore demonstrate better generalization. Looking at our graphs, it is worth mention that as the dropout level increases, the training accuracy decreases (due to the reason we mentioned earlier).
An oppisite effect is seen in the test accuracy graphs, where the lower accuracy corresponds to higher dropout level (0.8) rather than the lower dropout level (0.4).

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

3. **Generalization**: SGD's inherent stochasticity during training acts as a form of regularization, preventing overfitting and promoting better generalization.

4. **Avoiding local minima**: Deep learning models often have complex, high-dimensional loss landscapes with numerous local minima. GD relies on the entire dataset and may get trapped in suboptimal local minima.
SGD's stochastic nature provides a mechanism for escaping these suboptimal solutions.

###  Answer 4
1. **Yes**, this method will produce a gradient equivalent to GD. We remember that:
$$
L(\vec{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \ell(\vec{y}^i, \hat{\vec{y}^i}) + R(\vec{\theta})
$$

We can re-write the above using $K$ disjoint batches (assume we keep the same order for the $i$s):
$$
L(\vec{\theta}) 
= \frac{1}{N} \sum_{j=1}^{K} \sum_{i\ in B_j} \ell(\vec{y}^i, \hat{\vec{y}^i}) + R(\vec{\theta})
$$
Thus, the forward calculation is equivalent.

Now, we have $L(\vec{\theta})$ and wish to calculate $frac{\partial L}{\partial \vec{x}}$, we don't need the dataset anymore for this calculation, so it is equivalent to GD's.

2. 

"""

part2_q4 = r"""
**Your answer:**

Using the **chain rule** we get:
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

We can compute the above using only 2 variables to store the calculations - accumalated derivative $\delta$ and current derivative $d$:
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
*   Because the symptoms are non-leathal, and confirm the diagnosis.
*   So, we would choose a threshold with more weight towards lower FPR.
*   This means that less patients will falsly get diagnosed positive for the illness and get the expensive tests to confirm the diagnosis.
*   By that, we reduced the cost while maintaining the low loss of life.

**Scenatio 2 - lethal illness and no clear symptoms**  
*   The illness is lethal and the symptoms are not clear, so we must allow more false-negatives to be able to save lives.
*   Thus, we should choose the threshold with more weight towards the low FNR.
*   This means that more patients will be falsly diagnosed for the illness, and start the treatment to save their lives
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


1. Assuming no bias added to the layers, for the regular block we get $256 \times 3 \times 3 + 256 \times 3 \times 3 = 4608$
and for the bottleneck we get $256 \times 1 \times 1 + 64 \times 3 \times 3 + 64 \times 1 \times 1 = 896$.

2. assuming again no bias and assuming we have an input feature map with dimensions $H \times W \times C$, 
We know that the number of Floating Point Operations required for each convolutional layer equals to the sum of $kernel_size \times kernel_size \times C$ multiplications
per output element and $kernel_size \times kernel_size \times C - 1$ (excluding bias term) additions per output element. 
Thus for the whole blocks, the ratio of required FLOPS number is roughly the same as the number of parameters ratio - 1 : 5 (bottleneck : regular residual).

3. Well explain the two requested aspects:

- Spatial Combination within Feature Maps:  \\
    
& Regular Residual Block - In a regular residual block, the spatial combination within feature maps is achieved through two successive 3x3 convolutions.
The first convolutional layer processes the input feature maps and induces intermediate feature maps, while the second convolutional layer further operates on the intermediate feature maps to produce the final output.
The skip connection in the regular block directly adds the input feature maps to the output of the second convolutional layer.
This addition operation allows for spatial combination by adding the local spatial information preserved in the input to the transformed feature maps generated by the convolutional layers.

\\
& Bottleneck Residual Block - The bottleneck residual block uses a 1x1 convolutional layer for dimension reduction, followed by a 3x3 convolutional layer, and finally a 1x1 convolutional layer for dimension restoration. 
As we learned, the 1x1 convolution in the bottleneck block helps reduce the computational complexity by decreasing the number of channels before performing the more expensive 3x3 convolution.
This dimension reduction effectively reduces the spatial resolution within the feature maps, limiting the spatial combination within the block.

- Combination across Feature Maps: \\
    
& Regular Residual Block - In the regular residual block, the combination across feature maps is facilitated by the skip connection that directly adds the input feature maps to the output of the second convolutional layer. 
This addition operation combines the feature maps element-wise, allowing the information from different channels to be combined and interact. 
By summing the input and output, the regular block can effectively integrate information across feature maps.

\\
& Bottleneck Residual Block - The bottleneck residual block also provides a means to combine information across feature maps, but to a lesser extent compared to the regular block.
The bottleneck block achieves this through the 1x1 convolutional layers. The first 1x1 convolution reduces the number of channels, effectively reducing the number of feature maps.
The subsequent 1x1 convolution restores the dimensionality back to the original number of channels.
While this dimension reduction and restoration process allows some interaction across feature maps, the overall combination is less direct and may not fully exploit the interaction potential compared to the regular block.

\\
& To conclude, the regular residual block provides stronger spatial and feature map combination capabilities due to its consecutive 3x3 convolutions and direct skip connection.
The bottleneck residual block sacrifices some of these combination abilities in favor of reducing computational complexity by using 1x1 convolutions for dimension reduction and restoration.
The specific trade-offs between computational efficiency and combination abilities can be beneficial depending on the requirements of the neural network architecture and the available computational resources. 
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

1. The model managed to detect the boundaries of the objects in the pictures generally well, but its classifications were mostly wrong. 
It can be seen in the first picture, where two dolphins were classified as persons and the tail of the third one was classified as a surfboard.
In the second picture, two dogs were classified as cats and the cat was not detected at all.
We'll also notice that the dog that was classified correctly had a score of 0.51, which implies a significant level of uncertainty.
All in all, the model's performance was not good for these 2 pictures.

2. We'll examine numerous reasons for the model failures and suggest compatible solutions for the issues:
- A possible reason for the model failure is lack of light/dark images, which makes objects' features less noticable and produces difficulties in the models classification.
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