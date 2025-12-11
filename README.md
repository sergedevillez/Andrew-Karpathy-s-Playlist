# General description
This is a project formed of multiples small projects. The purpose of this project is to gain a basic understanding of languages models and their terms by following the multiple videos made by Andrej Karpathy on [Youtube](https://www.youtube.com/@AndrejKarpathy). More sources may be added later accordingly to understanding and interest.
See the [Youtube playlist](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

# Projects
## 1. Build Micrograd
Source: [Youtube](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
Micrograde is a tiny autograd (Automatic gradient) engine and neural network library implemented in pure Python. 
it implement backpropagation. It allow to iteratively tune the weights of the neural network to minimize the loss function and terefore improve the performance of the model on a given task.
**Example usage**
```python
from micrograd.engine import Value
# This show some "random basic operations. Behind the scene, micrograd will build a computational graph to keep track of the operations applied to the Value objects. (ie: c is a value, c is the result of a computation of a and b, etc.) => c is a child node of a and b.
# When calling backward on the final output g, micrograd will traverse the computational graph in reverse order, applying the chain rule to compute the gradients of g with respect to a and b.
a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}')  # Output: 24.7041, the outcome of the forward pass
g.backward()
print(f'{a.grad:.4f}')  # Output: 138.6264, the numerical value of dg/da
print(f'{b.grad:.4f}')  # Output: 645.5200, the numerical value of dg/db this is important to see how the output g changes with respect to small changes in b
```

Neural network are just mathematical expressions that take input data as an input and weights of a neural network as an input and output the predictions of the neural net or the loss function. They are just are certain class of mathematical functions that are particularly well-suited for modeling complex relationships in data.

Micrograd does something that would be excessive in production; it is a scalar valued autograd engine. In other words, it works on individual numbers (scalars) rather than on arrays or tensors. This makes it simpler and easier to understand, but also less efficient for large-scale computations. It is only done so we understand the core concepts of autograd and backpropagation without the added complexity of handling multi-dimensional arrays.

## 2. Build Makemore
Source: [Youtube](https://www.youtube.com/watch?v=PaCmpygFfXo)
- Introduce the bigram character level language model
- How to train the model
- How to sample from the model
- How we evaluate the quality of the model using the negative log likelihood loss (NLL Loss)
- Then we evaluate the model on two way.
    - We counted the frquency of each bigram in the training dataset then normalized them to get the true probabilities of each bigram.
    - We used the negative log likelihood as a guide to optimizing the count arrays to minimize the loss.

Source: [Youtube - Part 2](https://www.youtube.com/watch?v=TCH_1BHY58I)

Source: [Youtube - Part 3](https://www.youtube.com/watch?v=P6sfmUTpUmc)

Source: [Youtube - Part 4](https://www.youtube.com/watch?v=q8SA3rM6ckI)

Source: [Youtube - Part 5](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)

Construction of:
- Bigram (one character predict the enxt one wit a lookup table of counts)
- Bags of words
- MLP (Multi-layer perceptron)
- RNN (Recurrent Neural Network)
- GRU (Gated Recurrent Unit)
- Transformer (equivalent to GPT 2;with of course far less data and training)

=> This construct single words (in this case: "name"-like single words)

## 3. Build GPT
Source: [Youtube](https://www.youtube.com/watch?v=kCc8FmEb1nY)
Building a transformer purely decoder which take an input (text from Shakespear) and learn of to complete a given text accoding to its training. 

1. Get a a training set. In this case, a shakespear dataset.
2. Tokenized the dataset into tokens. Here :
    - Get the unique characters
    - Create a lookup table for convertir to and from the tokens' table.
3. Plug the token in the language model by using an embedding table. 65 embedding tokens would make 65 rows. Basically, each rows is a list of parameters for each unique characters. => a vector that feeds into the transformer. The parameters will be created as the vector evolve according to each usage new or repeated of the token in a sentence/string.

=> This construct multiples sentences (in this case: shakespear-like sentences.)

## 4. Build GPT tokenizer
Source: [Youtube](https://www.youtube.com/watch?v=zduSFxRajkE)
Opposed to the previous project, who was using a somewhat naïve approach to tokenization (single characters) in reality, the tokens are almost always chunk of characters. Theses token are composed with algorithms such as the bite-pair algorithm.



## Dictionary

### First part
- Autograd: Automatic gradient. Technique to automatically compute the gradient of a function.
- Backpropagation: Method that allow you to efficiently compute the gradient of a loss function with respect to the weights of a neural network.
- Forward pass: the process in which you pass an input through a neural network to obtain an output.
- Derivative: Measure of how a function changes as its input changes. In the context of neural networks, derivatives are used to compute gradients during backpropagation.
- Slope: How much the output of a function changes after a smal (or not) change in input. A slope is negative if the impact of the change is towards a decrease of the output, and positive if it is towards an increase of the output.
- Gradient: Vector that contains all the partial derivatives of a function with respect to its inputs. In neural networks, gradients are used to update the weights during training.
- Gradient checking: Technique used to verify the correctness of the gradients computed by backpropagation. It involves comparing the analytically computed gradients with numerically approximated gradients.
- Topological sort: Linear ordering of nodes in a directed graph such that for every directed edge from node A to node B, node A comes before node B in the ordering. In the context of computational graphs, topological sorting is used to determine the order in which nodes should be processed during backpropagation.
- Loss function: Function that measures the difference between the predicted output of a neural network and the actual target values. The goal of training is to minimize the loss function, get closer to zero.
- Learning rate decay: Technique used to gradually decrease the learning rate during training. This helps to stabilize the training process and can lead to better convergence.
- Regularization: Technique used to prevent overfitting in neural networks by adding a penalty term to the loss function. Common regularization techniques include L1 and L2 regularization, dropout, and early stopping.
- Training loop: Iterative process of training a neural network, which typically involves the following steps: forward pass, loss computation, backward pass (backpropagation), and weight update.
- Broadcasting: Technique used in tensor operations to make arrays with different shapes compatible for element-wise operations. It involves "stretching" the smaller array along the dimensions of the larger array without actually copying data.
    - Broadcasting rules:
        1. If the arrays have a different number of dimensions, the shape of the smaller-dimensional array is padded with ones on the left side until both shapes have the same length.
            - Example: 
                - Array A shape: (4,3,2)
                - Array B shape: (3,2)
                - Padded Array B shape: (1,3,2)
        2. The sizes of the arrays are compared element-wise from right to left. Two dimensions are compatible when:
            - They are equal, or
            - One of them is 1
                - Example:
                    - Padded Array B shape: (1,3,2)
                    - Array A shape: (4,3,2)
                    - Comparison from right to left :
                        - 2 and 2 are equal => compatible
                        - 3 and 3 are equal => compatible
                        - 1 and 4 => one of them is 1 => compatible
        3. If the sizes of the arrays are not compatible, an error is raised.
            - Example:
                - Array A shape: (4,3,2)
                - Array B shape: (2,2)
                - Comparison from right to left :
                    - 2 and 2 are equal => compatible
                    - 3 and 2 are not equal and neither is 1 => not compatible => error
        4. If one of the dimensions is 1, the array is "stretched" along that dimension to match the size of the other array.
            - Example:
                - Padded Array B shape: (1,3,2)
                - Array A shape: (4,3,2)
                - Stretched Padded Array B shape: (4,3,2)
        5. The resulting array has a shape that is the maximum size along each dimension of the input arrays.
            - Example:
                - Array A shape: (4,3,2)
                - Array B shape: (1,6,2)
                - Resulting array shape: (4,6,2)
    - Rules to follow:
        - Align shapes to the right
            - Example:
                - Array A shape: (4,3,2)
                - Array B shape:   (3,2)
        - All dimensions must be equal, equal to 1 or missing
        - Missing dimensions are treated as 1
        - Resulting shape is the maximum size along each dimension
- Statistical modeling: Process of using statistical methods to analyze and interpret data in order to make predictions or draw conclusions about a population based on a sample.
- Monotonic function: Function that is either entirely non-increasing or non-decreasing over its domain. In other words, a monotonic function preserves the order of its input values.
- Smoothing: Technique used to reduce noise or fluctuations in data, often by applying a mathematical function or algorithm to the data points. This is a fix that allow the model to not faces -inf loss when a single data point is mispredicted with high confidence.
- Gradient-based optimization: Class of optimization algorithms that use the gradient of the loss function to update the model parameters in order to minimize the loss. Examples include Stochastic Gradient Descent (SGD), Adam, and RMSprop.
- Softmax: Mathematical function that converts a vector of real numbers into a probability distribution. It is commonly used in the output layer of neural networks for multi-class classification problems.
    - In others actual usage: Exponentiation of each element followed by normalization.

### Second part
- Multi Layer Perceptron (MLP): Type of feedforward neural network that consists of multiple layers of neurons, including an input layer, one or more hidden layers, and an output layer. Each neuron in a layer is connected to every neuron in the next layer.
- earning rate decay: Technique used to gradually decrease the learning rate during training. This helps to stabilize the training process and can lead to better convergence.
- Training split: Around 80% of the dataset used to train the model.
- Dev/Validation split: Around 10% of the dataset used to tune hyperparameters and evaluate the model during training. (by example, size of hyperlayer, size of embedding, strength of regularization, etc.)
- Test split: Around 10% of the dataset used to evaluate the final performance of the model after training is complete. Evaluation on the test split is only allowed a few times otherwise the model may overfit to the test set.