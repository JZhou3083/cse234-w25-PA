import functools
import math
from typing import Callable, Tuple, List, Optional

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28

def linear(x:ad.Node, w: ad.Node, b:ad.Node) -> ad.Node:
    '''
    Linear: y = x @ W +b
    Shapes:
    - If x is (B,S, D), W must be (D,D2), b must be broadcastable to (B, S ,D2)
    - If x is (B,D), W must be (D, D2), b must be broadcastable to (B, D2)
    '''
    return ad.matmul(x, w)+b

def single_head_self_atten(
        X: ad.Node,
        wq:ad.Node, bq,
        wk: ad.Node, bk,
        wv: ad.Node, bv,
        d_k: int,
        *,
        mask = None,
        eps = 1e-9
        ):
    '''
    Compute single-head self attention from input X using your ops.
    X: (B,S,D)
    W*: (D,D) and b*:(1,1,D)
    returns: (B,S,D)
    '''
    Q = linear(X, wq,bq)
    K = linear(X,wk, bk)
    V = linear(X, wv, bv)

    return scaled_dot_product_attention(Q,K,V,d_k = d_k, mask = mask, eps= eps)

def scaled_dot_product_attention(
        Q,K,V,
        *,
        d_k: int,
        mask: Optional["ad.Node"] = None,
        eps: float = 1e-9
        )-> ad.Node:
    '''
    Single-head attention.
    Shapes:
      Q,K,V: (B,S,D)
      Returns: (B,S,D)
    '''
    KT = ad.transpose(K, dim0 = 1, dim1 = 2)

    scale = math.sqrt(d_k)
    scores = ad.matmul(Q,KT)/scale
    if mask is not None:
        scores = scores + mask

    return ad.matmul(ad.softmax(scores, dim=-1),V)


def feed_forward(X, w1, b1, w2, b2):

    hidden = ad.relu(linear(X,w1,b1))
    return linear(hidden, w2,b2)


def transformer(X: ad.Node, nodes: List[ad.Node],
                      model_dim: int, seq_length: int, eps, batch_size, num_classes) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    nodes: List[ad.Node] of parameters in the following order:
        0   w_q     (D,D)   1,  b_q     (1,1,D)
        2   w_k     (D,D)   3   b_k     (1,1,D)
        4   w_v     (D,D)   5   b_v     (1,1,D)
        6   w_o     (D,D)   7   b_o     (1,1,D)
        8   w1      (D,Dff) 9   b1      (1,1,Dff)
        10  w2      (Dff,D) 11  b2      (1,1,D)
        12 gamma1   (1,1,D) 13  beta1   (1,1,D)
        14 gamma2   (1,1,D) 15  beta2   (1,1,D)
        16 w_cls    (D,C)   17  b_cls   (1,C)

    model_dim   : D, Dimension of the model (hidden size).

    seq_length  : S, Length of the input sequence.
    eps         : LayerNorm epsilon
    batch_size  : B
    num_classes : C
    Returns
    -------
    output: ad.Node (B , C)
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """

    """TODO: Your code here"""
        # Unpack nodes
    (w_q, b_q,
     w_k, b_k,
     w_v, b_v,
     w_o, b_o,
     w1, b1,
     w2, b2,
     gamma1, beta1,
     gamma2, beta2,
     w_cls, b_cls) = nodes

    # Set d_k and d_v for single-head attention
    d_k = model_dim
    d_v = model_dim
    # 1) Self-attention output
    attn_out = single_head_self_atten(
        X, wq=w_q, bq=b_q, d_k=d_k, wk=w_k, bk=b_k, wv=w_v, bv=b_v, eps=eps
    )

    # 2) Attention output projection
    attn_proj = linear(attn_out, w_o, b_o)  # shape (B, S, D)

    # 3) Feed-forward network
    ff_hidden = linear(attn_proj, w1, b1)   # shape (B, S, Dff)
    ff_relu = ad.relu(ff_hidden)
    ff_out = linear(ff_relu, w2, b2)        # shape (B, S, D)

    # 4) Pool over sequence length
    pooled = ad.mean(ff_out, dim=1)         # shape (B, D)

    # 5) Classification head
    logits = linear(pooled, w_cls, b_cls)   # shape (B, C)

    return logits

def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    """TODO: Your code here"""

     # Probabilities
    probs = ad.softmax(Z)  # (B, C)
    
    # log(probs)
    log_probs = ad.log(probs)  # (B, C)
    
    # sum(y_one_hot * log_probs) over classes
    loss_per_sample = ad.sum_op(ad.mul(y_one_hot, log_probs), dim=(1,))  # (B,)

    # sum over batch
    total_loss = ad.sum_op(loss_per_sample, dim=(0,))  # scalar

    # multiply by -1
    total_loss = ad.mul_by_const(total_loss, -1.0)
    
    # average over batch
    loss = ad.div_by_const(total_loss, batch_size)  # scalar ()

    return loss

def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) ->Tuple[List[torch.Tensor], float]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    """TODO: Your code here"""
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size> num_examples:continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]

        # Compute forward and backward passes
        # TODO: Your code here
        # Forward + backward pass
        logits, loss, grads = f_run_model(X_batch, y_batch, model_weights)


        # Update weights and biases
        # TODO: Your code here
        # Hint: You can update the tensor using something like below:
        # W_Q -= lr * grad_W_Q.sum(dim=0)
        for w, g in zip(model_weights, grads):
            w -= lr * g  # in-place update

        # Accumulate the loss
        total_loss += loss.item() * X_batch.shape[0]


    # Compute the average loss

    average_loss = total_loss / num_examples
    print('Avg_loss:', average_loss)

    # TODO: Your code here
    # You should return the list of parameters and the loss
    print("Avg_loss:", average_loss)
    return model_weights, average_loss

def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # TODO: Tune your hyperparameters here
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10 #
    model_dim = 128 #
    eps = 1e-5

    # - Set up the training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # TODO: Define the forward graph.
    # Symbolic input and output variables
    X = ad.Variable(name="X")
    y_groundtruth = ad.Variable(name="y")

    # Initialize model weights BEFORE assigning them to ad.Variable
    np.random.seed(0)
    # weight initializations
    D = model_dim
    D_ff = model_dim  # or some other feed-forward dimension
    C = num_classes
    stdv = 1.0 / np.sqrt(D)
    W_Q_val    = np.random.uniform(-stdv, stdv, (D, D))
    b_Q_val    = np.random.uniform(-stdv, stdv, (1, 1, D))
    W_K_val    = np.random.uniform(-stdv, stdv, (D, D))
    b_K_val    = np.random.uniform(-stdv, stdv, (1, 1, D))
    W_V_val    = np.random.uniform(-stdv, stdv, (D, D))
    b_V_val    = np.random.uniform(-stdv, stdv, (1, 1, D))
    W_O_val    = np.random.uniform(-stdv, stdv, (D, D))
    b_O_val    = np.random.uniform(-stdv, stdv, (1, 1, D))
    W1_val     = np.random.uniform(-stdv, stdv, (D, D_ff))
    b1_val     = np.random.uniform(-stdv, stdv, (1, 1, D_ff))
    W2_val     = np.random.uniform(-stdv, stdv, (D_ff, D))
    b2_val     = np.random.uniform(-stdv, stdv, (1, 1, D))
    gamma1_val = np.ones((1, 1, D))
    beta1_val  = np.zeros((1, 1, D))
    gamma2_val = np.ones((1, 1, D))
    beta2_val  = np.zeros((1, 1, D))
    W_cls_val  = np.random.uniform(-stdv, stdv, (D, C))
    b_cls_val  = np.random.uniform(-stdv, stdv, (1, C))

    # create Variables and assign
    W_Q    = ad.Variable(name="W_Q");    W_Q.value = W_Q_val
    b_Q    = ad.Variable(name="b_Q");    b_Q.value = b_Q_val
    W_K    = ad.Variable(name="W_K");    W_K.value = W_K_val
    b_K    = ad.Variable(name="b_K");    b_K.value = b_K_val
    W_V    = ad.Variable(name="W_V");    W_V.value = W_V_val
    b_V    = ad.Variable(name="b_V");    b_V.value = b_V_val
    W_O    = ad.Variable(name="W_O");    W_O.value = W_O_val
    b_O    = ad.Variable(name="b_O");    b_O.value = b_O_val
    W1     = ad.Variable(name="W1");     W1.value = W1_val
    b1     = ad.Variable(name="b1");     b1.value = b1_val
    W2     = ad.Variable(name="W2");     W2.value = W2_val
    b2     = ad.Variable(name="b2");     b2.value = b2_val
    gamma1 = ad.Variable(name="gamma1"); gamma1.value = gamma1_val
    beta1  = ad.Variable(name="beta1");  beta1.value = beta1_val
    gamma2 = ad.Variable(name="gamma2"); gamma2.value = gamma2_val
    beta2  = ad.Variable(name="beta2");  beta2.value = beta2_val
    W_cls  = ad.Variable(name="W_cls");  W_cls.value = W_cls_val
    b_cls  = ad.Variable(name="b_cls");  b_cls.value = b_cls_val

    # final model_weights in correct order
    model_weights = [
        W_Q, b_Q,
        W_K, b_K,
        W_V, b_V,
        W_O, b_O,
        W1, b1,
        W2, b2,
        gamma1, beta1,
        gamma2, beta2,
        W_cls, b_cls
    ]

    y_predict = transformer(X, model_weights, model_dim, seq_length, eps, batch_size, num_classes)  # TODO: The output of the forward pass
    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)

    # TODO: Construct the backward graph.


    # TODO: Create the evaluator.
    grads: List[ad.Node] = ... # TODO: Define the gradient nodes here
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(model_weights):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        result = evaluator.run(
            input_values={
                # TODO: Fill in the mapping from variable to tensor


            }
        )
        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
        total_loss = 0.0
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size> num_examples:continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            logits = test_evaluator.run({
                # TODO: Fill in the mapping from variable to tensor


            })
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    model_weights: List[torch.Tensor] = [] # TODO: Initialize the model weights here
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
