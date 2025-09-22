from tqdm import tqdm
import time
from typing import Callable, Tuple, List, Optional
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms
import os
MODEL_DIR = os.path.join(os.getcwd(),'pa1','checkpoints')
max_len = 28

def linear(X:ad.Node, W: ad.Node, b :ad.Node = None) -> ad.Node:
    """_summary_

    Args:
        X (ad.Node): _description_
        W (ad.Node): _description_
        b (ad.Node): _description_

    Returns:
        ad.Node: _description_
    """
    return ad.add(ad.matmul(X,W), b) if b else ad.matmul(X,W)

def single_head_atten(X: ad.Node, W_Q: ad.Node, W_K: ad.Node, W_V: ad.Node, model_dim: int) -> ad.Node:
    """A single-Head Attention Layer

    Args:
        X (ad.Node): Input Node
        W_Q (ad.Node): Weight Matrix of query
        W_K (ad.Node): weight Matrix of K
        W_V (ad.Node): Weight Matrix of V
        model_dim (int): Embedding length

    Returns:
        ad.Node: attentions
    """
    # Project inputs
    Q = linear(X, W_Q)
    K = linear(X, W_K)
    V = linear(X, W_V)

    # Attention weights (scaled dot-product)
    attn_scores = linear(Q, ad.transpose(K, dim0= -1, dim1= -2))/np.sqrt(model_dim)
    attn_weights = ad.softmax(attn_scores, dim = -1)

    atten = ad.matmul(attn_weights, V)

    return atten

def encoder(X: ad.Node, nodes :List[ad.Node], model_dim: int ) -> ad.Node:
    """A encoder layer combining self-attention and feed-forward network

    Args:
        X (ad.Node): input node
        nodes (List[ad.Node]): model_weights
        model_dim (int): embedding length

    Returns:
        ad.Node: _description_
    """
    W_Q, W_K, W_V, W_O, W1, W2, b1, b2 = nodes
    attn_scores = single_head_atten(X, W_Q, W_K, W_V, model_dim)

    # Output projection
    Z = ad.matmul(attn_scores, W_O)

    # Fedforward layer
    H = ad.relu(ad.add(ad.matmul(Z, W1), b1))
    logits =ad.add(ad.matmul(H, W2),b2)
    return logits

def transformer(X: ad.Node, nodes: List[ad.Node],
                      model_dim: int, seq_length: int, eps, batch_size, num_classes) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """

    W_Q, W_K, W_V, W_O, W1, W2, b1, b2 = nodes
    logits = encoder(X, [W_Q, W_K, W_V, W_O, W1, W2, b1, b2],model_dim)

    # Average over sequence length for classification
    output = ad.mean(logits, dim = 1) # shape (batch_size, num_classes)

    return output

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
    softmax = ad.softmax(Z,dim =1)

    log_probs = ad.log(softmax +1e-9) # avoid log(0)
    loss = ad.mul_by_const(ad.sum_op(y_one_hot*log_probs,dim= (1,))/ batch_size, -1)

    return loss

def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
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

    for i in tqdm(range(num_batches)):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size> num_examples:continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]

        # Compute forward and backward passes
        logits, loss_val, *grads = f_run_model(X_batch, y_batch,model_weights)

        # Update weights and biases
        for w, g in zip(model_weights, grads):
            # print(f"[DEBUG] Weight shape: {w.shape}, Grad shape: {g.shape}\n")
            while g.ndim > w.ndim:  # likely batch dimension
                g = g.sum(dim=0)  # sum over batch or seq_length
            w -= lr*g
        # Hint: You can update the tensor using something like below:
        # Accumulate the loss
        total_loss += loss_val * batch_size


    # Compute the average loss
    average_loss = total_loss / num_examples
    # print('Avg_loss:', average_loss)

    # You should return the list of parameters and the loss
    return model_weights, average_loss

def train_model(use_last_model = None):
   """Train a transformer-based classifier on MNIST."""
   # Set up model params
   input_dim = 28           # Each row of the MNIST image
   seq_length = max_len     # Number of rows in the MNIST image
   num_classes = 10
   model_dim = 128
   eps = 1e-5
   # Training settings
   num_epochs = 10
   batch_size = 50
   lr = 0.008
   # Define variables for graph
   X_var = ad.Variable(name="X")  # (batch_size, seq_length, input_dim)
   y_groundtruth = ad.Variable(name="y")
   W_Q = ad.Variable(name="W_Q")
   W_K = ad.Variable(name="W_K")
   W_V = ad.Variable(name="W_V")
   W_O = ad.Variable(name="W_O")
   W_1 = ad.Variable(name="W_1")
   W_2 = ad.Variable(name="W_2")
   b_1 = ad.Variable(name="b_1")
   b_2 = ad.Variable(name="b_2")
   nodes = [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]

   # Forward graph
   y_predict: ad.Node = transformer(
       X_var, nodes, model_dim, seq_length, eps, batch_size, num_classes
   )
   loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)
   # Backward graph
   grads: List[ad.Node] = ad.gradients(loss, nodes)
   # Evaluators
   evaluator = ad.Evaluator([y_predict, loss, *grads])
   test_evaluator = ad.Evaluator([y_predict])
   # --- Load the dataset ---
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])
   train_dataset = datasets.MNIST(
       root="./data", train=True, transform=transform, download=True
   )
   test_dataset = datasets.MNIST(
       root="./data", train=False, transform=transform, download=True
   )
   X_train = train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
   y_train = train_dataset.targets.numpy()
   X_test = test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
   y_test = test_dataset.targets.numpy()
   encoder = OneHotEncoder(sparse_output=False)
   y_train = encoder.fit_transform(y_train.reshape(-1, 1))
   # --- Initialize model weights ---
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
   model_weights: List[torch.Tensor] = [
       torch.tensor(W_Q_val, dtype=torch.float64),
       torch.tensor(W_K_val, dtype=torch.float64),
       torch.tensor(W_V_val, dtype=torch.float64),
       torch.tensor(W_O_val, dtype=torch.float64),
       torch.tensor(W_1_val, dtype=torch.float64),
       torch.tensor(W_2_val, dtype=torch.float64),
       torch.tensor(b_1_val, dtype=torch.float64),
       torch.tensor(b_2_val, dtype=torch.float64),
   ]
   # --- Load repvious weights if requested ---
   if use_last_model is not None:
        try:
           print(f"Loading previous model from {use_last_model}...")
           model_weights = torch.load(os.path.join(MODEL_DIR, f"{use_last_model}.pth"))
           print("Model weights loaded.")
        except:
           print('None existed model name, Training with this Name')

   # --- Runner functions ---
   def f_run_model(X_batch, y_batch, model_weights):
       """Forward + backward pass."""
       result = evaluator.run(
           input_values={
               X_var: X_batch,
               y_groundtruth: y_batch,
               W_Q: model_weights[0].detach(),
               W_K: model_weights[1].detach(),
               W_V: model_weights[2].detach(),
               W_O: model_weights[3].detach(),
               W_1: model_weights[4].detach(),
               W_2: model_weights[5].detach(),
               b_1: model_weights[6].detach(),
               b_2: model_weights[7].detach(),
           }
       )
       return result
   def f_eval_model(X_val, model_weights: List[torch.Tensor]):
       """Forward pass only for predictions."""
       num_examples = X_val.shape[0]
       num_batches = (num_examples + batch_size - 1) // batch_size
       all_logits = []
       for i in tqdm(range(num_batches)):
           start_idx = i * batch_size
           end_idx = min(start_idx + batch_size, num_examples)
           if end_idx - start_idx < batch_size:
               continue
           X_batch = X_val[start_idx:end_idx, :max_len]
           logits = test_evaluator.run({
               X_var: X_batch,
               W_Q: model_weights[0].detach(),
               W_K: model_weights[1].detach(),
               W_V: model_weights[2].detach(),
               W_O: model_weights[3].detach(),
               W_1: model_weights[4].detach(),
               W_2: model_weights[5].detach(),
               b_1: model_weights[6].detach(),
               b_2: model_weights[7].detach(),
           })
           all_logits.append(logits[0])
       concatenated_logits = np.concatenate(all_logits, axis=0)
       predictions = np.argmax(concatenated_logits, axis=1)
       return predictions
   # --- Training loop ---
   X_train, X_test = torch.tensor(X_train), torch.tensor(X_test)
   y_train, y_test = torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
   best_val_accu = 0.0
   best_weights = None
   for epoch in range(num_epochs):
       X_train, y_train = shuffle(X_train, y_train)
       print("Traning model:")
       model_weights, loss_val = sgd_epoch(
           f_run_model, X_train, y_train, model_weights, batch_size, lr
       )
       print('Training completed, evaluateing the model: ')
       predict_label = f_eval_model(X_test, model_weights)
       acc = np.mean(predict_label == y_test.numpy())
       print(
           f"Epoch {epoch}: test accuracy = {acc}, "
           f"loss = {loss_val}"
       )
       if acc > best_val_accu:
           best_val_accu = acc
           best_weights = [w.clone() for w in model_weights]
   predict_label = f_eval_model(X_test, model_weights)

   # Ensure MODEL_DIR exists
   os.makedirs(MODEL_DIR, exist_ok=True)
   filename = f"ViT_Epochs{num_epochs}_lr{lr}_acc{best_val_accu}.pth"

   print(f"Saving best model to {os.path.join(MODEL_DIR, filename)}")
   torch.save(best_weights, os.path.join(MODEL_DIR, filename))
   return np.mean(predict_label == y_test.numpy())

if __name__ == "__main__":
    print(f"Final test accuracy: {train_model('ViT_Epochs10_lr0.008_acc0.8013')}")
