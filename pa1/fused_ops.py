from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self,
        node_A: Node,
        node_B: Node,
        normalized_shape: List[int],
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        a = input_values[0]
        b = input_values[1]
        # normalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]
        # Perform matrix multiplication
        matmul_result = torch.matmul(a, b)
        # Compute mean and variance for layer normalization
        mean = matmul_result.mean(dim=-1, keepdim=True)
        variance = matmul_result.var(dim=-1, keepdim=True, unbiased=False)
        # Normalize the result
        layernorm_result = (matmul_result - mean) / torch.sqrt(variance + eps)
        return layernorm_result

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        """TODO: your code here"""
        A, B = node.inputs
        eps = node.attrs['eps']
        normalized_shape = node.attrs['normalized_shape']
        dim = tuple(range(-len(normalized_shape),0))

        # Forward pass computations needed for backward
        Z = matmul(A, B)
        mean_Z = mean(Z, dim=dim, keepdim=True)
        var_Z = mean(power(Z - mean_Z, 2), dim=dim, keepdim=True)
        std_Z = sqrt(var_Z + eps)
        Z_hat = (Z - mean_Z) / std_Z

        # Gradient wrt Z (layernorm backward)
        g = output_grad
        g_mean = mean(g, dim=dim, keepdim=True)
        gzh_mean = mean(g * Z_hat, dim=dim, keepdim=True)

        grad_Z = (g - g_mean - Z_hat * gzh_mean) / std_Z

        # Gradients wrt A and B (matmul backward)
        B_T = transpose(B, -2, -1)
        A_T = transpose(A, -2, -1)

        grad_A = matmul(grad_Z, B_T)
        grad_B = matmul(A_T, grad_Z)

        return [grad_A, grad_B]



class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self,
        node_A: Node,
        node_B: Node,
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        a = input_values[0]
        b = input_values[1]
        dim = node.attrs["dim"]
        # Perform matrix multiplication
        matmul_result = torch.matmul(a, b)
        # Apply softmax
        softmax_result = torch.nn.functional.softmax(matmul_result, dim=dim)
        return softmax_result

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        A , B  = node.inputs
        dim = node.attrs['dim']

        # forward pass to get matmul output
        Z = matmul(A, B)

        # compute gradient wrt Z
        softmax_out = softmax(Z)
        dot = sum_op(output_grad * softmax_out, dim=dim, keepdim=True)
        grad_Z = softmax_out * (output_grad - dot)

        # gradients wrt A and B (matmul backward)
        B_T = transpose(B, -2, -1)
        A_T = transpose(A, -2, -1)

        grad_A = matmul(grad_Z, B_T)
        grad_B = matmul(A_T, grad_Z)

        return [grad_A, grad_B]

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()