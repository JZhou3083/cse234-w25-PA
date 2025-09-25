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

        B_T = transpose(B, -2, -1)
        A_T = transpose(A, -2, -1)

        INTER =  matmul(A,B)
        mean_inter = mean(INTER, dim = dim, keepdim= True)
        inter_minus_mean = INTER- mean_inter
        var_INTER = mean(power(inter_minus_mean,2), dim = dim, keepdim = True)
        std_INTER = sqrt(var_INTER +eps)

        # normalized input
        INTER_hat = inter_minus_mean/ std_INTER

        # gradient wrt input
        g = output_grad
        g_mean =mean(g*INTER_hat, dim = dim, keepdim =True)
        ginthat_mean = mean(g * INTER_hat, dim=dim, keepdim=True)

        grad_int = (g - g_mean - INTER_hat * ginthat_mean) / std_INTER

        grad_A = matmul(grad_int,B_T)
        grad_B = matmul(A_T, grad_int)

        return [grad_A, grad_B]






        # grad_B_final
        mean_gradB = mean(grad_B, dim = dim, keepdim=True)
        gradB_minus_mean = grad_B- mean_gradB
        var_gradB = mean(power(gradB_minus_mean, 2), dim=dim, keepdim=True)
        std_gradB = sqrt(var_gradB + eps)

        gradB_hat = gradB_minus_mean/std_gradA
        gBhat_mean = mean(g*gradB_hat, dim = dim, keepdim=True)

        grad_B_final = (g - g_mean - gradB_hat * gBhat_mean) / std_gradB

        return [grad_A_final, grad_B_final]





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
        """TODO: your code here"""
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        """TODO: your code here"""
        raise NotImplementedError

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()