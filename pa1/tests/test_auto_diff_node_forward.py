from typing import List

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import auto_diff as ad

def check_compute_output(
    node: ad.Node, input_values: List[torch.Tensor], expected_output: torch.Tensor
) -> None:
    output = node.op.compute(node, input_values)
    output = torch.nan_to_num(output, nan=0.0)
    expected_output = torch.nan_to_num(expected_output, nan=0.0)
    torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=1e-4)

def test_add():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.add(x1, x2)

    check_compute_output(
        y, 
        [
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        ],
        torch.tensor([[1.80, 2.70, 0.40, 3.40], [0.90, 6.60, -2.60, 6.20]]),
    )

def test_add_by_const():
    x1 = ad.Variable("x1")
    y = ad.add_by_const(x1, 2.0)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[1.0, 4.0, 2.5, 5.4], [2.3, 2.0, -3.8, 5.1]]),
    )

def test_mul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.mul(x1, x2)

    check_compute_output(
        y,
        [
            torch.tensor([[1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        ],
        torch.tensor([[2.80, 1.40, -0.05, 0.00], [0.18, 0.00, -18.56, 9.61]]),
    )

def test_mul_by_const():
    x1 = ad.Variable("x1")
    y = ad.mul_by_const(x1, 2.7)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[-2.70, 5.40, 1.35, 9.18], [0.81, 0.00, -15.66, 8.37]]),
    )

def test_greater():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.greater(x1, x2)
    check_compute_output(
        y,
        [
            torch.tensor([[1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[2.5, 4.0, -0.1, 0.1], [-8.0, 5.0, -2.5, -1.0]]),
        ],
        torch.tensor([[0, 0, 1, 1], [1, 0, 0, 1]]).float()
    )

def test_sub():
    x1 =ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.sub(x1, x2)
    check_compute_output(
        y, 
        [
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[2.5, 4.0, -0.1, 0.1], [-8.0, 5.0, -2.5, -1.0]]),
        ],

        torch.tensor([[-3.5, -2.0, 0.6, 3.3], [8.3, -5.0, -3.3, 4.1]]),

    )

def test_zeros_like():
    x1 = ad.Variable("x1")
    y = ad.zeros_like(x1)
    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1],[6,2,32,3]])],
        torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],[0,0,0,0]]),
    )

def test_ones_like():
    x1 = ad.Variable("x1")
    y = ad.ones_like(x1)
    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1],[6,2,32,3]])],
        torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],[1,1,1,1]]),
    )

def test_sum_op():
    x1 = ad.Variable('x1')
    y1_keepdim = ad.sum_op(x1,dim=(1,), keepdim=True)

    y1_nokeepdim = ad.sum_op(x1, dim=(1,), keepdim=False)
    y2_keepdim = ad.sum_op(x1,dim=(0,1), keepdim=True)
    y2_nokeepdim = ad.sum_op(x1, dim=(0,1), keepdim=False)
    check_compute_output(
        y1_keepdim,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1],[6,2,32,3]])],
        torch.tensor([[4.9], [-2.4], [43.0]])
    )

    check_compute_output(
        y1_nokeepdim,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1],[6,2,32,3]])],
        torch.tensor([4.9, -2.4, 43.0])
    )
    check_compute_output(
        y2_keepdim,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1],[6,2,32,3]])],
        torch.tensor([[45.5]])
    )
    check_compute_output(
        y2_nokeepdim,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1],[6,2,32,3]])],
        torch.tensor(45.5)
    )

def test_expand_as():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.expand_as(x1, x2)

    check_compute_output(
        y,
        [
            torch.tensor([[1.0], [3.0], [5.0]]),
            torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]),
        ],
        torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]]),
    )

def test_expand_as_3d():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.expand_as_3d(x1, x2)

    check_compute_output(
        y, 
        [
            torch.tensor([[1.0], [3.0], [5.0]]), #shape (3,1)
             torch.tensor([[[7.0, 8.0, 9.0]], 
                          [[10.0, 11.0, 12.0]],
                          [[13.0, 14.0, 15.0]]]) # shape (3,1,3)
         ],
        torch.tensor([[[1.0, 1.0, 1.0]], [[3.0, 3.0, 3.0]], [[5.0, 5.0, 5.0]]])
    )

def test_div():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.div(x1, x2)

    check_compute_output(
        y,
        [
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[2.5, 4.0, -0.1, 0.1], [-8.0, 5.0, -2.5, -1.0]]),
        ],
        torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
    )

def test_div_by_const():
    x1 = ad.Variable("x1")
    y = ad.div_by_const(x1, 5.0)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[-0.2, 0.4, 0.1, 0.68], [0.06, 0.0, -1.16, 0.62]]),
    )

def test_log():

    x1 = ad.Variable("x1")
    y = ad.log(x1)

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        # torch.tensor([[0.0000, 0.6931, -0.6931, 1.2238],[-1.2040, -torch.inf, 1.7579, 1.1314]])
        torch.nan_to_num(torch.tensor([[ 0.0000,  0.6931, -0.6931,  1.2238],[-1.2040, -torch.inf, torch.nan,  1.1314]]), nan=0.0)
    )

def test_matmul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)

    x1_val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x2_val = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    check_compute_output(
        y,
        [x1_val, x2_val],
        torch.tensor([[27.0, 30.0, 33.0], [61.0, 68.0, 75.0], [95.0, 106.0, 117.0]]),
    )

def test_matmul_3d():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)

    x1_val = torch.tensor([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]],
                          [[9.0, 8.0, 7.0],
                           [6.0, 5.0, 4.0],
                           [3.0, 2.0, 1.0]]])
    
    x2_val = torch.tensor([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]],
                          [[9.0, 8.0, 7.0],
                           [6.0, 5.0, 4.0],
                           [3.0, 2.0, 1.0]]])

    expected = torch.tensor([[[30.0, 36.0, 42.0],
                            [66.0, 81.0, 96.0],
                            [102.0, 126.0, 150.0]],
                           [[150.0, 126.0, 102.0],
                            [96.0, 81.0, 66.0],
                            [42.0, 36.0, 30.0]]])

    check_compute_output(
        y,
        [x1_val, x2_val],
        expected
    )

def test_layernorm():
    x = ad.Variable("x")
    y = ad.layernorm(x, normalized_shape=[3])

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)],
        torch.tensor([[-1.224745, 0.0, 1.224745], [-1.224745, 0.0, 1.224745]], dtype=torch.float32)
    )

def test_relu():
    x = ad.Variable("x")
    y = ad.relu(x)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0]], dtype=torch.float32)],
        torch.tensor([[0.0, 2.0, 0.0], [3.0, 0.0, 5.0]], dtype=torch.float32)
    )

def test_transpose():
    x = ad.Variable("x")
    y = ad.transpose(x, 1, 0)

    check_compute_output(
        y,
        [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])],
        torch.tensor([[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]])
    )

def test_softmax():
    x = ad.Variable("x")
    y = ad.softmax(x)

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)],
        torch.tensor([[0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]], dtype=torch.float32)
    )

def test_broadcast():
    x = ad.Variable("x")
    y = ad.broadcast(x, input_shape=[3, 2], target_shape=[2, 3, 2])

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])],
        torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        ])
    )

def test_sqrt():
    x = ad.Variable("x")
    y = ad.sqrt(x)

    check_compute_output(
        y,
        [torch.tensor([[4.0, 9.0], [16.0, 25.0]], dtype=torch.float32)],
        torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32)
    )

def test_power():
    x = ad.Variable("x")
    y = ad.power(x, 2)

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)],
        torch.tensor([[1.0, 4.0], [9.0, 16.0]], dtype=torch.float32)
    )

def test_mean():
    x = ad.Variable("x")
    y1 = ad.mean(x, dim=(0,),keepdim=True)
    y1_f = ad.mean(x, dim=(0,))
    y2 = ad.mean(x, dim=(1,),keepdim=True)
    y2_f = ad.mean(x, dim=(1,),keepdim=False)
    check_compute_output(
        y1,
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)],
        torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    )
    check_compute_output(
        y1_f,
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)],
        torch.tensor([2.0, 3.0], dtype=torch.float32)
    )
    check_compute_output(
        y2,
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)],
        torch.tensor([[1.5], [3.5]], dtype=torch.float32)
    )
    check_compute_output(
        y2_f,
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)],
        torch.tensor([1.5, 3.5], dtype=torch.float32)
    )

if __name__ == "__main__":
    test_mul()
    test_mul_by_const()
    test_div()
    test_div_by_const()
    test_layernorm()
    test_relu()
    test_softmax()
    test_matmul()
    test_matmul_3d()
    test_transpose()
    test_broadcast()
    test_power()
    test_sqrt()
