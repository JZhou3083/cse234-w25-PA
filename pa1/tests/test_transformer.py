from typing import Dict, List

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import auto_diff as ad
import transformer2 as tf

def check_evaluator_output(
    evaluator: ad.Evaluator,
    input_values: Dict[ad.Node, torch.Tensor],
    expected_outputs: List[torch.Tensor],
) -> None:
    output_values = evaluator.run(input_values)
    assert len(output_values) == len(expected_outputs)
    for output_val, expected_val in zip(output_values, expected_outputs):
        torch.testing.assert_close(output_val, expected_val, atol=1e-4, rtol=1e-4)

def test_linear():
    x = ad.Variable("x")
    w = ad.Variable("w")
    b = ad.Variable("b")
    y = tf.Linear(x, w, b)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={
            x: torch.tensor([[[1.0,2.0,3.0,4.0], [2,3,4,5]], [[5.0,6.0,7.0,8.0], [6,7,8,9]]]),
            w: torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]),
            b: torch.tensor([0.1, 0.2]),
        },
        expected_outputs=[torch.tensor([[[ 5.1000,  6.2000],
         [ 6.7000,  8.2000]],
        [[11.5000, 14.2000],
         [13.1000, 16.2000]]])],
    )