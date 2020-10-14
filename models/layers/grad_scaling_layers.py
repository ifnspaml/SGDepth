import unittest

import torch
import torch.nn as nn
import torch.autograd as autograd


class ScaleGrad(autograd.Function):
    @staticmethod
    def forward(ctx, scale, *inputs):
        ctx.scale = scale

        outputs = inputs[0] if (len(inputs) == 1) else inputs

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_inputs = tuple(
            ctx.scale * grad
            for grad in grad_outputs
        )

        return (None, *grad_inputs)


class ScaledSplit(nn.Module):
    """Identity maps an input into outputs and scale gradients in the backward pass

    Args:
        *grad_weights: one or multiple weights to apply to the gradients in
            the backward pass

    Examples:

        >>> # Multiplex to two outputs, the gradients are scaled
        >>> # by 0.3 and 0.7 respectively
        >>> scp = ScaledSplit(0.3, 0.7)
        >>> # The input may consist of multiple tensors
        >>> inp
        (tensor(...), tensor(...))
        >>> otp1, otp2 = scp(inp)
        >>> # Considering the forward pass both outputs behave just like inp.
        >>> # In the backward pass the gradients will be scaled by the respective
        >>> # weights
        >>> otp1
        (tensor(...), tensor(...))
        >>> otp2
        (tensor(...), tensor(...))
    """

    def __init__(self, *grad_weights):
        super().__init__()
        self.set_scales(*grad_weights)

    def set_scales(self, *grad_weights):
        self.grad_weights = grad_weights

    def get_scales(self, *grad_weights):
        return self.grad_weights

    def forward(self, *inputs):
        # Generate nested tuples, where the outer layer
        # corresponds to the output & grad_weight pairs
        # and the inner layer corresponds to the list of inputs
        split = tuple(
            tuple(ScaleGrad.apply(gw, inp) for inp in inputs)
            for gw in self.grad_weights
        )

        # Users that passed only one input don't expect
        # a nested tuple as output but rather a tuple of tensors,
        # so unpack if there was only one input
        unnest_inputs = tuple(
            s[0] if (len(s) == 1) else s
            for s in split
        )

        # Users that specified only one output weight
        # do not expect a tuple of tensors put rather
        # a single tensor, so unpack if there was only one weight
        unnest_outputs = unnest_inputs[0] if (len(unnest_inputs) == 1) else unnest_inputs

        return unnest_outputs


class GRL(ScaledSplit):
    """Identity maps an input and invert the gradient in the backward pass

    This layer can be used in adversarial training to train an encoder
    encoder network to become _worse_ a a specific task.
    """

    def __init__(self):
        super().__init__(-1)


class TestScaledSplit(unittest.TestCase):
    def test_siso(self):
        factor = 0.5

        scp = ScaledSplit(factor)

        # Construct a toy network with inputs and weights
        inp = torch.tensor([ 1, 1, 1], dtype=torch.float32, requires_grad=True)
        wgt = torch.tensor([-1, 0, 1], dtype=torch.float32, requires_grad=False)

        pre_split = inp * wgt
        post_split = scp.forward(pre_split)

        self.assertTrue(torch.equal(pre_split, post_split), 'ScaledSplit produced non-identity in forward pass')

        # The network's output is a single number
        sum_pre = pre_split.sum()
        sum_post = post_split.sum()

        # Compute the gradients with and withou scaling
        grad_pre, = autograd.grad(sum_pre, inp, retain_graph=True)
        grad_post, = autograd.grad(sum_post, inp)

        # Check if the scaling matches expectations
        self.assertTrue(torch.equal(grad_pre * factor, grad_post), 'ScaledSplit produced inconsistent gradient')

    def test_simo(self):
        # TODO

        pass

    def test_miso(self):
        # TODO

        pass


    def test_mimo(self):
        # TODO

        pass


if __name__ == '__main__':
    # Use python3 -m grad_scaling_layers
    # to run the unit tests and check if you
    # broke something

    unittest.main()
