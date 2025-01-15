# %%
import einops
import torch
from torch.amp import custom_fwd, custom_bwd
import functools
from torch.autograd import Function


# https://github.com/pytorch/pytorch/issues/132388#issuecomment-2365344299
def _custom_setup_context(
    setup_context_fn=None,
    *,
    device_type: str,
):
    if setup_context_fn is None:
        return functools.partial(_custom_setup_context, device_type=device_type)

    @functools.wraps(setup_context_fn)
    def decorate_setup_context(ctx, *args, **kwargs):
        ctx._dtype = torch.get_autocast_dtype(device_type)
        ctx._fwd_used_autocast = torch.is_autocast_enabled(device_type)
        return setup_context_fn(ctx, *args, **kwargs)

    return decorate_setup_context


class RoutedLinear(Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(input, weight, bias, batch_learning_rates, is_param_level):
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        input, weight, bias, batch_learning_rates, is_param_level = inputs
        ctx.save_for_backward(input, weight, bias, batch_learning_rates)
        ctx.is_param_level = is_param_level

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        input, weight, bias, batch_learning_rates = ctx.saved_tensors
        is_param_level = ctx.is_param_level
        grad_input = grad_weight = grad_bias = None

        learning_rates_expanded = batch_learning_rates.unsqueeze(-1).expand(
            grad_output.shape
        )
        if not is_param_level:
            # multiply the gradient on the output by the per-batch-element learning rates
            # this affects both the gradient on the weight node and the gradient on the input
            grad_output = grad_output * learning_rates_expanded
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if is_param_level:
            grad_output = grad_output * learning_rates_expanded
        if ctx.needs_input_grad[1]:
            grad_weight = einops.einsum(grad_output, input, "... j, ... i -> j i")
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.flatten(0, -2).sum(0)

        return grad_input, grad_weight, grad_bias, None, None


routed_linear = RoutedLinear.apply

# TESTS
# TODO test with pytest instead
if __name__ == "__main__":

    def get_grads(input, weight, bias, batch_learning_rates, is_param_level):
        input.requires_grad = True
        weight.requires_grad = True
        bias.requires_grad = True
        if input.grad is not None:
            input.grad.zero_()
        if weight.grad is not None:
            weight.grad.zero_()
        if bias.grad is not None:
            bias.grad.zero_()
        output = routed_linear(
            input, weight, bias, batch_learning_rates, is_param_level
        )
        output.sum().backward()

    input = torch.tensor(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
        ],
        requires_grad=True,
        dtype=torch.float,
    )
    weight = torch.tensor([[1, 2]], requires_grad=True, dtype=torch.float)
    bias = torch.tensor([0], requires_grad=True, dtype=torch.float)
    batch_learning_rates = torch.tensor([1, 2, 1, 0, 1], requires_grad=False)

    get_grads(input, weight, bias, batch_learning_rates, is_param_level=False)
    expected_input_grad = torch.tensor(
        [[1, 2], [2, 4], [1, 2], [0, 0], [1, 2]], dtype=torch.float
    )
    assert torch.allclose(input.grad, expected_input_grad)
    expected_weight_grad = torch.tensor(
        [[1 + 2 * 3 + 5 + 0 * 7 + 9, 2 + 2 * 4 + 6 + 0 * 8 + 10]], dtype=torch.float
    )
    assert torch.allclose(weight.grad, expected_weight_grad)
    expected_bias_grad = torch.tensor([1 + 2 + 1 + 0 + 1], dtype=torch.float)
    assert torch.allclose(bias.grad, expected_bias_grad)

    get_grads(input, weight, bias, batch_learning_rates, is_param_level=True)
    # if paramater level masking is True, it should not affect gradeints on the input
    expected_input_grad = torch.tensor(
        [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]], dtype=torch.float
    )
    assert torch.allclose(input.grad, expected_input_grad)
    assert torch.allclose(weight.grad, expected_weight_grad)
    assert torch.allclose(bias.grad, expected_bias_grad)

    # now do a test on a slightly more complicated setting where we have 2 batch dimensions like in Transformers

    # input shape is (2, 5, 3) for (batch, seq, d_model)
    input = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
            [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]],
        ],
        requires_grad=True,
        dtype=torch.float,
    )
    # weight shape is (1, 3) for (d_expanded, d_model)
    weight = torch.tensor([[10, 11, 12]], requires_grad=True, dtype=torch.float)
    # bias shape is (1,) for (d_expanded)
    bias = torch.tensor([2], requires_grad=True, dtype=torch.float)

    # batch_learning_rates shape is (2, 5) for (batch, seq)
    batch_learning_rates = torch.tensor(
        [
            [1, 2, 1, 0, 1],
            [1, 1, -1, 1, 0],
        ],
        requires_grad=False,
    )

    get_grads(input, weight, bias, batch_learning_rates, is_param_level=False)
    expected_input_grad = torch.tensor(
        [
            [[10, 11, 12], [20, 22, 24], [10, 11, 12], [0, 0, 0], [10, 11, 12]],
            [[10, 11, 12], [10, 11, 12], [-10, -11, -12], [10, 11, 12], [0, 0, 0]],
        ],
        dtype=torch.float,
    )
    assert torch.allclose(input.grad, expected_input_grad)
    expected_weight_grad = torch.tensor(
        [
            [
                (1 + 2 * 4 + 7 + 0 * 10 + 13) + (16 + 19 + 22 * -1 + 25 + 28 * 0),
                (2 + 2 * 5 + 8 + 0 * 11 + 14) + (17 + 20 + 23 * -1 + 26 + 29 * 0),
                (3 + 2 * 6 + 9 + 0 * 12 + 15) + (18 + 21 + 24 * -1 + 27 + 30 * 0),
            ]
        ],
        dtype=torch.float,
    )
    assert torch.allclose(weight.grad, expected_weight_grad)
    expected_bias_grad = torch.tensor([batch_learning_rates.sum()], dtype=torch.float)
    assert torch.allclose(bias.grad, expected_bias_grad)

    get_grads(input, weight, bias, batch_learning_rates, is_param_level=True)
    expected_input_grad = torch.tensor(
        [
            [[10, 11, 12], [10, 11, 12], [10, 11, 12], [10, 11, 12], [10, 11, 12]],
            [[10, 11, 12], [10, 11, 12], [10, 11, 12], [10, 11, 12], [10, 11, 12]],
        ],
        dtype=torch.float,
    )
    assert torch.allclose(input.grad, expected_input_grad)
    assert torch.allclose(weight.grad, expected_weight_grad)
    assert torch.allclose(bias.grad, expected_bias_grad)

    input = torch.randn(2, 5, 3, requires_grad=True)
    weight = torch.randn(4, 3, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)
    batch_learning_rates = torch.ones(2, 5, requires_grad=False)

    output = torch.nn.functional.linear(input, weight, bias)
    output.sum().backward()
    expected_input_grad = input.grad.clone()
    expected_weight_grad = weight.grad.clone()
    expected_bias_grad = bias.grad.clone()

    get_grads(input, weight, bias, batch_learning_rates, False)
    assert torch.allclose(input.grad, expected_input_grad)
    assert torch.allclose(weight.grad, expected_weight_grad)
    assert torch.allclose(bias.grad, expected_bias_grad)
    get_grads(input, weight, bias, batch_learning_rates, True)
    assert torch.allclose(input.grad, expected_input_grad)
    assert torch.allclose(weight.grad, expected_weight_grad)
    assert torch.allclose(bias.grad, expected_bias_grad)
    # all learning rates are zero, but param level masking is True,
    # so gradients on input should be unchanged but gradients on weight and bias should be zero
    get_grads(input, weight, bias, torch.zeros_like(batch_learning_rates), True)
    assert torch.allclose(input.grad, expected_input_grad)
    assert torch.allclose(weight.grad, torch.zeros_like(weight.grad))
    assert torch.allclose(bias.grad, torch.zeros_like(bias.grad))
    # now everything should be zero since masking the activation
    get_grads(input, weight, bias, torch.zeros_like(batch_learning_rates), False)
    assert torch.allclose(input.grad, torch.zeros_like(input.grad))
    assert torch.allclose(weight.grad, torch.zeros_like(weight.grad))
    assert torch.allclose(bias.grad, torch.zeros_like(bias.grad))

# %%
