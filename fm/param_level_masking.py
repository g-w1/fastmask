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
