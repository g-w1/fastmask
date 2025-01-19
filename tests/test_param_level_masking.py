from fm.param_level_masking import RoutedLinear
import torch

routed_linear = RoutedLinear.apply


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
    output = routed_linear(input, weight, bias, batch_learning_rates, is_param_level)
    output.sum().backward()


def test_basic_param_and_activation_level():
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


def test_multi_batch_dims():
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


def test_matches_regular_torch_behavior_when_lrs_are_one():
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
