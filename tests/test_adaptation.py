import torch

from metaprice.meta_learning import compute_query_loss, inner_adapt


def test_query_gradient_reaches_initial_weight_without_mutating_it():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    x = torch.tensor([[1.0]])
    target = torch.zeros_like(x)
    adapted = inner_adapt(model, x, target, inner_steps=1, inner_lr=0.25)
    loss = compute_query_loss(model, adapted, x, target)
    (gradient,) = torch.autograd.grad(loss, model.weight)
    torch.testing.assert_close(model.weight, x)
    torch.testing.assert_close(gradient, x * 0.5)
