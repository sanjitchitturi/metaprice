import torch

from metaprice.meta_learning import compute_query_loss, inner_adapt
from metaprice.models import compose_parameters


def test_parameter_composition_uses_attention_weights_without_mutating_learners():
    models = [torch.nn.Linear(1, 1, bias=False) for _ in range(3)]
    with torch.no_grad():
        for value, model in zip((1.0, 3.0, 5.0), models, strict=True):
            model.weight.fill_(value)

    weights = torch.tensor([0.2, 0.3, 0.5], requires_grad=True)
    composed = compose_parameters(models, weights)

    torch.testing.assert_close(composed[0], torch.tensor([[3.6]]))
    torch.testing.assert_close(models[0].weight, torch.tensor([[1.0]]))
    torch.testing.assert_close(models[1].weight, torch.tensor([[3.0]]))
    torch.testing.assert_close(models[2].weight, torch.tensor([[5.0]]))


def test_query_loss_reaches_attention_through_composed_initialization():
    models = [torch.nn.Linear(1, 1, bias=False) for _ in range(3)]
    with torch.no_grad():
        for value, model in zip((0.5, 1.0, 1.5), models, strict=True):
            model.weight.fill_(value)

    logits = torch.tensor([0.0, 0.3, -0.2], requires_grad=True)
    weights = torch.softmax(logits, dim=0)
    initial = compose_parameters(models, weights)
    x = torch.tensor([[1.0]])
    target = torch.tensor([[0.0]])
    adapted = inner_adapt(models[0], x, target, inner_steps=1, inner_lr=0.1, initial_params=initial)
    loss = compute_query_loss(models[0], adapted, x, target)
    (gradient,) = torch.autograd.grad(loss, logits)

    assert torch.isfinite(gradient).all()
    assert torch.any(gradient.abs() > 0)
