"""Differentiable MAML adaptation without mutating shared model parameters."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import functional_call


def predict_with_params(model, params, inputs):
    """Evaluate with explicit parameters, preserving the autograd graph."""
    names = [name for name, _ in model.named_parameters()]
    if len(names) != len(params):
        raise ValueError("Parameter count does not match the model.")
    return functional_call(model, dict(zip(names, params, strict=True)), (inputs,), strict=True)


def inner_adapt(
    model,
    support_x,
    support_y,
    inner_steps=5,
    inner_lr=0.01,
    initial_params=None,
):
    """Return differentiable adapted tensors without mutating ``model``.

    ``initial_params`` supports MetaPrice's attention-weighted parameter
    initialization. When omitted, adaptation begins from the model parameters,
    which keeps the standard MAML baseline unchanged.
    """
    if inner_steps < 0 or inner_lr <= 0:
        raise ValueError("Adaptation requires nonnegative steps and a positive learning rate.")
    if support_x.numel() == 0 or support_y.numel() == 0:
        raise ValueError("The support set must be nonempty.")
    params = list(model.parameters()) if initial_params is None else list(initial_params)
    if len(params) != len(tuple(model.parameters())):
        raise ValueError("Initial parameter count does not match the model.")
    for _ in range(inner_steps):
        loss = F.mse_loss(predict_with_params(model, params, support_x), support_y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        params = [p - inner_lr * grad for p, grad in zip(params, grads, strict=True)]
    return params


def compute_query_loss(model, params, query_x, query_y):
    """Compute query MSE through the adaptation graph."""
    if query_x.numel() == 0:
        raise ValueError("The query set must be nonempty.")
    return F.mse_loss(predict_with_params(model, params, query_x), query_y)


def meta_train_step(model, tasks, inner_steps, inner_lr, device="cpu"):
    """Average independent task losses linked to the shared initialization."""
    if not tasks:
        raise ValueError("A meta-training batch must contain at least one task.")
    losses = []
    for task in tasks:
        sx, sy = task.support_tensors(device)
        qx, qy = task.query_tensors(device)
        adapted = inner_adapt(model, sx, sy, inner_steps, inner_lr)
        losses.append(compute_query_loss(model, adapted, qx, qy))
    return torch.stack(losses).mean()


def adapt_and_predict(model, task, inner_steps=5, inner_lr=0.01, device="cpu"):
    sx, sy = task.support_tensors(device)
    qx, _ = task.query_tensors(device)
    adapted = inner_adapt(model, sx, sy, inner_steps, inner_lr)
    with torch.no_grad():
        predictions = predict_with_params(model, adapted, qx)
    return predictions, adapted
