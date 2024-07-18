import torch


def my_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    acc = 0
    for pred, target in zip(preds, targets):
        pred = "".join([str(int(x)) for x in pred])
        target = "".join([str(int(x)) for x in target])
        if pred == target:
            acc += 1
    return acc / len(targets)
