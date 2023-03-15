import typing as tp
import torch
from torch.nn import CTCLoss


def my_ctc_loss(
    output: torch.Tensor,
    target: tp.List[torch.Tensor],
) -> torch.Tensor:
    input_lengths = [output.size(0)]
    input_lengths = torch.LongTensor(input_lengths)
    target_lengths = target[1]
    target = torch.LongTensor(target[0])
    ctc_loss = CTCLoss()
    return ctc_loss(output, target, input_lengths, target_lengths)


def my_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
) -> int:
    output = output.numpy().astype(int)
    target = target.numpy().astype(int)
    output = "".join(map(str, output))
    target = "".join(map(str, target))
    if output == target:
        return 1
    return 0
