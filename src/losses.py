import typing as tp
import torch
from torch.nn import CTCLoss


def MyCTCLoss(
    output: torch.Tensor, 
    target: tp.List[torch.Tensor]
) -> torch.Tensor:
    input_lengths = [output.size(0)]
    input_lengths = torch.LongTensor(input_lengths)
    target_lengths = target[1]
    criterion = CTCLoss()
    target = torch.LongTensor(target[0])
    loss = criterion(output, target, input_lengths, target_lengths)
    return loss


def MyAccuracy(
    output: torch.Tensor, 
    target: torch.Tensor
) -> int:
    output = output.numpy().astype(int)
    target = target.numpy().astype(int)
    output = ''.join(map(str, output))
    target = ''.join(map(str, target))
    if output == target:
        return 1
    else:
        return 0
