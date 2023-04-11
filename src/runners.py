from typing import Any, Mapping
import torch
from catalyst.runners import SupervisedRunner
from src.tools import get_code


class SupervisedOCRRunner(SupervisedRunner):
    def __init__(
        self,
        input_key: str = "image",
        output_key: str = "output",
        target_key: str = "target",
        **kwargs,
    ):
        super().__init__(
            input_key=input_key,
            output_key=output_key,
            target_key=target_key,
            **kwargs,
        )

    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        output = super().forward(batch, **kwargs)
        output["output_size"] = torch.IntTensor([output["output"].size(0)] * self.batch_size)  # noqa: WPS221
        output["pred_codes"] = get_code(output["output"])
        return output
