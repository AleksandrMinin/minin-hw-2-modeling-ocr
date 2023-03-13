"""Модуль содержит в себе реализацию CRNN модели."""
import timm
import torch


class CRNN(torch.nn.Module):
    """Реализует CRNN модель для OCR задачи.
    
    CNN-backbone берется из timm, в RNN части стоит GRU.
    """

    def __init__(
            self, 
            cnn_backbone_name: str,
            cnn_backbone_pretrained: bool,
            cnn_output_size: int,
            rnn_features_num: int,
            rnn_dropout: float,
            rnn_bidirectional: bool,
            rnn_num_layers: int,
            num_classes: int,
            
    ) -> None:
        """
        Parameters: (лучше вынести в config, в этом примере перечислены параметры для простоты)
        
            cnn_backbone_name: имя backbone-а 
            cnn_backbone_pretrained: True, чтобы брать предобученный бэкбон, иначе False
            cnn_output_size: размер выходного тензора из CNN (можно сделать прогон в init-е и посчитать)
            rnn_features_num: размер выхода каждого GRU слоя
            rnn_dropout: если не ноль, добавляет dropout после каждого GRU слоя с указанным значением
            rnn_bidirectional: True, чтобы использовать двунаправленную GRU, иначе False
            rnn_num_layers: количество слоев GRU,
            num_classes: Количество классов - длина алфавита + 1.
        """
        super().__init__()

        # Предобученный бекбон для фичей. Можно обрезать, не обязательно использовать всю глубину. 
        self.backbone = timm.create_model(
            cnn_backbone_name, pretrained=cnn_backbone_pretrained,
        )
        self.backbone.global_pool = torch.nn.Identity()
        self.backbone.fc = torch.nn.Identity()

        # Боттлнек. Можно обойтись и без него если rnn_features_num == cnn_output_size.
        self.gate = torch.nn.Linear(cnn_output_size, rnn_features_num)

        # Рекуррентная часть.
        self.rnn = torch.nn.GRU(
            rnn_features_num,
            rnn_features_num,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
            num_layers=rnn_num_layers,
        )

        classifier_in_features = rnn_features_num
        if rnn_bidirectional:
            classifier_in_features = 2 * rnn_features_num

        # Классификатор.
        self.fc = torch.nn.Linear(classifier_in_features, num_classes)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:    # noqa: WPS210
        """
        Прямой проход по графу вычислений.

        Parameters:
            tensor: Изображение-тензор.

        Returns:
            Сырые логиты для каждой позиции в карте признаков.
        """
        cnn_features = self.backbone(tensor)
        batch_size, channels, height, width = cnn_features.shape
        cnn_features = cnn_features.view(
            batch_size, height * channels, width,
        ).permute(2, 0, 1)
        cnn_features = self.gate(cnn_features)
        rnn_output, _ = self.rnn(cnn_features)
        logits = self.fc(rnn_output)
        output = self.softmax(logits)
        return output
