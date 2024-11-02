import segmentation_models_pytorch as smp


class ModelFactory:
    """
    Другие задачи и модели будут сделаны позже
    """

    @staticmethod
    def create_model(config):
        task = config.get("task", "segmentation")
        model_name = config.get("model", "Linknet")  # По умолчанию Linknet
        encoder_name = config.get("encoder_name", "efficientnet-b7")
        encoder_weights = config.get("encoder_weights", "imagenet")
        in_channels = config.get("in_channels", 1)
        num_classes = config.get("num_classes", 1)

        if task == "segmentation":
            ModelClass = getattr(smp, model_name, None)
            if ModelClass is None:
                raise ValueError(f"Модель '{model_name}' не поддерживается")

            return ModelClass(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
            )
        else:
            raise ValueError(f"Задача '{task}' не поддерживается")
