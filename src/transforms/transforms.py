import torch
from torchvision import transforms


class SegTransform:
    def __init__(self):
        colorJitter = transforms.ColorJitter(
            brightness=0.6, contrast=0.6, saturation=0.6, hue=0.5
        )
        rotater = transforms.RandomRotation(degrees=(0, 180))
        inverter = transforms.RandomInvert(p=0.5)

        self.input_transform = transforms.Compose([colorJitter, inverter])
        self.transform = transforms.Compose([rotater])

    def apply_transform(self, img, true_masks):
        if not isinstance(img, torch.Tensor) or not isinstance(
            true_masks, torch.Tensor
        ):
            raise TypeError("Both img and true_masks should be torch.Tensor")

        img = self.input_transform(img)
        image_and_true_masks = torch.cat((img, true_masks), dim=1)
        image_and_true_masks = self.transform(image_and_true_masks)

        aug_image = image_and_true_masks[:, 0, :, :]
        true_masks = image_and_true_masks[:, 1:, :, :].contiguous()
        aug_image = torch.unsqueeze(aug_image, 1).contiguous()
        return aug_image.contiguous(), true_masks


class standart_image_transfoms:
    def __init__(self):
        ColorJitter = transforms.ColorJitter(
            brightness=0.6, contrast=0.6, saturation=0.6, hue=0.5
        )
        rotater = transforms.RandomRotation(degrees=(0, 180))
        inverter = transforms.RandomInvert(p=0.5)
        transform = transforms.Compose([ColorJitter, inverter, rotater])
        self.transform = transform

    def transformation(self, images):
        images = self.transform(images)
        return images


class standart_RGB_image_transfoms:
    def __init__(self):
        ColorJitter = transforms.ColorJitter(
            brightness=0.6, contrast=0.6, saturation=0.6, hue=0.5
        )
        rotater = transforms.RandomRotation(degrees=(0, 180))
        transform = transforms.Compose([ColorJitter, rotater])
        self.transform = transform

    def transformation(self, images):
        images = self.transform(images)
        return images


class fur_transfoms:
    def __init__(self):
        ColorJitter = transforms.ColorJitter(
            brightness=0.6, contrast=0.6, saturation=0.6, hue=0.5
        )
        rotater = transforms.RandomRotation(degrees=(0, 180))
        inverter = transforms.RandomInvert(p=0.5)
        input_transform = transforms.Compose([ColorJitter, inverter])
        transform = transforms.Compose([rotater])
        self.input_transform = input_transform
        self.transform = transform

        self.fur = torch.fft.fft2
        self.real = torch.real
        self.imag = torch.imag

    def transformation(self, images, true_masks):
        images = self.input_transform(images)
        image_and_true_masks = torch.cat((images, true_masks), dim=1)
        image_and_true_masks = self.transform(image_and_true_masks)
        aug_image = image_and_true_masks[:, 0, :, :]
        true_masks = image_and_true_masks[:, 1:, :, :].contiguous()
        aug_image = torch.unsqueeze(aug_image, 1).contiguous()
        fur_imag = torch.zeros(
            aug_image.size()[0], 3, aug_image.size()[2], aug_image.size()[3]
        )
        fur_imag[:, 2, :, :] = aug_image[:, 0, :, :]
        aug_image = self.fur(aug_image)
        fur_imag[:, 1, :, :] = self.real(aug_image[:, 0, :, :])
        fur_imag[:, 0, :, :] = self.imag(aug_image[:, 0, :, :])

        for b in range(fur_imag.size()[0]):
            for chan in [0, 1]:
                fur_imag[b, chan] = torch.log(
                    fur_imag[b, chan] - fur_imag[b, chan].min() + 1
                )
                max_val = fur_imag[b, chan].max()
                if max_val != 0:
                    fur_imag[b, chan] = fur_imag[b, chan] / max_val

        return fur_imag.contiguous(), true_masks
