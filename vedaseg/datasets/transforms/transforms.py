import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tt
from PIL import Image

from .registry import TRANSFORMS

CV2_MODE = {
    'bilinear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST,
    'cubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
}

CV2_BORDER_MODE = {
    'default': cv2.BORDER_DEFAULT,
    'constant': cv2.BORDER_CONSTANT,
    'reflect': cv2.BORDER_REFLECT,
    'reflect101': cv2.BORDER_REFLECT101,
    'replicate': cv2.BORDER_REPLICATE,
}


class Compose:
    def __init__(self, transforms, bitransforms=[]):
        self.transforms = transforms
        self.bitransforms = bitransforms

    def __call__(self, img1, mask1, img2=None, mask2=None):
        if len(self.bitransforms) > 0:
            for t in self.bitransforms:
                img1, mask1 = t(img1, mask1, img2, mask2)

        for t in self.transforms:
            img1, mask1 = t(img1, mask1)
        return img1, mask1


@TRANSFORMS.register_module
class FactorScale:
    def __init__(self, scale_factor=1.0, mode='bilinear'):
        self.mode = mode
        self.scale_factor = scale_factor

    def rescale(self, image, mask):
        h, w, c = image.shape

        if self.scale_factor == 1.0:
            return image, mask

        new_h = int(h * self.scale_factor)
        new_w = int(w * self.scale_factor)

        torch_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        torch_mask = torch.from_numpy(mask.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        torch_image = F.interpolate(torch_image, size=(new_h, new_w),
                                    mode=self.mode, align_corners=True)
        torch_mask = F.interpolate(torch_mask, size=(new_h, new_w),
                                   mode='nearest')

        new_image = torch_image.squeeze().permute(1, 2, 0).numpy()
        new_mask = torch_mask.squeeze().permute(1, 2, 0).numpy()

        return new_image, new_mask

    def __call__(self, image, mask):
        return self.rescale(image, mask)


@TRANSFORMS.register_module
class SizeScale(FactorScale):
    def __init__(self, target_size, mode='bilinear'):
        self.target_size = target_size
        super().__init__(mode=mode)

    def __call__(self, image, mask):
        h, w, _ = image.shape
        long_edge = max(h, w)
        self.scale_factor = self.target_size / long_edge

        return self.rescale(image, mask)


@TRANSFORMS.register_module
class RandomScale(FactorScale):
    def __init__(self, min_scale, max_scale, scale_step=0.0, mode='bilinear'):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step = scale_step
        super().__init__(mode=mode)

    @staticmethod
    def get_scale_factor(min_scale, max_scale, scale_step):
        if min_scale == max_scale:
            return min_scale

        if scale_step == 0:
            return random.uniform(min_scale, max_scale)

        num_steps = int((max_scale - min_scale) / scale_step + 1)
        scale_factors = np.linspace(min_scale, max_scale, num_steps)
        scale_factor = np.random.choice(scale_factors).item()

        return scale_factor

    def __call__(self, image, mask):
        self.scale_factor = self.get_scale_factor(self.min_scale, self.max_scale, self.scale_step)
        return self.rescale(image, mask)


@TRANSFORMS.register_module
class RandomCrop:
    def __init__(self, height, width, image_value, mask_value):
        self.height = height
        self.width = width
        self.image_value = image_value
        self.mask_value = mask_value
        self.channel = len(image_value)

    def __call__(self, image, mask):
        h, w, c = image.shape
        target_height = h + max(self.height - h, 0)
        target_width = w + max(self.width - w, 0)

        y1 = int(random.uniform(0, target_height - self.height + 1))
        y2 = y1 + self.height
        x1 = int(random.uniform(0, target_width - self.width + 1))
        x2 = x1 + self.width

        image_pad_value = np.reshape(np.array(self.image_value, dtype=image.dtype), [1, 1, self.channel])
        new_image = np.tile(image_pad_value, (target_height, target_width, 1))
        new_image[:h, :w, :] = image
        new_image = new_image[y1:y2, x1:x2, :]

        if mask is not None:
            mask_pad_value = np.reshape(np.array(np.tile(self.mask_value, mask.shape[2]), dtype=mask.dtype), [1, 1, mask.shape[2]])
            new_mask = np.tile(mask_pad_value, (target_height, target_width, 1))
            new_mask[:h, :w, :] = mask
            new_mask = new_mask[y1:y2, x1:x2, :]
        else:
            new_mask = None

        return new_image, new_mask


@TRANSFORMS.register_module
class PadIfNeeded:
    def __init__(self, image_value, mask_value, size=None, size_divisor=None, scale_bias=None):
        self.size = size
        self.size_divisor = size_divisor
        self.scale_bias = scale_bias

        self.image_value = image_value
        self.mask_value = mask_value
        self.channel = len(image_value)

        assert (self.size is None) ^ (self.size_divisor is None)

    def __call__(self, image, mask):
        h, w, c = image.shape

        if self.size:
            assert h <= self.size[0] and w <= self.size[1]
            target_height = h + max(self.size[0] - h, 0)
            target_width = w + max(self.size[1] - w, 0)

        elif self.size_divisor:
            target_height = int(np.ceil(h / self.size_divisor) * self.size_divisor) + self.scale_bias
            target_width = int(np.ceil(w / self.size_divisor) * self.size_divisor) + self.scale_bias

        image_pad_value = np.reshape(np.array(self.image_value, dtype=image.dtype), [1, 1, self.channel])
        new_image = np.tile(image_pad_value, (target_height, target_width, 1))
        new_image[:h, :w, :] = image

        if mask is not None:
            mask_pad_value = np.reshape(np.array(np.tile(self.mask_value, mask.shape[2]), dtype=mask.dtype), [1, 1, mask.shape[2]])
            new_mask = np.tile(mask_pad_value, (target_height, target_width, 1))
            new_mask[:h, :w, :] = mask
        else:
            new_mask = None

        # assert np.count_nonzero(mask != self.mask_value) == np.count_nonzero(new_mask != self.mask_value)

        return new_image, new_mask


@TRANSFORMS.register_module
class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        return image, mask


@TRANSFORMS.register_module
class VerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        return image, mask


@TRANSFORMS.register_module
class RandomRotate:
    def __init__(self, p=0.5, degrees=30, mode='bilinear', border_mode='reflect101', image_value=None, mask_value=None):
        self.p = p
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.mode = CV2_MODE[mode]
        self.border_mode = CV2_BORDER_MODE[border_mode]
        self.image_value = image_value
        self.mask_value = mask_value

    def __call__(self, image, mask):
        if random.random() < self.p:
            h, w, c = image.shape

            angle = random.uniform(*self.degrees)
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

            image = cv2.warpAffine(image, M=matrix, dsize=(w, h), flags=self.mode, borderMode=self.border_mode,
                                   borderValue=self.image_value)

            if mask is not None:
                mask = cv2.warpAffine(mask, M=matrix, dsize=(w, h), flags=cv2.INTER_NEAREST, borderMode=self.border_mode,
                                      borderValue=self.mask_value)

        return image, mask


@TRANSFORMS.register_module
class RandomErase:
    def __init__(self, max_size, min_size, image_value, mask_value, ignore=255, p=0.5):
        self.p = p
        self.min_h, self.min_w = min_size
        self.max_h, self.max_w = max_size
        self.image_value = image_value
        self.mask_value = mask_value
        self.ignore = ignore

    def __call__(self, image, mask):
        if random.random() < self.p:

            hs, ws, _ = np.where(mask != self.ignore)

            h, w = max(hs), max(ws)

            eh = int(random.uniform(self.min_h, self.max_h + 1))
            ew = int(random.uniform(self.min_w, self.max_w + 1))

            if 1 in mask:
                ihs, iws, _ = np.where(mask == 1)
                idx = np.random.choice(len(ihs))
                ih, iw = ihs[idx], iws[idx]
                y1 = int(random.uniform(max(0, ih - eh), min(h - eh + 1, ih)))
                y2 = eh + y1
                x1 = int(random.uniform(max(0, iw - ew), min(w - ew + 1, iw)))
                x2 = ew + x1

            else:
                y1 = int(random.uniform(0, h - eh + 1))
                y2 = eh + y1
                x1 = int(random.uniform(0, w - ew + 1))
                x2 = ew + x1

            image[y1:y2, x1:x2, :] = self.image_value
            mask[y1:y2, x1:x2, :] = self.mask_value

        return image, mask


@TRANSFORMS.register_module
class RandomAffine(object):
    def __init__(self, crange, image_value, mask_value,
                 p=0.5, mode='bilinear', border_mode='reflect101'):
        self.crange = (-crange, crange) \
            if isinstance(crange, (int, float)) else crange

        self.p = p
        self.mode = CV2_MODE[mode]
        self.border_mode = CV2_BORDER_MODE[border_mode]
        self.image_value = image_value
        self.mask_value = mask_value

    def _get_transform_matrix(self, img):
        jitters = np.random.uniform(*self.crange, 4).reshape(2, 2)
        transform_matrix = np.array([[1., 0., 0.],
                                    [0., 1., 0.]])
        transform_matrix[:2, :2] += jitters

        return transform_matrix

    def __call__(self, img, mask):
        if random.random() < self.p:
            matrix = self._get_transform_matrix(img)

            h, w, _ = img.shape
            img = cv2.warpAffine(
                img,
                M=matrix.astype(np.float64),
                dsize=(w, h),
                flags=self.mode,
                borderMode=self.border_mode,
                borderValue=self.image_value
            )
            mask = cv2.warpAffine(
                mask,
                M=matrix.astype(np.float64),
                dsize=(w, h),
                flags=self.mode,
                borderMode=self.border_mode,
                borderValue=self.mask_value
            )

        return img, mask


@TRANSFORMS.register_module
class GaussianBlur:
    def __init__(self, p=0.5, ksize=7):
        self.p = p
        self.ksize = (ksize, ksize) if isinstance(ksize, int) else ksize

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = cv2.GaussianBlur(image, ksize=self.ksize, sigmaX=0)

        return image, mask


@TRANSFORMS.register_module
class Normalize:
    def __init__(self, mean=(123.675, 116.280, 103.530), std=(58.395, 57.120, 57.375)):
        self.mean = mean
        self.std = std
        self.channel = len(mean)

    def __call__(self, image, mask):
        mean = np.reshape(np.array(self.mean, dtype=image.dtype), [1, 1, self.channel])
        std = np.reshape(np.array(self.std, dtype=image.dtype), [1, 1, self.channel])
        denominator = np.reciprocal(std, dtype=image.dtype)

        new_image = (image - mean) * denominator
        new_mask = mask

        return new_image, new_mask


@TRANSFORMS.register_module
class ColorJitter(tt.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness=brightness,
                         contrast=contrast,
                         saturation=saturation,
                         hue=hue)

    def __call__(self, image=None, mask=None):
        new_image = Image.fromarray(image.astype(np.uint8))
        new_image = super().__call__(new_image)
        new_image = np.array(new_image).astype(np.float32)
        return new_image, mask


@TRANSFORMS.register_module
class Blend:
    def __init__(self, image_value, mask_value, p=0.5, mixup_alpha=0.2, mixup_beta=0.4):
        self.p = p
        self.image_value = image_value
        self.mask_value = mask_value
        self.channel = len(image_value)
        self.mixup_alpha = mixup_alpha
        self.mixup_beta = mixup_beta

        # cv.addWeighted(src1, alpha, src2, beta, gamma

    def __call__(self, image1, mask1, image2, mask2):
        if random.random() < self.p:

            if 1 not in mask1:
                image1, image2 = image2, image1
                mask1, mask2 = mask2, mask1

            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]

            target_height = max(h1, h2)
            target_width = max(w1, w2)

            alpha = random.betavariate(self.mixup_alpha, self.mixup_beta)
            if alpha < 0.5:
                alpha = 1 - alpha

            image_pad_value1 = np.reshape(np.array(self.image_value, dtype=image1.dtype) * alpha, [1, 1, self.channel])
            image_pad_value2 = np.reshape(np.array(self.image_value, dtype=image1.dtype) * (1 - alpha), [1, 1, self.channel])
            mask_pad_value = np.reshape(np.array(np.tile(self.mask_value, mask1.shape[2]), dtype=mask1.dtype), [1, 1, mask1.shape[2]])

            new_image = np.tile(image_pad_value1, (target_height, target_width, 1))
            new_image_p = np.tile(image_pad_value2, (target_height, target_width, 1))
            new_mask = np.tile(mask_pad_value, (target_height, target_width, 1))

            y1s = int(random.uniform(0, target_height - h1 + 1))
            y1e = h1 + y1s
            x1s = int(random.uniform(0, target_width - w1 + 1))
            x1e = w1 + x1s

            y2s = int(random.uniform(0, target_height - h2 + 1))
            y2e = h2 + y2s
            x2s = int(random.uniform(0, target_width - w2 + 1))
            x2e = w2 + x2s

            new_image[y1s:y1e, x1s:x1e, :] = image1 * alpha
            new_image_p[y2s:y2e, x2s:x2e, :] = image2 * (1 - alpha)
            new_image += new_image_p

            new_mask[y1s:y1e, x1s:x1e, :] = mask1
            new_mask[y2s:y2e, x2s:x2e, :] |= mask2

            new_image = new_image[y1s:y1e, x1s:x1e, :]
            new_mask = new_mask[y1s:y1e, x1s:x1e, :]

        else:
            new_image, new_mask = image1, mask1

        return new_image, new_mask


@TRANSFORMS.register_module
class Concat:
    def __init__(self, image_value, mask_value, p=0.5):
        self.p = p
        self.image_value = image_value
        self.mask_value = mask_value
        self.channel = len(image_value)
        self.pool = [0, 1, 2, 3]

    def __call__(self, image1, mask1, image2, mask2):
        if random.random() < self.p:
            direct = np.random.choice(self.pool)

            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]

            if direct % 2:
                target_height = max(h1, h2)
                target_width = w1 + w2
            else:
                target_height = h1 + h2
                target_width = max(w1, w2)

            image_pad_value = np.reshape(np.array(self.image_value, dtype=image1.dtype), [1, 1, self.channel])
            mask_pad_value = np.reshape(np.array(np.tile(self.mask_value, mask1.shape[2]), dtype=mask1.dtype), [1, 1, mask1.shape[2]])

            new_image = np.tile(image_pad_value, (target_height, target_width, 1))
            new_mask = np.tile(mask_pad_value, (target_height, target_width, 1))

            y1 = int(random.uniform(0, target_height - h2 + 1))
            y2 = y1 + h2
            x1 = int(random.uniform(0, target_width - w2 + 1))
            x2 = x1 + w2

            if direct == 0:
                new_image[-h1:, :w1, :] = image1
                new_mask[-h1:, :w1, :] = mask1
                new_image[:h2, x1:x2, :] = image2
                new_mask[:h2, x1:x2, :] = mask2
            elif direct == 1:
                new_image[:h1, :w1, :] = image1
                new_mask[:h1, :w1, :] = mask1
                new_image[y1:y2, w1:, :] = image2
                new_mask[y1:y2, w1:, :] = mask2
            elif direct == 2:
                new_image[:h1, :w1, :] = image1
                new_mask[:h1, :w1, :] = mask1
                new_image[h1:, x1:x2, :] = image2
                new_mask[h1:, x1:x2, :] = mask2
            else:
                new_image[:h1, -w1:, :] = image1
                new_mask[:h1, -w1:, :] = mask1
                new_image[y1:y2, :w2, :] = image2
                new_mask[y1:y2, :w2, :] = mask2
            # assert np.count_nonzero(mask != self.mask_value) == np.count_nonzero(new_mask != self.mask_value)

            if direct % 2:
                new_image = new_image[:h1, :, :]
                new_mask = new_mask[:h1, :, :]
            else:
                new_image = new_image[:, :w1, :]
                new_mask = new_mask[:, :w1, :]
        else:
            new_image = image1
            new_mask = mask1

        return new_image, new_mask


@TRANSFORMS.register_module
class ToTensor:
    def __call__(self, image, mask):
        image = torch.from_numpy(image).permute(2, 0, 1)
        if mask is not None:
            mask = torch.from_numpy(mask).permute(2, 0, 1)

        return image, mask
