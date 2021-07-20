import torch
from skimage import metrics

from typing import List, TypeVar, Generic, Union


T = TypeVar("T")


class Metric(Generic[T]):
    def __init__(self, name: Union[str, None] = None):
        self.data: List[T] = []
        self.name = name

    def Call(self, var: T):
        self.data.append(var)

    def Flush(self):
        self.data = []


class PSNR(Metric[float]):
    def __init__(self, name: Union[str, None] = None, dataRange: int = None):
        super().__init__(name)
        self._dataRange = dataRange

    def Call(
        self, image1: torch.Tensor, image2: torch.Tensor,
    ):
        PSNR: float = metrics.peak_signal_noise_ratio(
            image1, image2, data_range=self._dataRange
        )

        self.data.append(PSNR)


class SSIM(Metric[float]):
    def __init__(
        self, multichannel=False, name: Union[str, None] = None, dataRange: int = None
    ):
        super().__init__(name)
        self._multichannel = multichannel
        self._dataRange = dataRange

    def Call(
        self, image1: torch.Tensor, image2: torch.Tensor,
    ):
        SSIM = metrics.structural_similarity(
            image1, image2, multichannel=self._multichannel, data_range=self._dataRange
        )

        self.data.append(SSIM)
