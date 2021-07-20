from torch.utils.data.dataset import Dataset
from .dataset_loaders import BaseDatasetPair, BaseImage

from typing import Callable, List, TypeVar, Generic, Any

TDatasetPair = TypeVar("TDatasetPair", bound=BaseDatasetPair, covariant=True)
TDatasetImage = TypeVar("TDatasetImage", bound=BaseImage, covariant=True)


class ImageDataset(Dataset, Generic[TDatasetImage]):
    def __init__(
        self,
        imageSet: List[TDatasetPair],
        transforms: Callable[[Any], List[Any]] = lambda sample: sample,
        cacheLimit: int = 0,
    ):

        self._imageSet = imageSet
        self._dataLength: int = len(self._imageSet)
        self._transforms = transforms

        self._totalCache: int = 0
        self._cacheLimit: int = cacheLimit

    def _HandleCache(self, image: Any, meta: TDatasetImage):

        if meta._cache is None:
            cacheSize = image.size / 1000000
            if self._cacheLimit > (self._totalCache + cacheSize):
                meta._cache = image
                self._totalCache += cacheSize

    def __getitem__(self, index: int):

        imagePair = self._imageSet[index]

        trainingMeta: TDatasetImage
        gtruthMeta: TDatasetImage
        trainingMeta, gtruthMeta = imagePair.GetPair()

        loadedTrain = trainingMeta.Load(useCache=True)
        loadedGtruth = gtruthMeta.Load(useCache=True)

        self._HandleCache(loadedGtruth, gtruthMeta)
        self._HandleCache(loadedTrain, trainingMeta)

        (
            transformedTraining,
            transformedGtruth,
            trainingMeta,
            gtruthMeta,
        ) = self._transforms([loadedTrain, loadedGtruth, trainingMeta, gtruthMeta])

        return [
            transformedTraining,
            transformedGtruth,
        ]

    def __len__(self):
        return self._dataLength
