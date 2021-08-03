import torch
import torch.utils.data

from .cel import (
    CELPair,
    DatasetFilterCallbackType,
    nopFilter,
    CELDatasetLoader,
)
from image_dataset.image_dataset import ImageDataset

from typing import Callable, List

INDOOR_EVAL_SCENARIO_START = 130
INDOOR_TEST_SCENARIO_START = 145

OUTDOOR_EVAL_SCENARIO_START = 105
OUTDOOR_TEST_SCENARIO_START = 120

INDOOR_KEYWORD = "indor"
OUTDOOR_KEYWORD = "outdoor"


class CELDataloaderFactory:
    def __init__(
        self,
        inputFolder: str,
        patchSize: int = 512,
        datasetWorkers: int = 0,
        batch: int = 1,
        cacheLimit: int = 0,
    ):

        self._dir = inputFolder
        self._patchSize = patchSize
        self._datasetWorkers = datasetWorkers
        self._batch = batch
        self._cacheLimit = cacheLimit

    def _FilterScenariosMax(self, maxScenario: int, images: List[CELPair]):
        newList: List[CELPair] = []
        for image in images:
            if image.truthList[0].scenario <= maxScenario:
                newList.append(image)

        return newList

    def _FilterScenariosMin(self, maxScenario: int, images: List[CELPair]):
        newList: List[CELPair] = []
        for image in images:
            if image.truthList[0].scenario > maxScenario:
                newList.append(image)

        return newList

    def _FilterTrain(self, set: List[CELPair]):
        pairsToRemove = []

        for pair in set:
            if pair.trainList[0].location == INDOOR_KEYWORD:
                if pair.trainList[0].scenario >= INDOOR_EVAL_SCENARIO_START:
                    pairsToRemove.append(pair)

            else:
                if pair.trainList[0].scenario >= OUTDOOR_EVAL_SCENARIO_START:
                    pairsToRemove.append(pair)

        for pair in pairsToRemove:
            set.remove(pair)

        return set

    def _FilterEval(self, set: List[CELPair]):
        pairsToRemove = []

        for pair in set:
            if pair.trainList[0].location == INDOOR_KEYWORD:
                if (
                    pair.trainList[0].scenario >= INDOOR_TEST_SCENARIO_START
                    or pair.trainList[0].scenario < INDOOR_EVAL_SCENARIO_START
                ):
                    pairsToRemove.append(pair)

            else:
                if (
                    pair.trainList[0].scenario >= OUTDOOR_TEST_SCENARIO_START
                    or pair.trainList[0].scenario < OUTDOOR_EVAL_SCENARIO_START
                ):
                    pairsToRemove.append(pair)

        for pair in pairsToRemove:
            set.remove(pair)

        return set

    def _FilterTest(self, set: List[CELPair]):
        pairsToRemove = []

        for pair in set:
            if pair.trainList[0].location == INDOOR_KEYWORD:
                if pair.trainList[0].scenario < INDOOR_TEST_SCENARIO_START:
                    pairsToRemove.append(pair)

            else:
                if pair.trainList[0].scenario < OUTDOOR_TEST_SCENARIO_START:
                    pairsToRemove.append(pair)

        for pair in pairsToRemove:
            set.remove(pair)

        return set

    def GetTrain(
        self,
        transforms: Callable,
        inputFilter: DatasetFilterCallbackType = nopFilter,
        truthFilter: DatasetFilterCallbackType = nopFilter,
    ):
        datasetLoader = CELDatasetLoader(self._dir, inputFilter, truthFilter)
        trainSet = datasetLoader.GetSet()

        trainSet = self._FilterTrain(trainSet)
        trainDataset = ImageDataset(trainSet, transforms, cacheLimit=self._cacheLimit)

        trainDatasetLoader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=self._batch,
            shuffle=True,
            num_workers=self._datasetWorkers,
            collate_fn=ImageDataset.collate_fn
        )
        return trainDatasetLoader

    def GetEval(
        self,
        transforms: Callable,
        inputFilter: DatasetFilterCallbackType = nopFilter,
        truthFilter: DatasetFilterCallbackType = nopFilter,
    ):
        datasetLoader = CELDatasetLoader(self._dir, inputFilter, truthFilter)
        evalSet = datasetLoader.GetSet()

        evalSet = self._FilterEval(evalSet)
        testDataset = ImageDataset(evalSet, transforms, cacheLimit=self._cacheLimit)

        testDatasetLoader = torch.utils.data.DataLoader(
            testDataset, batch_size=1, shuffle=False, num_workers=self._datasetWorkers,
            collate_fn=ImageDataset.collate_fn
        )
        return testDatasetLoader

    def GetTest(
        self,
        transforms: Callable,
        inputFilter: DatasetFilterCallbackType = nopFilter,
        truthFilter: DatasetFilterCallbackType = nopFilter,
    ):
        datasetLoader = CELDatasetLoader(self._dir, inputFilter, truthFilter)
        testSet = datasetLoader.GetSet()

        testSet = self._FilterTest(testSet)
        testDataset = ImageDataset(testSet, transforms, cacheLimit=self._cacheLimit)

        testDatasetLoader = torch.utils.data.DataLoader(
            testDataset, batch_size=1, shuffle=False, num_workers=self._datasetWorkers,
            collate_fn=ImageDataset.collate_fn
        )
        return testDatasetLoader
