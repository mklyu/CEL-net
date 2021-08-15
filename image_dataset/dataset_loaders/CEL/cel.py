import json
import rawpy
import glob
import os
import numpy as np

from image_dataset.dataset_loaders import (
    BaseDatasetLoader,
    BaseDatasetPair,
    BaseImage,
)

from typing import Dict, List, Callable, Union


INDOOR_FOLDER = "indoor"
OUTDOOR_FOLDER = "outdoor"

RAW_FOLDER = "ARW"
RAW_FORMAT = "ARW"

INDOOR_META_FILE = "indoor_meta.json"
OUTDOOR_META_FILE = "outdoor_meta.json"


IMAGE_BPS = 16


class CELImage(BaseImage):
    def __init__(self, imagePath: str) -> None:

        basename = os.path.basename(imagePath)
        formatSplit = basename.rsplit(".", 1)
        format = formatSplit[1]

        BaseImage.__init__(self, imagePath, format)

        dataSplit = formatSplit[0].split("_")

        self.scenario = int(dataSplit[1])

        self.location = dataSplit[0]
        self.exposure = float(dataSplit[3])


    @classmethod
    def FromArray(cls, array: List[str]):
        imageArray: List[CELImage] = []
        for image in array:
            imageArray.append(cls(image))

        return imageArray


class CELTruthImage(CELImage):
    def __init__(
        self,
        imagePath: str,
    ) -> None:
        super().__init__(imagePath)

    def LoadHook(self):
        rawImage = rawpy.imread(self.path)
        postProcImage = rawImage.postprocess(
            no_auto_bright=True,
            output_bps=IMAGE_BPS,
            four_color_rgb=True,
            use_camera_wb=True,
        )

        return postProcImage


class CELPair(BaseDatasetPair):
    def __init__(
        self,
        trainingImage: Union[CELImage, List[CELImage]],
        truthImage: Union[CELImage, List[CELImage]],
    ) -> None:

        self.trainList: List[CELImage] = self._LoadListOrImage(trainingImage)
        self.truthList: List[CELImage] = self._LoadListOrImage(truthImage)

    def GetPair(self):
        train = self._GetRandomImage(self.trainList)
        truth = self._GetRandomImage(self.truthList)

        return [train, truth]

    def _LoadListOrImage(self, listOrImage: Union[CELImage, List[CELImage]]):
        if isinstance(listOrImage, list):
            return listOrImage
        else:
            return [listOrImage]

    def _GetRandomImage(self, arr):
        randint = np.random.randint(0, arr.__len__())
        return arr[randint]


DatasetFilterCallbackType = Callable[[List[CELImage]], List[CELImage]]
nopFilter: Callable[[List[CELImage]], List[CELImage]] = lambda images: images


class CELDatasetLoader(BaseDatasetLoader):
    def __init__(
        self,
        path: str,
        trainFilter: DatasetFilterCallbackType = nopFilter,
        truthFilter: DatasetFilterCallbackType = nopFilter,
    ) -> None:
        self._dir = path
        self._trainFilter = trainFilter
        self._truthFilter = truthFilter

    def _GrabImages(self, path: str, imageFormat: str):
        return glob.glob(path + "*." + imageFormat)

    def _GenerateScenarioDict(self, imageList: List[CELImage]):
        newDict: Dict[int, List[CELImage]] = {}

        for image in imageList:
            if image.scenario in newDict.keys():
                newDict[image.scenario].append(image)
            else:
                newDict[image.scenario] = [image]

        return newDict

    def _GeneratePairs(
        self,
        trainList: List[CELImage],
        truthList: List[CELImage],
        meta: Dict,
    ):

        truthDict: Dict[int, List[CELImage]] = {}
        trainDict: Dict[int, List[CELImage]] = {}
        pairs: List[CELPair] = []

        truthDict = self._GenerateScenarioDict(truthList)
        trainDict = self._GenerateScenarioDict(trainList)

        trainKeysToRemove = []
        for scenario in trainDict.keys():
            if scenario not in truthDict.keys():
                trainKeysToRemove.append(scenario)

        for key in trainKeysToRemove:
            del trainDict[key]

        for trainKey in trainDict:
            trainInd = trainDict[trainKey]
            truthInd = truthDict[trainKey]

            for index, image in enumerate(truthInd):
                image = CELTruthImage(
                    image.path,
                    meta[image.scenario.__str__()][image.exposure.__str__()],
                )
                truthInd[index] = image

            newPair = CELPair(trainInd, truthInd)
            pairs.append(newPair)

        return pairs

    def GetSet(self):

        indoorPath = self._dir + INDOOR_FOLDER
        outdoorPath = self._dir + OUTDOOR_FOLDER

        indoorTrain = self._GrabImages(indoorPath + "/" + RAW_FOLDER + "/", RAW_FORMAT)
        outdoorTrain = self._GrabImages(
            outdoorPath + "/" + RAW_FOLDER + "/", RAW_FORMAT
        )

        indoorTruth = self._GrabImages(indoorPath + "/" + RAW_FOLDER + "/", RAW_FORMAT)
        outdoorTruth = self._GrabImages(
            outdoorPath + "/" + RAW_FOLDER + "/", RAW_FORMAT
        )

        with open(self._dir + "/" + INDOOR_META_FILE, "r") as file:
            indoorMeta = json.load(file)
        with open(self._dir + "/" + OUTDOOR_META_FILE, "r") as file:
            outdoorMeta = json.load(file)

        outdoorTrain = CELImage.FromArray(outdoorTrain)
        outdoorTruth = CELImage.FromArray(outdoorTruth)

        indoorTrain = CELImage.FromArray(indoorTrain)
        indoorTruth = CELImage.FromArray(indoorTruth)

        outdoorTrain = self._trainFilter(outdoorTrain)
        outdoorTruth = self._truthFilter(outdoorTruth)

        indoorTrain = self._trainFilter(indoorTrain)
        indoorTruth = self._truthFilter(indoorTruth)

        outdoorPairs = self._GeneratePairs(outdoorTrain, outdoorTruth, outdoorMeta)
        indoorPairs = self._GeneratePairs(indoorTrain, indoorTruth, indoorMeta)

        return outdoorPairs + indoorPairs
