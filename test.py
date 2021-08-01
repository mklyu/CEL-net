import os
import logging
import sys
from typing import Union, Tuple, List

import torch
from torchvision import transforms
import torch.nn
import torch.utils.data
from torch import optim
import functools
import numpy as np
import imageio

from model_wrapper import ModelWrapper
from image_dataset.dataset_loaders.CEL import CELDataloaderFactory, cel_filters
import metric_handlers
from networks import AdaUNet4X3
import common

# --- General Settings ---

IMAGE_BPS: int = 16
# can be a tuple, make sure that values are divisible by 16
# PATCH_SIZE = (2000,3008) # the maximum for our dataset. You'll probably need to use CPU for this, and around 60 GB of RAM
PATCH_SIZE: Union[Tuple[int], int] = 512
# if GPU has insufficient memory (will result in crashes), switch DEVICE to "cpu"
#  make sure you have enoguh RAM available if you do, especially if you cache images
DEVICE: str = "cuda:0"

# Maximum memory allowed to be used in megabytes. Approx 80-60 gigabytes is ideal
# If you are running just one test, set this to 0.
IMAGE_CACHE_SIZE_MAX = 0

OUTPUT_DIRECTORY: str = "./output/"
DATASET_DIRECTORY: str = "./dataset/"
WEIGHTS_DIRECTORY: str = "./local/model.pt"

TUNE_FACTORS = [0.5]

# Save image rate
SAVE_IMAGE_RATE = 10

# --- Dataset Filtering ---
TEST_INPUT_EXPOSURE: List[float]  = [0.1]
TEST_TRUTH_EXPOSURE: List[float]  = [5]


def GetTestCallbacks(
    wrapper: ModelWrapper,
    PSNR: metric_handlers.PSNR,
    SSIM: metric_handlers.SSIM,
    imageNumberMetric: metric_handlers.Metric[int],
):
    def SaveMetrics(
        inputImage: torch.Tensor,
        gTruthImage: torch.Tensor,
        unetOutput: torch.Tensor,
        loss: float,
    ):

        imageNumber = imageNumberMetric.data.__len__()
        imageNumberMetric.Call(imageNumber)

        gtruthProcessed = gTruthImage[0].permute(1, 2, 0).cpu().data.numpy()
        unetOutputProcessed = unetOutput[0].permute(1, 2, 0).cpu().data.numpy()

        unetOutputProcessed = np.minimum(np.maximum(unetOutputProcessed, 0), 1)

        PSNR.Call(gtruthProcessed, unetOutputProcessed)
        SSIM.Call(gtruthProcessed, unetOutputProcessed)

    return SaveMetrics


def GetSaveImagesCallback(
    wrapper: ModelWrapper, directory: str, rate: int, prefix: str,
):

    imageIndex = [0]

    def Callback(
        inputImage: torch.Tensor,
        gTruthImage: torch.Tensor,
        unetOutput: torch.Tensor,
        loss: float,
    ):
        if (imageIndex[0] % rate) == 0:
            imname = prefix + "_" + imageIndex[0].__str__()
            imdir = directory + "/" + imname + ".jpg"

            convertedImage = unetOutput[0].permute(1, 2, 0).cpu().data.numpy()

            convertedImage = np.minimum(np.maximum(convertedImage, 0), 1)

            convertedImage *= 255
            convertedImage = convertedImage.astype(np.uint8)
            imageio.imwrite(imdir, convertedImage, "jpg")

        imageIndex[0] += 1

    return Callback


def Run():

    exposureNormTransform = common.NormByExposureTime(IMAGE_BPS)

    testTransforms = transforms.Compose(
        [
            common.GetEvalTransforms(
                IMAGE_BPS, PATCH_SIZE, normalize=False, device=DEVICE
            ),
            exposureNormTransform,
        ]
    )

    testInputFilter = functools.partial(cel_filters.FilterExactInList, TEST_INPUT_EXPOSURE)
    testTruthFilter = functools.partial(cel_filters.FilterExactInList, TEST_TRUTH_EXPOSURE)

    dataloaderFactory = CELDataloaderFactory(
        DATASET_DIRECTORY, batch=1, cacheLimit=IMAGE_CACHE_SIZE_MAX,
    )

    testDataloader = dataloaderFactory.GetTest(
        testTransforms, testInputFilter, testTruthFilter
    )

    network = AdaUNet4X3(adaptive=True)
    optimiser = optim.Adam(network.parameters(), lr=1e-4)
    wrapper = ModelWrapper(network, optimiser, torch.nn.L1Loss(), DEVICE)

    metricPSNR = metric_handlers.PSNR(name="PSNR")
    metricSSIM = metric_handlers.SSIM(multichannel=True, name="SSIM")
    tuneFactorMetric = metric_handlers.Metric[float](name="Tune factor")
    imageNumberMetric = metric_handlers.Metric[int](name="Image")

    csvFileDir: str = OUTPUT_DIRECTORY + "data.csv"

    metricsToCsv = metric_handlers.MetricsToCsv(
        csvFileDir, [imageNumberMetric, tuneFactorMetric, metricPSNR, metricSSIM]
    )

    wrapper.OnTestIter += GetTestCallbacks(
        wrapper, metricPSNR, metricSSIM, imageNumberMetric
    )

    if not os.path.exists(WEIGHTS_DIRECTORY):
        raise IOError("File " + WEIGHTS_DIRECTORY + " not found")
    else:
        wrapper.LoadWeights(
            WEIGHTS_DIRECTORY, strictWeightLoad=True, loadOptimiser=False
        )

    for factor in TUNE_FACTORS:

        tuneFactorLambda = lambda *args: tuneFactorMetric.Call(factor)
        saveImageCallback = GetSaveImagesCallback(
            wrapper, OUTPUT_DIRECTORY, SAVE_IMAGE_RATE, factor.__str__()
        )

        wrapper.OnTestIter += tuneFactorLambda
        wrapper.OnTestIter += saveImageCallback

        network.InterpolateAndLoadWeights(factor)
        wrapper.Test(testDataloader)

        wrapper.OnTestIter -= tuneFactorLambda

    metricsToCsv.Write()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    Run()