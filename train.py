import logging
import sys
import os
from typing import Union, Tuple, List

import torch
import torch.nn
import torch.utils.data
from torch import optim
import functools
from torchvision import transforms

import common
from model_wrapper import ModelWrapper
from networks import CELNet
from image_dataset.dataset_loaders.CEL import CELDataloaderFactory, cel_filters

# --- General Settings ---

IMAGE_BPS: int = 16
# can be a tuple, make sure that values are divisible by 16
PATCH_SIZE: Union[Tuple[int], int] = 512

DEVICE: str = "cuda:0"

# fiddle with these if training seems oddly slow
DATASET_WORKER_COUNT: int = 2
BATCH_COUNT = 2

# Maximum RAM allowed to be used in megabytes. Approx 80-60 gigabytes is optimal
IMAGE_CACHE_SIZE_MAX = 10000

DATASET_DIRECTORY: str = "./dataset/"
WEIGHTS_DIRECTORY: str = "./local/model.pt"

# --- Dataset Filtering ---

# train inputs
TRAIN_INPUT_EXPOSURE: List[float] = [0.1]
TRAIN_TRUTH_EXPOSURE: List[float] = [1]

# tune inputs
TUNE_INPUT_EXPOSURE: List[float] = [0.1]
TUNE_TRUTH_EXPOSURE: List[float] = [10]


def Run():

    # construct image transformations

    exposureNormTransform = common.NormByExposureTime(IMAGE_BPS)

    trainTransforms = transforms.Compose(
        [
            common.GetTrainTransforms(
                IMAGE_BPS, PATCH_SIZE, normalize=False, device=DEVICE
            ),
            exposureNormTransform,
        ]
    )

    # construct filters to sort database

    trainInputFilter = functools.partial(cel_filters.FilterExactInList, TRAIN_INPUT_EXPOSURE)
    trainTruthFilter = functools.partial(cel_filters.FilterExactInList, TRAIN_TRUTH_EXPOSURE)

    tuneInputFilter = functools.partial(cel_filters.FilterExactInList, TUNE_INPUT_EXPOSURE)
    tuneTruthFilter = functools.partial(cel_filters.FilterExactInList, TUNE_TRUTH_EXPOSURE)

    dataloaderFactory = CELDataloaderFactory(
        DATASET_DIRECTORY, batch=BATCH_COUNT, cacheLimit=IMAGE_CACHE_SIZE_MAX,
    )

    network = CELNet(adaptive=False)
    optimiser = optim.Adam(network.parameters(), lr=1e-4)
    wrapper = ModelWrapper(network, optimiser, torch.nn.L1Loss(), DEVICE)

    if not os.path.exists(WEIGHTS_DIRECTORY):
        network._initialize_weights()
        wrapper.metaDict["model_tune_state"] = False
        wrapper.Save(WEIGHTS_DIRECTORY)

    wrapper.LoadWeights(WEIGHTS_DIRECTORY, strictWeightLoad=True)
    isModelInTuneState = wrapper.metaDict["model_tune_state"]

    wrapper.OnTrainEpoch += lambda *args: wrapper.Save(WEIGHTS_DIRECTORY)

    if not isModelInTuneState:

        trainDataloader = dataloaderFactory.GetTrain(
            trainTransforms, trainInputFilter, trainTruthFilter
        )
        wrapper.Train(trainDataloader, trainToEpoch=1000, learningRate=1e-4)
        wrapper.Train(trainDataloader, trainToEpoch=2000, learningRate=1e-5)
        wrapper.Train(trainDataloader, trainToEpoch=3000, learningRate=1e-6)

        # free up memory
        del trainDataloader

        wrapper.metaDict["model_tune_state"] = True
        wrapper.Save(WEIGHTS_DIRECTORY)

    # tuning starts here, rebuild everything
    tuneDataloader = dataloaderFactory.GetTrain(
        trainTransforms, tuneInputFilter, tuneTruthFilter
    )

    network = CELNet(adaptive=True)
    optimParams = network.TuningMode()

    optimiser = optim.Adam(optimParams, lr=1e-4)
    wrapper = ModelWrapper(network, optimiser, torch.nn.L1Loss(), DEVICE)
    wrapper.LoadWeights(WEIGHTS_DIRECTORY, loadOptimiser=False, strictWeightLoad=True)

    wrapper.OnTrainEpoch += lambda *args: wrapper.Save(WEIGHTS_DIRECTORY)

    wrapper.Train(tuneDataloader, trainToEpoch=3500, learningRate=1e-4)
    wrapper.Train(tuneDataloader, trainToEpoch=4000, learningRate=1e-5)
    wrapper.Train(tuneDataloader, trainToEpoch=4500, learningRate=1e-6)


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    Run()
