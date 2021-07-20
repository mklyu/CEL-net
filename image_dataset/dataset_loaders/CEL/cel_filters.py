from image_dataset.dataset_loaders.CEL import CELImage

from typing import List


def FilterExact(exposure: float, images: List[CELImage]):
    newList: List[CELImage] = []

    for image in images:
        if image.exposure == exposure:
            newList.append(image)

    return newList


def FilterExactInList(exposures: List[float], images: List[CELImage]):
    newList: List[CELImage] = []

    for image in images:
        if image.exposure in exposures:
            newList.append(image)

    return newList


def FilterNot(exposure: float, images: List[CELImage]):
    newList: List[CELImage] = []

    for image in images:
        if not image.exposure == exposure:
            newList.append(image)

    return newList


def FilterBetween(minExposure: float, maxExposure: float, images: List[CELImage]):
    newList: List[CELImage] = []

    for image in images:
        if image.exposure <= maxExposure and image.exposure >= minExposure:
            newList.append(image)

    return newList

