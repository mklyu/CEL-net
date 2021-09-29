import lensfunpy as lf
from PIL import Image
from PIL.ExifTags import TAGS
import glob
import cv2
import exif

OUTPUT_BPS: int = 16
INPUT_DIR = "./output/"
OUTPUT_DIR: str = "./output/undistorted/"


class LFPUndistort:
    def __init__(self) -> None:

        db = lf.Database()

        self.cam = db.find_cameras("Sony", "ILCE-5100")[0]
        self.lens = db.find_lenses(self.cam, "Sony", "E PZ 16-50mm f/3.5-5.6 OSS")[0]

    def ConvertImage(
        self,
        image,
        focalLength: float,
        fNumber: float,
        focusDistance: float,
    ):
        height, width = image.shape[0], image.shape[1]

        modifier = lf.Modifier(self.lens, self.cam.crop_factor, width, height)
        modifier.initialize(focalLength, fNumber, focusDistance)

        undistCoords = modifier.apply_geometry_distortion()
        imageUndistorted = cv2.remap(image, undistCoords, None, cv2.INTER_LANCZOS4)

        return imageUndistorted

def GetImageName(imagePath):
    return imagePath.split("/")[-1][0:-3]


if __name__ == "__main__":

    print("Undistorting images inside " + INPUT_DIR)
    print("Output path set to " + OUTPUT_DIR)

    undistort = LFPUndistort()
    inputImages = glob.glob(INPUT_DIR + "*.jpg")

    totalImages = inputImages.__len__()

    index = 0
    for imagePath in inputImages:

        index += 1
        print(index.__str__() + " of " + totalImages.__str__())

        image = cv2.imread(imagePath, -1)
        imageExif = exif.Image(imagePath)

        try:
            focalLength = imageExif.focal_length
            fNumber = imageExif.f_number
            userComment = imageExif.user_comment
        except AttributeError:
            raise Exception("Failed to read EXIF from " + imagePath)

        focusDistance = 10
        if userComment== "indoor":
            focusDistance = 4
        if userComment == "outdoor":
            focusDistance = 20

        imUndistorted = undistort.ConvertImage(
            image, imageExif.focal_length, imageExif.f_number, focusDistance
        )

        undistortedImagePath = OUTPUT_DIR + GetImageName(imagePath) + ".jpg"
        cv2.imwrite(undistortedImagePath, imUndistorted)

    print("Done")

        
