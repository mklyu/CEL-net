import lensfunpy as lf
from PIL import Image
import glob
import cv2

OUTPUT_BPS: int = 16
INPUT_DIR = "./output/"
OUTPUT_DIR: str = "./output/undistorted/"


class LFPUndistort:
    def __init__(self) -> None:

        db = lf.Database()

        self.cam = db.find_cameras("Sony", "ILCE-5100")[0]
        self.lens = db.find_lenses(self.cam, "Sony", "E PZ 16-50mm f/3.5-5.6 OSS")[0]

    def ConvertImage(
        self, image, focalLength: float, fNumber: float, focusDistance: float,
    ):
        height, width = image.shape[0], image.shape[1]

        modifier = lf.Modifier(self.lens, self.cam.crop_factor, width, height)
        modifier.initialize(focalLength, fNumber, focusDistance)

        undistCoords = modifier.apply_geometry_distortion()
        imageUndistorted = cv2.remap(image, undistCoords, None, cv2.INTER_LANCZOS4)

        return imageUndistorted


def GetMetadata(imagename):

    image = Image.open(imagename)
    # extract EXIF data
    metadata = {}
    exifData = image.getexif()

    for tagId in exifData:

        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tagId, tagId)
        data = exifData.get(tagId)

        # decode bytes
        if isinstance(data, bytes):
            try:
                data = data.decode()
            except:
                pass

        metadata[tag] = float(data)

    return metadata

def GetImageName(self, imagePath):
    return imagePath.split("/")[-1][0:-3]

if __name__ == "__main__":

    inputImages = glob.glob(INPUT_DIR + "*.jpg")

    index = 0
    for imagePath in inputImages:
        index += 1


        metadata = GetMetadata(JPG_DIR + GetImageName(imagePath) + "JPG")

        im = cv2.imread("tmp_image.tiff", -1)
        height, width = im.shape[0], im.shape[1]

        mod = lf.Modifier(lens, cam.crop_factor, width, height)
        mod.initialize(metadata["FocalLength"], metadata["FNumber"], FOCUS_DISTANCE)

        undistCoords = mod.apply_geometry_distortion()
        imUndistorted = cv2.remap(im, undistCoords, None, cv2.INTER_LANCZOS4)
        cv2.imwrite(undistortedImagePath, imUndistorted)
        print(undistortedImagePath)
