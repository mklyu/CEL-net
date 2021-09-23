import lensfunpy as lf
import cv2

OUTPUT_BPS: int = 16


class LFPUndistort:
    def __init__(self) -> None:

        db = lf.Database()

        self.cam = db.find_cameras("Sony", "ILCE-5100")[0]
        self.lens = db.find_lenses(self.cam, "Sony", "E PZ 16-50mm f/3.5-5.6 OSS")[0]

    def GetImageName(self, imagePath):
        return imagePath.split("/")[-1][0:-3]

    def ConvertImage(
        self, image, focalLength: float, fNumber: float, focusDistance: float,
    ):
        height, width = image.shape[0], image.shape[1]

        modifier = lf.Modifier(self.lens, self.cam.crop_factor, width, height)
        modifier.initialize(focalLength, fNumber, focusDistance)

        undistCoords = modifier.apply_geometry_distortion()
        imageUndistorted = cv2.remap(image, undistCoords, None, cv2.INTER_LANCZOS4)

        return imageUndistorted
