import lensfunpy as lf
import cv2
import glob
import rawpy
from PIL import Image
from PIL.ExifTags import TAGS
import imageio

OUTPUT_DIR = "outdoor/png_fixed/"
INPUT_DIR = "outdoor/ARW/"

JPG_DIR = "outdoor/JPG/"

DONE_IMAGES = glob.glob(OUTPUT_DIR + "*.png")
FOCAL_LENGTH = 16.0
APERTURE = 9
FOCUS_DISTANCE = 4

OUTPUT_BPS : int = 16


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

        if tag in ["FocalLength", "FNumber", "ISOSpeedRatings"]:
            metadata[tag] = float(data)

    return metadata


db = lf.Database()
cam = db.find_cameras("Sony", "ILCE-5100")[0]
lens = db.find_lenses(cam, "Sony", "E PZ 16-50mm f/3.5-5.6 OSS")[0]


def GetImageName(img_path):
    return img_path.split("/")[-1][0:-3]


inputImages = glob.glob(INPUT_DIR + "*.ARW")

index = 0
for imagePath in inputImages:
    index += 1

    rawImage = rawpy.imread(imagePath)
    undistortedImagePath = OUTPUT_DIR + GetImageName(imagePath) + "png"

    if undistortedImagePath in DONE_IMAGES:
        continue

    postprocessedImage = rawImage.postprocess(
        no_auto_bright=True, output_bps=OUTPUT_BPS, four_color_rgb=True, use_camera_wb=True
    )

    imageio.imsave("tmp_image.tiff", postprocessedImage)
    print(imagePath, "number: ", index, " out of: ", inputImages.__len__())

    metadata = GetMetadata(JPG_DIR + GetImageName(imagePath) + "JPG")

    im = cv2.imread("tmp_image.tiff", -1)
    height, width = im.shape[0], im.shape[1]

    mod = lf.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(metadata["FocalLength"], metadata["FNumber"], FOCUS_DISTANCE)

    undistCoords = mod.apply_geometry_distortion()
    imUndistorted = cv2.remap(im, undistCoords, None, cv2.INTER_LANCZOS4)
    cv2.imwrite(undistortedImagePath, imUndistorted)
    print(undistortedImagePath)
