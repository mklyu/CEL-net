# CEL-Net
### Continuous Exposure for Extreme Low-Light Imaging | [Paper](https://arxiv.org/pdf/2012.04112.pdf)
Evgeny Hershkovitch Neiterman∗, Michael Klyuchka∗, Gil Ben-Artzi


<sub>Equall contribution is marked by an asterisk (*) </sub>

# Setup

The code uses Python 3, with Pip as the recommended package manager.


Tested on Unbuntu 18.04 LTS.


An NVIDIA GPU is recommended for accelerated training and inference.

Pytroch 1.9.0

Torchvision 0.10.0



For your convenience, a frozen package snapshot of our workspace is available as "pip_freeze.txt".


Run:

> python3 -m venv venv

> source ./venv/bin/activate

> pip3 install --upgrade pip

> pip3 install -r pip-requirements.txt

OR:

> run setup.sh [./setup.sh]


The default dataset location is in the "dataset" folder

Please download our dataset [here](https://arielacil-my.sharepoint.com/:f:/g/personal/neiterman_ariel_ac_il/EruoYnkCjzREtC1hbPRN3Y0BQahIPkyMVfy-dp2AqiioGQ?e=i222JI).
Download the "test" and "train" folders and move them to the "dataset" folder.
The JPGs are the camera's ISP output, they are not needed.
If any errors occur during download or unzipping, try downloading subfolders manually and combining them into the original structure.

Weights are stored by default in the "local" folder
Outputs are stored by default in the "output" folder

# Train

Run:

> source ./venv/bin/activate

> python3 train.py

Arguments to look for inside "train.py":

TRAIN_INPUT_EXPOSURE: allowed train stage input exposure

TRAIN_TRUTH_EXPOSURE: allowed train stage ground truth exposure

TUNE_INPUT_EXPOSURE: allowed tune stage input exposure

TUNE_TRUTH_EXPOSURE: allowed tune stage ground truth exposure

Accepted values for these are: 0.1, 0.5, 1, 5, 10

Example: we want to train on inputs 0.1, 0.5, and 1 with a ground truth of 10

TRAIN_INPUT_EXPOSURE = [0.1,0.5,1]

TRAIN_TRUTH_EXPOSURE = [10]

If you have enough RAM and set the cache to be high, the entire dataset will eventually be loaded to RAM and the training process will accelerate (should take about a day, depending on your machine).

Training directly from the disk (i.e. 0 RAM allocated) might take a week or two, depending on your disk speed.


# Inference

Run:

> source ./venv/bin/activate

> python3 test.py

Arguments to look for:

TUNE_FACTORS: list of tune factors to run tests with. 

Will run full tests on each tune factor and save data in the "output" folder.

The rest of the arguments are the same as for training.

If you plan to run inference on the full images, chances are you will have to switch to CPU.

Make sure you have enough RAM for this too (approx 35+ GB RAM).

# Undistorting Outputs

The original dataset photos have a fisheye effect.
We provided an undistortion code that will fix all the images inside the "output" folder.

The default output of this code is the "output/undistorted" folder.

Simply run:

> python3 undistort.py

Please note that this code was written with full-sized images in mind. And will produce distortions with smaller patch sizes.



# Possible problems

3080ti cards might not work on standard PyTorch pip packages (as of September 2021).

This might help:

> pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
