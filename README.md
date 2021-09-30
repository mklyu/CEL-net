# Setup

Run:

> python3 -m venv venv

> source ./venv/bin/activate

> pip3 install --upgrade pip

> pip3 install -r pip-requirements.txt

OR:

> run setup.sh [./setup.sh]


Default dataset location is at the [dataset/] folder
Please download our dataset at: !TODO!

Weights are stored by default at the [local/] folder
Outputs are stored by default at the [output/] folder

# Train

Run:

> source ./venv/bin/activate

> python3 train.py

Arguments to look for inside [train.py]:

TRAIN_INPUT_EXPOSURE: allowed train stage input exposure

TRAIN_TRUTH_EXPOSURE: allowed train stage ground truth exposure

TUNE_INPUT_EXPOSURE: allowed tune stage input exposure

TUNE_TRUTH_EXPOSURE: allowed tune stage ground truth exposure

Accepted values for these are: 0.1, 0.5, 1, 5, 10

Example: we want to train on inputs 0.1, 0.5 and 1 with ground truth of 10

TRAIN_INPUT_EXPOSURE = [0.1,0.5,1]

TRAIN_TRUTH_EXPOSURE = [10]

If you have enough ram and set cache to be high, the entire dataset will eventually be loaded to RAM and the training process will accelerate (should take about a day, depends on your machine).

Training directly from the disc (i.e. 0 RAM allocated) might take a week or so.


# Inference

Run:

> source ./venv/bin/activate

> python3 test.py

Arguments to look for:

TUNE_FACTORS: list of tune factors to run tests with. 

Will run full tests on each tune factor and save data in [output/] folder.

The rest of the arguments are the same as for training.

If you plan to run inference on the full images, chances you will have to switch to CPU. 
Make sure you have enough RAM for this too (apporx 35+ GB RAM).

# Undistorting outputs

The original dataset photos have a fisheye effect.
We provided an undistortion code that will fix all the images inside the [output/] folder.

The default output of this code is the [output/undistorted/] folder.

Simply run:

> python3 undistort.py



# Possible problems

3080ti cards might not work on standard pytorch packages (as of september 2021).

This might help:

> pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html