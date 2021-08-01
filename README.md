# Setup

python3 -m venv venv
pip3 install --upgrade pip
pip3 install -r pip-requirements.txt

OR:

run setup.sh [./setup.sh]

## General Usage

Default dataset location is "dataset/"
Please download our dataset at: !TODO!

Weights are stored by default at "/local/"
Outputs are stored by default at "/output/"

### Train

python3 train.py

Arguments to look for:

TRAIN_INPUT_EXPOSURE: allowed train stage input exposure
TRAIN_TRUTH_EXPOSURE: allowed train stage ground truth exposure

TUNE_INPUT_EXPOSURE: allowed tune stage input exposure
TUNE_TRUTH_EXPOSURE: allowed tune stage ground truth exposure

Accepted values for these are : 0.1, 0.5, 1 , 5 , 10

Example: we want to train on inputs 0.1, 0.5 and 1 with ground truth of 10
TRAIN_INPUT_EXPOSURE = [0.1,0.5,1]
TRAIN_TRUTH_EXPOSURE = [10]

Training will be slow at first, 
if you have enough ram and set cache to be high, the entire dataset will eventually be loaded to RAM and the training process will accelerate.


### Inference

python3 test.py

Arguments to look for:

TUNE_FACTORS: list of tune factors to run tests with. 
Will run full tests on each tune factor and save data in "/output/" folder.

Rest of the arguments are the same as for training.

If you plan to run inference on the full images, chances you will have to switch to CPU. 
Make sure you have enough RAM for this too (apporx 35+ GB RAM).