import argparse

FROM_SCRATCH = True

SAVE_PRED_EVERY = 1000
NUM_STEPS = 20000

EPOCH = 100
NUM_STEPS_STOP = 20  # early stopping

DIR_NAME = 'SkipAE'
INPUT_SIZE = '512, 256'

DATA_DIRECTORY_TARGET = '../dataset/CityScapes'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'

DATA_LIST_PATH_TARGET_VAL = './dataset/cityscapes_list/val.txt'


LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
POWER = 0.9


RANDOM_SEED = 1338

IGNORE_LABEL = 0

BATCH_SIZE = 4
NUM_WORKERS = 0

NUM_CLASSES = 13

EVAL_TARGET = -1


SNAPSHOT_DIR = './snapshots/'


class BaseOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description="CUDA square framework")
        self.parser = parser

    def parse(self):
        # get the basic options
        opt = self.parser.parse_args()
        self.opt = opt

        return self.opt


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        # Training options
        self.parser.add_argument("--epoch", type=str, default=EPOCH)

        self.parser.add_argument("--eval-target", type=int, default=EVAL_TARGET)
        self.parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                            help="Number of images sent to the network in one step.")
        self.parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                            help="number of workers for multithread dataloading.")

        self.parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                            help="The index of the label to ignore during the training.")
        self.parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                            help="Comma-separated string with height and width of source images.")

        self.parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                            help="Base learning rate for training with polynomial decay.")

        self.parser.add_argument("--momentum", type=float, default=MOMENTUM)
        self.parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
        self.parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                            help="Number of training steps.")
        self.parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                            help="Number of training steps for early stopping.")
        self.parser.add_argument("--power", type=float, default=POWER,
                            help="Decay parameter to compute the learning rate.")
        self.parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                            help="Random seed to have reproducible results.")
        self.parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                            help="Save summaries and checkpoint every often.")
        self.parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                            help="Where to save snapshots of the model.")
        self.parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                            help="Regularisation parameter for L2-loss.")

        self.parser.add_argument("--from-scratch", action='store_true', default=FROM_SCRATCH,
                                 help="Whether to choose training from scratch or not")
        self.parser.add_argument("--dir-name", type=str, default=DIR_NAME,
                                 help="snapshot directory")

        self.parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                            help="Path to the directory containing the target dataset.")
        self.parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                            help="Path to the file listing the images in the target dataset.")

        self.parser.add_argument("--data-list-target-val", type=str, default=DATA_LIST_PATH_TARGET_VAL,
                            help="Path to the file listing the images in the target dataset.")