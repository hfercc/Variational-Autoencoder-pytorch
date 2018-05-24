from __future__ import print_function

import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data

from data_loaders.XRAB3_data_loader import XRAB3_DataLoader
from graph.mse_loss import Loss as Loss_mse
from graph.flexarch_mse_model import VAE as VAE_flex_mse
from train.mse_trainer import Trainer as Trainer_mse
from utils.utils import *
from utils.weight_initializer import Initializer

# New functionality I implemented
from modules import MultiLayerConv

def main():
    # Parse the JSON arguments
    args = parse_args()

    # Create the experiment directories
    args.summary_dir, args.checkpoint_dir = create_experiment_dirs(
        args.experiment_dir)

    if args.loss == 'ce':
        model = VAE_ce()
    else:
        ##############################################
        # This is the start of the code that I wrote #
        # to modify the original project             #
        ##############################################
        input_size = (1,512,512)
        encoder, decoder, encoded_dims = MultiLayerConv(1, 96, 32, input_size = input_size, num_layers=7)
        model = VAE_flex_mse(encoder, decoder, encoded_dims)
        ##############################################
        # End of my code                             #
        ##############################################

    # to apply xavier_uniform:
    Initializer.initialize(model=model, initialization=init.xavier_uniform_, gain=init.calculate_gain('relu'))

    if args.loss == 'ce':
        loss = Loss_ce()
    else:
        loss = Loss_mse()

    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        model.cuda()
        loss.cuda()
        cudnn.enabled = True
        cudnn.benchmark = True

    print("Loading Data...")
    if args.dataset == "CIFAR10":
        data = CIFAR10DataLoader(args)
    elif args.dataset == "XRAB3":
        data = XRAB3_DataLoader(args)
    else:
        raise Exception("Invalid dataset in configs.dataset")
    print("Data loaded successfully\n")

    if args.loss == 'ce':
        trainer = Trainer_ce(model, loss, data.train_loader, data.test_loader, args)
    else:
        trainer = Trainer_mse(model, loss, data.train_loader, data.test_loader, args)

    if args.to_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n")
        except KeyboardInterrupt:
            print("Training had been Interrupted\n")

    if args.to_test:
        print("Testing on training data...")
        trainer.test_on_trainings_set()
        print("Testing Finished\n")


if __name__ == "__main__":
    main()
