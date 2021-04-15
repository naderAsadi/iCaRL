import argparse

def get_parser():
    parser = argparse.ArgumentParser("IncLearner",
                                     description="Incremental Learning trainer.")

    # Model
    parser.add_argument("-m", "--model", default="icarl", type=str,
                        help="Incremental learner to train.")
    parser.add_argument("-c", "--convnet", default="resnet18", type=str,
                        help="Backbone convnet.")
    parser.add_argument("-memory", "--memorysize", default=2000, type=int,
                        help="Max number of storable examplars.")
    
    # Data
    parser.add_argument("-d", "--dataset", default="cifar100", type=str,
                        help="Dataset to test on.")
    parser.add_argument("-inc", "--increment", default="10", type=str,
                        help="Number of class to add per task.")
    parser.add_argument("-b", "--batchsize", default=128, type=int,
                        help="Batch size.")

    # Training
    parser.add_argument("-lr", "--lr", default=0.2, type=float,
                        help="Learning rate.")
    parser.add_argument("-wd", "--weightdecay", default=0.00005, type=float,
                        help="Weight decay.")
    parser.add_argument("-sc", "--scheduling", default=[49, 63], nargs="*", type=int,
                        help="Epoch step where to reduce the learning rate.")
    parser.add_argument("-lrdecay", "--lrdecay", default=1/5, type=float,
                        help="LR multiplied by it.")
    parser.add_argument("-opt", "--optimizer", default="sgd", type=str,
                        help="Optimizer to use.")
    parser.add_argument("-e", "--epochs", default=100, type=int,
                        help="Number of epochs per task.")

    return parser