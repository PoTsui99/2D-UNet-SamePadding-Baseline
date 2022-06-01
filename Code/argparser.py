import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # name refers to the name of the folder which includes model.pth, specially for test mode
    parser.add_argument('--name', default=None, help='model name(training mode)/folder name(testing mode)')
    parser.add_argument('--batch-size', '--b', default=8, type=int, help='size of mini-batch (default: 8)')
    parser.add_argument('--early-stop', '--es', default=20, type=int,
                        help='early stopping (default: 30)')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of total epochs')
    # optimizer参数
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='optimizer, choice=[Adam, SGD]')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        help='learning rate')
    # SGD参数
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=bool, help='nesterov')

    args = parser.parse_args()

    return args