#!/usr/bin/env python3

import argparse

from tools import test, train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(train=False, test=False)
    args = parser.parse_args()

    if args.train:
        train()
    if args.test:
        test()
