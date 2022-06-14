import numpy as np
import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('-i', help = "Please specify a path to input directory after the --i flag.")
parser.add_argument('--d', help = "Please specify a directory after the --d flag.")
parser.add_argument('--v', nargs = '+', help = "Either specify a maximum strain and a modus or call with 'n' if you wish no vmd analysis.")
args = parser.parse_args()

if __name__ == "__main__":
    pass