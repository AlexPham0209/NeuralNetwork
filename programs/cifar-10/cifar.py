import sys
sys.path.insert(1, '../NeuralNetwork')

import cupy as cp
import numpy as np

import json
import random

from src.layers.dense import Dense
from src.layers.conv2d import Conv2D
from src.layers.pooling import MaxPooling
from src.layers.flatten import Flatten
from src.network import Model

import src.activation as act
import src.loss as ls

