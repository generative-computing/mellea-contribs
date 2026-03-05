"""Comprehensive alias database for common Python imports.

This module provides mappings from common aliases and names to their proper
import statements, as well as common module relocation patterns for fixing
incorrect import paths.
"""

# ~100+ common aliases organized by category
COMMON_ALIASES: dict[str, str] = {
    # =========================================================================
    # Data Science & Numerical Computing
    # =========================================================================
    "np": "import numpy as np",
    "numpy": "import numpy",
    "pd": "import pandas as pd",
    "pandas": "import pandas",
    "plt": "import matplotlib.pyplot as plt",
    "matplotlib": "import matplotlib",
    "sns": "import seaborn as sns",
    "seaborn": "import seaborn",
    "sp": "import scipy as sp",
    "scipy": "import scipy",
    "stats": "from scipy import stats",
    "xgb": "import xgboost as xgb",
    "xgboost": "import xgboost",
    "lgb": "import lightgbm as lgb",
    "lightgbm": "import lightgbm",
    "cv2": "import cv2",
    "PIL": "from PIL import Image",
    "Image": "from PIL import Image",
    # =========================================================================
    # Machine Learning & Deep Learning
    # =========================================================================
    "tf": "import tensorflow as tf",
    "tensorflow": "import tensorflow",
    "keras": "import keras",
    "torch": "import torch",
    "nn": "from torch import nn",
    "F": "import torch.nn.functional as F",
    "optim": "from torch import optim",
    "sk": "import sklearn as sk",
    "sklearn": "import sklearn",
    # =========================================================================
    # Standard Library - pathlib
    # =========================================================================
    "Path": "from pathlib import Path",
    "PurePath": "from pathlib import PurePath",
    "PosixPath": "from pathlib import PosixPath",
    "WindowsPath": "from pathlib import WindowsPath",
    # =========================================================================
    # Standard Library - datetime
    # =========================================================================
    "datetime": "from datetime import datetime",
    "timedelta": "from datetime import timedelta",
    "date": "from datetime import date",
    "time": "from datetime import time",
    "timezone": "from datetime import timezone",
    # =========================================================================
    # Standard Library - collections
    # =========================================================================
    "defaultdict": "from collections import defaultdict",
    "Counter": "from collections import Counter",
    "deque": "from collections import deque",
    "namedtuple": "from collections import namedtuple",
    "OrderedDict": "from collections import OrderedDict",
    "ChainMap": "from collections import ChainMap",
    # =========================================================================
    # Standard Library - dataclasses
    # =========================================================================
    "dataclass": "from dataclasses import dataclass",
    "field": "from dataclasses import field",
    "asdict": "from dataclasses import asdict",
    "astuple": "from dataclasses import astuple",
    # =========================================================================
    # Standard Library - typing
    # =========================================================================
    "Optional": "from typing import Optional",
    "List": "from typing import List",
    "Dict": "from typing import Dict",
    "Tuple": "from typing import Tuple",
    "Set": "from typing import Set",
    "Any": "from typing import Any",
    "Union": "from typing import Union",
    "Callable": "from typing import Callable",
    "TypeVar": "from typing import TypeVar",
    "Generic": "from typing import Generic",
    "Literal": "from typing import Literal",
    "ClassVar": "from typing import ClassVar",
    "Final": "from typing import Final",
    "Annotated": "from typing import Annotated",
    "Protocol": "from typing import Protocol",
    "TypedDict": "from typing import TypedDict",
    "Iterator": "from typing import Iterator",
    "Iterable": "from typing import Iterable",
    "Sequence": "from typing import Sequence",
    "Mapping": "from typing import Mapping",
    "MutableMapping": "from typing import MutableMapping",
    "TYPE_CHECKING": "from typing import TYPE_CHECKING",
    # =========================================================================
    # Standard Library - functools & itertools
    # =========================================================================
    "partial": "from functools import partial",
    "lru_cache": "from functools import lru_cache",
    "cache": "from functools import cache",
    "wraps": "from functools import wraps",
    "reduce": "from functools import reduce",
    "chain": "from itertools import chain",
    "combinations": "from itertools import combinations",
    "permutations": "from itertools import permutations",
    "product": "from itertools import product",
    "groupby": "from itertools import groupby",
    "islice": "from itertools import islice",
    # =========================================================================
    # Standard Library - abc
    # =========================================================================
    "ABC": "from abc import ABC",
    "abstractmethod": "from abc import abstractmethod",
    "abstractproperty": "from abc import abstractproperty",
    # =========================================================================
    # Standard Library - contextlib
    # =========================================================================
    "contextmanager": "from contextlib import contextmanager",
    "asynccontextmanager": "from contextlib import asynccontextmanager",
    "suppress": "from contextlib import suppress",
    # =========================================================================
    # Standard Library - enum
    # =========================================================================
    "Enum": "from enum import Enum",
    "IntEnum": "from enum import IntEnum",
    "Flag": "from enum import Flag",
    "auto": "from enum import auto",
    # =========================================================================
    # Standard Library - re (regex)
    # =========================================================================
    "re": "import re",
    "Pattern": "from re import Pattern",
    "Match": "from re import Match",
    # =========================================================================
    # Standard Library - json
    # =========================================================================
    "json": "import json",
    # =========================================================================
    # Standard Library - os & sys
    # =========================================================================
    "os": "import os",
    "sys": "import sys",
    "environ": "from os import environ",
    "getenv": "from os import getenv",
    # =========================================================================
    # Standard Library - io
    # =========================================================================
    "StringIO": "from io import StringIO",
    "BytesIO": "from io import BytesIO",
    # =========================================================================
    # Standard Library - copy
    # =========================================================================
    "copy": "from copy import copy",
    "deepcopy": "from copy import deepcopy",
    # =========================================================================
    # Standard Library - logging
    # =========================================================================
    "logging": "import logging",
    "Logger": "from logging import Logger",
    "getLogger": "from logging import getLogger",
    # =========================================================================
    # Standard Library - warnings
    # =========================================================================
    "warnings": "import warnings",
    "warn": "from warnings import warn",
    # =========================================================================
    # Standard Library - subprocess
    # =========================================================================
    "subprocess": "import subprocess",
    "Popen": "from subprocess import Popen",
    "PIPE": "from subprocess import PIPE",
    # =========================================================================
    # Standard Library - threading & multiprocessing
    # =========================================================================
    "threading": "import threading",
    "Thread": "from threading import Thread",
    "Lock": "from threading import Lock",
    "multiprocessing": "import multiprocessing",
    "Process": "from multiprocessing import Process",
    "Pool": "from multiprocessing import Pool",
    # =========================================================================
    # Web & API - requests
    # =========================================================================
    "requests": "import requests",
    # =========================================================================
    # Web & API - BeautifulSoup
    # =========================================================================
    "BeautifulSoup": "from bs4 import BeautifulSoup",
    "bs4": "import bs4",
    # =========================================================================
    # Web & API - Flask
    # =========================================================================
    "Flask": "from flask import Flask",
    "request": "from flask import request",
    "jsonify": "from flask import jsonify",
    "render_template": "from flask import render_template",
    "Blueprint": "from flask import Blueprint",
    # =========================================================================
    # Web & API - FastAPI
    # =========================================================================
    "FastAPI": "from fastapi import FastAPI",
    "HTTPException": "from fastapi import HTTPException",
    "Depends": "from fastapi import Depends",
    "Query": "from fastapi import Query",
    "Body": "from fastapi import Body",
    "APIRouter": "from fastapi import APIRouter",
    # =========================================================================
    # Web & API - Pydantic
    # =========================================================================
    "BaseModel": "from pydantic import BaseModel",
    "Field": "from pydantic import Field",
    "validator": "from pydantic import validator",
    "root_validator": "from pydantic import root_validator",
    # =========================================================================
    # Async
    # =========================================================================
    "asyncio": "import asyncio",
    "aiohttp": "import aiohttp",
    "async_timeout": "import async_timeout",
    # =========================================================================
    # Testing
    # =========================================================================
    "pytest": "import pytest",
    "mock": "from unittest import mock",
    "patch": "from unittest.mock import patch",
    "MagicMock": "from unittest.mock import MagicMock",
    "Mock": "from unittest.mock import Mock",
    "AsyncMock": "from unittest.mock import AsyncMock",
    "unittest": "import unittest",
    "TestCase": "from unittest import TestCase",
    # =========================================================================
    # Database
    # =========================================================================
    "sqlalchemy": "import sqlalchemy",
    "create_engine": "from sqlalchemy import create_engine",
    "Column": "from sqlalchemy import Column",
    "Integer": "from sqlalchemy import Integer",
    "String": "from sqlalchemy import String",
    "ForeignKey": "from sqlalchemy import ForeignKey",
    "sqlite3": "import sqlite3",
    # =========================================================================
    # Serialization
    # =========================================================================
    "pickle": "import pickle",
    "yaml": "import yaml",
    "toml": "import toml",
    "csv": "import csv",
}

# Common relocation patterns for wrong import paths
MODULE_RELOCATIONS: dict[str, dict[str, str]] = {
    "sklearn": {
        # Linear models
        "LinearRegression": "sklearn.linear_model",
        "LogisticRegression": "sklearn.linear_model",
        "Ridge": "sklearn.linear_model",
        "Lasso": "sklearn.linear_model",
        "ElasticNet": "sklearn.linear_model",
        "SGDClassifier": "sklearn.linear_model",
        "SGDRegressor": "sklearn.linear_model",
        # Ensemble
        "RandomForestClassifier": "sklearn.ensemble",
        "RandomForestRegressor": "sklearn.ensemble",
        "GradientBoostingClassifier": "sklearn.ensemble",
        "GradientBoostingRegressor": "sklearn.ensemble",
        "AdaBoostClassifier": "sklearn.ensemble",
        "AdaBoostRegressor": "sklearn.ensemble",
        "ExtraTreesClassifier": "sklearn.ensemble",
        "ExtraTreesRegressor": "sklearn.ensemble",
        "VotingClassifier": "sklearn.ensemble",
        "BaggingClassifier": "sklearn.ensemble",
        # Model selection
        "train_test_split": "sklearn.model_selection",
        "cross_val_score": "sklearn.model_selection",
        "GridSearchCV": "sklearn.model_selection",
        "RandomizedSearchCV": "sklearn.model_selection",
        "KFold": "sklearn.model_selection",
        "StratifiedKFold": "sklearn.model_selection",
        "cross_validate": "sklearn.model_selection",
        # Preprocessing
        "StandardScaler": "sklearn.preprocessing",
        "MinMaxScaler": "sklearn.preprocessing",
        "LabelEncoder": "sklearn.preprocessing",
        "OneHotEncoder": "sklearn.preprocessing",
        "OrdinalEncoder": "sklearn.preprocessing",
        "Normalizer": "sklearn.preprocessing",
        "Binarizer": "sklearn.preprocessing",
        "PolynomialFeatures": "sklearn.preprocessing",
        # Metrics
        "accuracy_score": "sklearn.metrics",
        "precision_score": "sklearn.metrics",
        "recall_score": "sklearn.metrics",
        "f1_score": "sklearn.metrics",
        "confusion_matrix": "sklearn.metrics",
        "classification_report": "sklearn.metrics",
        "mean_squared_error": "sklearn.metrics",
        "mean_absolute_error": "sklearn.metrics",
        "r2_score": "sklearn.metrics",
        "roc_auc_score": "sklearn.metrics",
        "roc_curve": "sklearn.metrics",
        # Clustering
        "KMeans": "sklearn.cluster",
        "DBSCAN": "sklearn.cluster",
        "AgglomerativeClustering": "sklearn.cluster",
        "SpectralClustering": "sklearn.cluster",
        # Decomposition
        "PCA": "sklearn.decomposition",
        "TruncatedSVD": "sklearn.decomposition",
        "NMF": "sklearn.decomposition",
        "LDA": "sklearn.decomposition",
        # Tree
        "DecisionTreeClassifier": "sklearn.tree",
        "DecisionTreeRegressor": "sklearn.tree",
        # SVM
        "SVC": "sklearn.svm",
        "SVR": "sklearn.svm",
        "LinearSVC": "sklearn.svm",
        # Neighbors
        "KNeighborsClassifier": "sklearn.neighbors",
        "KNeighborsRegressor": "sklearn.neighbors",
        # Naive Bayes
        "GaussianNB": "sklearn.naive_bayes",
        "MultinomialNB": "sklearn.naive_bayes",
        "BernoulliNB": "sklearn.naive_bayes",
        # Pipeline
        "Pipeline": "sklearn.pipeline",
        "make_pipeline": "sklearn.pipeline",
        # Feature selection
        "SelectKBest": "sklearn.feature_selection",
        "RFE": "sklearn.feature_selection",
        # Imputation
        "SimpleImputer": "sklearn.impute",
    },
    "tensorflow": {
        "keras": "tensorflow.keras",
        "layers": "tensorflow.keras.layers",
        "Model": "tensorflow.keras.models",
        "Sequential": "tensorflow.keras.models",
        "Dense": "tensorflow.keras.layers",
        "Conv2D": "tensorflow.keras.layers",
        "LSTM": "tensorflow.keras.layers",
        "Dropout": "tensorflow.keras.layers",
        "BatchNormalization": "tensorflow.keras.layers",
        "Adam": "tensorflow.keras.optimizers",
        "SGD": "tensorflow.keras.optimizers",
        "callbacks": "tensorflow.keras.callbacks",
        "EarlyStopping": "tensorflow.keras.callbacks",
        "ModelCheckpoint": "tensorflow.keras.callbacks",
    },
    "torch": {
        "DataLoader": "torch.utils.data",
        "Dataset": "torch.utils.data",
        "TensorDataset": "torch.utils.data",
        "Adam": "torch.optim",
        "SGD": "torch.optim",
        "CrossEntropyLoss": "torch.nn",
        "MSELoss": "torch.nn",
        "Linear": "torch.nn",
        "Conv2d": "torch.nn",
        "LSTM": "torch.nn",
        "Module": "torch.nn",
    },
    "scipy": {
        "norm": "scipy.stats",
        "uniform": "scipy.stats",
        "pearsonr": "scipy.stats",
        "spearmanr": "scipy.stats",
        "ttest_ind": "scipy.stats",
        "chi2_contingency": "scipy.stats",
        "minimize": "scipy.optimize",
        "curve_fit": "scipy.optimize",
        "fsolve": "scipy.optimize",
        "interp1d": "scipy.interpolate",
        "griddata": "scipy.interpolate",
        "distance": "scipy.spatial",
        "cdist": "scipy.spatial.distance",
        "pdist": "scipy.spatial.distance",
        "sparse": "scipy.sparse",
        "csr_matrix": "scipy.sparse",
        "csc_matrix": "scipy.sparse",
        "fft": "scipy.fft",
        "rfft": "scipy.fft",
    },
}
