# pylint: disable='missing-module-docstring'
from .api import *
import importlib.metadata

__version__ = importlib.metadata.version(__name__)
