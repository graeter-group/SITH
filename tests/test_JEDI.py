from genericpath import exists
from numpy import extract
import pytest
from JEDI import JEDI
from SITH_Utilities import *
import pathlib


def test_initialized():
    jedi = JEDI()

# region File Input


def test_singleGood():
    jedi = JEDI()


def test_multiDeformedGood():
    pass


def test_noRelaxed():
    pass


def test_noDeformed():
    pass


def test_emptyDefDirectory():
    pass


def test_emptyRelaxed():
    pass


def test_emptyDeformed():
    pass


# endregion



def test_basic():
    jedi = JEDI()

def test_multiDeformed():
    jedi = JEDI()

def test_populateQ():
    jedi = JEDI()

def test_totalEnergies():
    jedi = JEDI()

def test_energyMatrix():
    jedi = JEDI()

def test_fullEnergyAnalysis():
    jedi = JEDI()
    jedi.energyAnalysis()

def test_fullRun():
    jedi = JEDI()

#region invalid Geometries (might be unnecessary or more for extractors?)

def test_badRelaxed():
    pass
def test_badDeformed():
    pass
def test_incompleteRelaxed():
    pass
def test_incompleteDeformed():
    pass

#endregion