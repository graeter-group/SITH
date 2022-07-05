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
    jedi = JEDI('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk', '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')


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
    jedi = JEDI('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk', '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    jedi.energyAnalysis()

def test_populateQ():
    jedi = JEDI()
    #check that q0, qF, and delta_q are all correct

def test_totalEnergies():
    jedi = JEDI()

def test_energyMatrix():
    jedi = JEDI()

def test_fullEnergyAnalysis():
    jedi = JEDI()
    #set manual values for each and check dot multiplication
    jedi.energyAnalysis()

def test_fullRun():
    jedi = JEDI()
    jedi.energyAnalysis()

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