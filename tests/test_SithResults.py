from genericpath import exists
from numpy import extract
import pytest
from SITH_Utilities import *
from SITH import SITH
import pathlib
from SithResults import SithResults


def test_buildDQ():
    sith = SITH()
    sith.energyAnalysis()
    jp = SithResults()
    blah = jp.buildDeltaQString(sith)
    print(blah)


def test_buildDQ2():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    sith.energyAnalysis()
    jp = SithResults()
    blah = jp.buildDeltaQString(sith)
    print(blah)


def test_compareEnergies():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    sith.energyAnalysis()
    jp = SithResults()
    blah = jp.buildDeltaQString(sith)
    jp.writeDeltaQ(sith)
    jp.compareEnergies(sith)
    jp.writeEnergyMatrix(sith)
    jp.writeSummary(sith)
