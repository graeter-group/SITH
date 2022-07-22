import pytest
from SITH import SITH
from SITH_Utilities import *
from SithResults import SithResults
import pathlib


def test_initialized():
    sith = SITH()

# region File Input


def test_singleGood():
    sith = SITH()


def test_multiDeformedGood():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')


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
    sith = SITH()


def test_multiDeformed():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    sith.energyAnalysis()


def test_populateQ():
    sith = SITH()
    # check that q0, qF, and delta_q are all correct


def test_totalEnergies():
    sith = SITH()


def test_energyMatrix():
    sith = SITH()


def test_fullEnergyAnalysis():
    sith = SITH()
    # set manual values for each and check dot multiplication
    sith.energyAnalysis()


def test_fullRun():
    sith = SITH()
    sith.energyAnalysis()

def test_killAtoms():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    sith.killDOFs([0, 12])
    sith.populateQ()
    sith.energyAnalysis()

# region invalid Geometries (might be unnecessary or more for extractors?)


def test_badRelaxed():
    pass


def test_badDeformed():
    pass


def test_incompleteRelaxed():
    pass


def test_incompleteDeformed():
    pass

# endregion

def test_AAfromDaniel():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    sith.energyAnalysis()
    jp = SithResults()
    blah = jp.buildDeltaQString(sith)
    jp.writeDeltaQ(sith)
    jp.compareEnergies(sith)
    jp.writeEnergyMatrix(sith)
    jp.writeSummary(sith)