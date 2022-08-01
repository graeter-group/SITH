import pytest
from src.SITH.SITH import SITH
from src.SITH.Utilities import *
from src.SITH.SithWriter import SithWriter
import pathlib


def test_initialized():
    sith = SITH()

# region File Input


def test_singleGood():
    sith = SITH()


def test_multiDeformedGood():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')


def test_noReference():
    pass


def test_noDeformed():
    pass


def test_emptyDefDirectory():
    pass


def test_emptyReference():
    pass


def test_emptyDeformed():
    pass


# endregion


def test_basic():
    sith = SITH()


def test_multiDeformed():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')

    sith.extractData()
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
    sith.extractData()
    # set manual values for each and check dot multiplication
    sith.energyAnalysis()


def test_fullRun():
    sith = SITH()
    sith.extractData()
    sith.energyAnalysis()


def test_killDOFs():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    sith.setKillDOFs([(1, 2), (2, 1, 5, 6)])
    sith.extractData()

def test_killDOFsBAD():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    sith.setKillDOFs([(1, 6), (2, 1, 5, 6)])
    sith.extractData()

def test_killDOFsBAD2():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    sith.setKillDOFs([(1, 6)])
    sith.extractData()

def test_killAtoms():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    sith.setKillAtoms([2])
    sith.extractData()

# region invalid Geometries (might be unnecessary or more for extractors?)


def test_badReference():
    pass


def test_badDeformed():
    pass


def test_incompleteReference():
    pass


def test_incompleteDeformed():
    pass

# endregion


def test_AAfromDaniel():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/glycine-ds-test/Gly-x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/glycine-ds-test/deformed')
    sith.setKillDOFs([(1, 16)])
    sith.extractData()
    sith.energyAnalysis()
    jp = SithWriter()
    blah = jp.buildDeltaQString(sith)
    jp.writeDeltaQ(sith)
    jp.compareEnergies(sith)
    jp.writeEnergyMatrix(sith)
    jp.writeSummary(sith)

def test_movedx0():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/moh-x0-1.7.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    sith.extractData()
    sith.energyAnalysis()
    jp = SithWriter()
    jp.writeDeltaQ(sith, "1.7-dq.txt")
    jp.writeError(sith, "1.7-error.txt")
    jp.writeEnergyMatrix(sith, "1.7-energy.txt")
    jp.writeSummary(sith, "1.7-summary.txt")
