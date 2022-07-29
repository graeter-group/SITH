from src.SITH.Utilities import *
from src.SITH.SITH import SITH
from src.SITH.SithWriter import SithWriter


def test_buildDQ():
    sith = SITH()
    sith.extractData()
    sith.energyAnalysis()
    jp = SithWriter()
    blah = jp.buildDeltaQString(sith)
    print(blah)


def test_buildDQ2():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    sith.extractData()
    sith.energyAnalysis()
    jp = SithWriter()
    blah = jp.buildDeltaQString(sith)
    print(blah)


def test_compareEnergies():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    sith.extractData()
    sith.energyAnalysis()
    jp = SithWriter()
    blah = jp.buildDeltaQString(sith)
    jp.writeDeltaQ(sith)
    jp.compareEnergies(sith)
    jp.writeEnergyMatrix(sith)
    jp.writeSummary(sith)
    
def test_writeAll():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    sith.extractData()
    sith.energyAnalysis()
    jp = SithWriter()
    assert jp.writeAll(sith)
    blah = 2
