from src.SITH.Utilities import *
from src.SITH.SITH import SITH
from src.SITH.SithWriter import SithWriter
from tests.test_variables import *

"""There aren't necessarily very many options for testing here which are not unnecessarily detailed so 
the testing of SithWriter is kept to a minimum of simply 'it works' and manually checking the output files. 
This can be expanded upon later if necessary."""


def test_buildDQ():
    sith = SITH(x0string, xFstring)
    sith.extractData()
    sith.energyAnalysis()
    jp = SithWriter()
    blah = jp.buildDeltaQString(sith)
    print(blah)


def test_buildDQ2():
    sith = SITH(x0string, deformedString)
    sith.extractData()
    sith.energyAnalysis()
    jp = SithWriter()
    blah = jp.buildDeltaQString(sith)
    print(blah)


def test_compareEnergies():
    sith = SITH(x0string, deformedString)
    sith.extractData()
    sith.energyAnalysis()
    jp = SithWriter()
    blah = jp.buildDeltaQString(sith)
    assert jp.writeDeltaQ(sith)
    assert jp.writeError(sith)
    assert jp.writeEnergyMatrix(sith)
    assert jp.writeSummary(sith)


def test_writeAll():
    sith = SITH(x0string, deformedString)
    sith.extractData()
    sith.energyAnalysis()
    jp = SithWriter()
    assert jp.writeAll(sith)
    blah = 2


def test_writeSummaryXYZ():
    sith = SITH(x0string, deformedString)
    sith.extractData()
    sith.energyAnalysis()
    jp = SithWriter()
    assert jp.writeSummary(sith, "xyzSummary.txt", includeXYZ=True)
