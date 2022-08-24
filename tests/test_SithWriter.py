from src.SITH.Utilities import *
from src.SITH.SITH import SITH
from src.SITH.SithWriter import *
from tests.test_resources import *

"""There aren't necessarily very many options for testing here which are not unnecessarily detailed so 
the testing of SithWriter is kept to a minimum of simply 'it works' and manually checking the output files. 
This can be expanded upon later if necessary."""


def test_buildDQ():
    sith = SITH(x0string, xFstring)
    sith.extractData()
    sith.energyAnalysis()

    blah = buildDeltaQString(sith)


def test_buildDQ2():
    sith = SITH(x0string, deformedString)
    sith.extractData()
    sith.energyAnalysis()

    blah = buildDeltaQString(sith)


def test_compareEnergies():
    sith = SITH(x0string, deformedString)
    sith.extractData()
    sith.energyAnalysis()

    assert writeDeltaQ(sith)
    assert writeError(sith)
    assert writeEnergyMatrix(sith)
    assert writeSummary(sith)


def test_writeAll():
    sith = SITH(x0string, deformedString)
    sith.extractData()
    sith.energyAnalysis()

    assert writeAll(sith)


def test_writeXYZ():
    sith = SITH(x0string, deformedString)
    sith.extractData()
    sith.energyAnalysis()

    writeXYZ(sith.reference)


def test_writeSummaryXYZ():
    sith = SITH(x0string, deformedString)
    sith.extractData()
    sith.energyAnalysis()

    assert writeSummary(sith, "xyz", includeXYZ=True)
