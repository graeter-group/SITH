import pytest
from pathlib import Path

from src.SITH.SITH import SITH
from src.SITH.Utilities import *
from src.SITH.SithWriter import SithWriter
from tests.test_variables import *


def test_initDefault():
    sith = SITH()
    # TODO change this to check that it's just working directory plus x0.fchk and xF.fchk
    assert sith._referencePath == defaultRefPath
    assert sith._deformedPath == defaultDefPath
    assert sith.energies is None
    assert sith.deformationEnergy is None
    assert sith.pEnergies is None
    assert sith.reference is None
    assert sith.deformed is None
    assert sith.q0 is None
    assert sith.qF is None
    assert sith.deltaQ is None
    assert not sith._kill
    assert sith._killAtoms == list()
    assert sith._killDOFs == list()


def test_initDir():
    sith = SITH(defaultRefPath, defDirPath)
    # TODO change this to check that it's just working directory plus x0.fchk and xF.fchk
    assert sith._referencePath == defaultRefPath
    assert sith._deformedPath == defDirPath
    assert sith.energies is None
    assert sith.deformationEnergy is None
    assert sith.pEnergies is None
    assert sith.reference is None
    assert sith.deformed is None
    assert sith.q0 is None
    assert sith.qF is None
    assert sith.deltaQ is None
    assert not sith._kill
    assert sith._killAtoms == list()
    assert sith._killDOFs == list()


def test_initFile():
    sith = SITH(defaultRefPath, defaultDefPath)
    # TODO change this to check that it's just working directory plus x0.fchk and xF.fchk
    assert sith._referencePath == defaultRefPath
    assert sith._deformedPath == defaultDefPath
    assert sith.energies is None
    assert sith.deformationEnergy is None
    assert sith.pEnergies is None
    assert sith.reference is None
    assert sith.deformed is None
    assert sith.q0 is None
    assert sith.qF is None
    assert sith.deltaQ is None
    assert not sith._kill
    assert sith._killAtoms == list()
    assert sith._killDOFs == list()


def test_referenceProp():
    sith = SITH(frankensteinPath, frankensteinPath)
    sith.extractData()
    assert sith.reference == refGeo


def test_deformedProp():
    sith = SITH(frankensteinPath, frankensteinPath)
    sith.extractData()
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == refGeo


def test_hessianProp():
    sith = SITH(frankensteinPath, frankensteinPath)
    sith.extractData()
    assert np.array_equal(sith.hessian, refGeo.hessian)


def test_validateFiles():
    with pytest.raises(Exception) as e:
        sith = SITH(dnePath)
    assert str(e.value) == "Path to reference geometry data does not exist."

    with pytest.raises(Exception) as e:
        sith = SITH(dePath=dnePath)
    assert str(e.value) == "Path to deformed geometry data does not exist."

# tests _getContents


def test_emptyInput():
    with pytest.raises(Exception) as e:
        sith = SITH(emptyPath)
        sith.extractData()
    assert str(e.value) == "Reference data file is empty."

    with pytest.raises(Exception) as e:
        sith = SITH(dePath=emptyPath)
        sith.extractData()
    assert str(e.value) == "One or more deformed files are empty."

    with pytest.raises(Exception) as e:
        sith = SITH(dePath=emptyDir)
        sith.extractData()
    assert str(e.value) == "Deformed directory is empty."


def test_getContents():
    sith = SITH(frankensteinPath, frankensteinPath)
    sith._getContents()
    assert sith._rData == frankenLines
    assert len(sith._dData) == 1
    assert sith._dData[0] == (frankensteinPath, frankenLines)


def test_getContentsDir():
    sith = SITH(frankensteinPath, frankensteinDir)
    sith._getContents()
    assert sith._rData == frankenLines
    assert len(sith._dData) == 2
    assert sith._dData[0][1] == frankenLines
    assert sith._dData[1][1] == frankenLines


def test_extractDataFile():
    sith = SITH(frankensteinPath, frankensteinPath)
    sith.extractData()
    assert sith.reference == refGeo
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == refGeo
    assert not sith._kill


def test_extractDataDir():
    sith = SITH(frankensteinPath, frankensteinDir)
    sith.extractData()
    assert sith.reference == refGeo
    assert len(sith.deformed) == 2
    refGeoCopy.name = 'frankenstein-1'
    assert sith.deformed[0] == refGeoCopy
    refGeoCopy.name = 'frankenstein-2'
    assert sith.deformed[1] == refGeoCopy
    assert not sith._kill


def test_setKillDOFs():
    sith = SITH(frankensteinPath, frankensteinPath)
    killDOFs = [(1, 2), (2, 1, 3)]
    sith.setKillDOFs(killDOFs)
    assert sith._kill
    assert np.array_equal(sith._killDOFs, killDOFs)


def test_setKillAtoms():
    sith = SITH(frankensteinPath, frankensteinPath)
    killAtoms = [1, 6]
    sith.setKillDOFs(killAtoms)
    assert sith._kill
    assert np.array_equal(sith._killDOFs, killAtoms)


def test_kill():
    # Doesn't directly test SITH.__kill() or SITH.__killDOFs(dofs) because they are private but ensures that they work
    # properly in cases where they are called
    sith = SITH(frankensteinPath, frankensteinPath)
    killDOFs = [(1, 2)]
    sith.setKillDOFs(killDOFs)
    sith.extractData()
    killedGeo = deepcopy(refGeo)
    killedGeo._killDOFs([0])
    assert sith.reference == killedGeo
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == killedGeo


def test_killAtom():
    # Doesn't directly test SITH.__kill() or SITH.__killDOFs(dofs) because they are private but ensures that they work
    # properly in cases where they are called
    sith = SITH(frankensteinPath, frankensteinPath)
    killAtoms = [2]
    sith.setKillAtoms(killAtoms)
    sith.extractData()
    killedGeo = deepcopy(refGeo)
    killedGeo._killDOFs([0, 5, 6, 7, 12])
    assert sith.reference == killedGeo
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == killedGeo


def test_kill_bad():
    sith = SITH(frankensteinPath, frankensteinPath)
    killDOFs = [(1, 19)]
    sith.setKillDOFs(killDOFs)
    sith.extractData()
    assert sith.reference == refGeo
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == refGeo


def test_removeMismatchedDOFs_noKill():
    # no atoms specified for kill and no mismatch
    sith = SITH(frankensteinPath, frankensteinPath)
    sith._reference = refGeo
    sith._deformed = [refGeo]
    sith.removeMismatchedDOFs()
    assert sith.reference == refGeo
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == refGeo

    # no atoms specified for kill but should kill DOF (1, 16) from deformed cus not in reference
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/glycine-ds-test/Gly-x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/glycine-ds-test/deformed/Gly-streched4.fchk')
    sith._getContents()
    extractor = Extractor(sith._referencePath, sith._rData)
    extractor._extract()
    sith._reference = extractor.getGeometry()
    extractor = Extractor(sith._deformedPath, sith._dData[0][1])
    extractor._extract()
    sith._deformed = [extractor.getGeometry()]
    ref = deepcopy(sith.reference)
    defd = deepcopy(sith.deformed[0])
    sith.removeMismatchedDOFs()
    defd._killDOFs([4])
    assert sith.reference == ref
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == defd


def test_removeMismatchedDOFs_kill():
    # atoms specified for kill but should kill DOF (1, 16) from deformed cus not in reference
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/glycine-ds-test/Gly-x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/glycine-ds-test/deformed/Gly-streched4.fchk')
    sith._getContents()
    extractor = Extractor(sith._referencePath, sith._rData)
    extractor._extract()
    ref = extractor.getGeometry()
    extractor = Extractor(sith._deformedPath, sith._dData[0][1])
    extractor._extract()
    defd = extractor.getGeometry()
    sith.setKillDOFs([(1, 2)])
    sith.extractData()
    ref._killDOFs([0])
    defd._killDOFs([0, 4])
    assert sith.reference == ref
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == defd


def test_validateGeometries():
    pass


def test_populatQ():
    pass


# region File Input

#setKill, extract, analysis


def test_singleGood():
    sith = SITH()
    assert sith._referencePath == defaultRefPath
    assert sith._deformedPath == defaultDefPath
    assert sith.energies is None
    assert sith.deformationEnergy is None
    assert sith.pEnergies is None
    assert sith.reference is None
    assert sith.deformed is None
    assert sith.q0 is None
    assert sith.qF is None
    assert sith.deltaQ is None
    assert not sith._kill
    assert sith._killAtoms == list()
    assert sith._killDOFs == list()


def test_multiDeformedGood():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')


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
    jp.writeSummary(sith)


def test_alanine():
    sith = SITH('/hits/fast/mbm/farrugma/sw/SITH/tests/Ala-stretched00.fchk',
                '/hits/fast/mbm/farrugma/sw/SITH/tests/Ala-stretched10.fchk')
    sith.setKillDOFs([(1, 19)])
    with pytest.raises(Exception) as e:
        sith.extractData()
    assert str(
        e.value) == "Deformed geometry (Ala-stretched10) is missing reference DOF (4, 1, 5)."


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
