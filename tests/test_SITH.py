import os
import pytest
from pathlib import Path

from src.SITH.SITH import SITH
from src.SITH.Utilities import *
from tests.test_variables import *


def test_initDefault():

    sith = SITH(x0string, xFstring)
    assert sith._referencePath == Path.cwd() / x0string
    assert sith._deformedPath == Path.cwd() / xFstring
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
    sith = SITH(x0string, deformedString)
    assert sith._referencePath == Path.cwd() / x0string
    assert sith._deformedPath == Path.cwd() / deformedString
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
    sith = SITH(x0string, xFstring)
    assert sith._referencePath == Path.cwd() / x0string
    assert sith._deformedPath == Path.cwd() / xFstring
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
        sith = SITH(dnePath, xFstring)
    assert str(e.value) == "Path to reference geometry data does not exist."

    with pytest.raises(Exception) as e:
        sith = SITH(x0string, dePath=dnePath)
    assert str(e.value) == "Path to deformed geometry data does not exist."

# tests _getContents


def test_emptyInput():
    with pytest.raises(Exception) as e:
        sith = SITH(emptyPath, xFstring)
        sith.extractData()
    assert str(e.value) == "Reference data file is empty."

    with pytest.raises(Exception) as e:
        sith = SITH(x0string, dePath=emptyPath)
        sith.extractData()
    assert str(e.value) == "One or more deformed files are empty."

    with pytest.raises(Exception) as e:
        sith = SITH(x0string, dePath=emptyDir)
        sith.extractData()
    assert str(e.value) == "Deformed directory is empty."


def test_getContents():
    sith = SITH(frankensteinPath, frankensteinPath)
    sith._getContents()
    assert sith._rData == frankenLines
    assert len(sith._dData) == 1
    assert sith._dData[0] == (
        Path(os.getcwd()+'/'+frankensteinPath), frankenLines)


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


def test_killDOFsBAD():
    sith = SITH(x0string, deformedString)
    sith.setKillDOFs([(1, 6), (2, 1, 5, 6)])
    sith.extractData()


def test_killDOFsBAD2():
    sith = SITH(x0string, deformedString)
    sith.setKillDOFs([(1, 6)])
    sith.extractData()


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
    sith = SITH('tests/glycine-ds-test/Gly-x0.fchk',
                'tests/glycine-ds-test/deformed/Gly-streched4.fchk')
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
    sith = SITH('tests/glycine-ds-test/Gly-x0.fchk',
                'tests/glycine-ds-test/deformed/Gly-streched4.fchk')
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
    ditto = deepcopy(refGeo)
    ditto._killDOFs([1])
    sith = SITH(x0string, xFstring)
    sith._reference = refGeo
    sith._deformed = [ditto]
    with pytest.raises(Exception) as e:
        sith._validateGeometries()
    assert str(
        e.value) == "Incompatible number of atoms or dimensions amongst input files."

    ditto = deepcopy(refGeo)
    ditto.nAtoms -= 1
    sith = SITH(x0string, xFstring)
    sith._reference = refGeo
    sith._deformed = [ditto]
    with pytest.raises(Exception) as e:
        sith._validateGeometries()
    assert str(
        e.value) == "Incompatible number of atoms or dimensions amongst input files."


def test_populateQ():
    sith = SITH(frankensteinPath, frankensteinPath)
    sith._reference = refGeo
    sith._deformed = [refGeo]
    sith._populateQ()
    coords2D = np.array(coords)[np.newaxis]
    assert np.array_equal(sith.q0, np.transpose(coords2D))
    assert np.array_equal(sith.qF, np.transpose(coords2D))
    assert np.array_equal(sith.deltaQ, np.zeros((len(coords), 1)))

    sith = SITH(frankensteinPath, frankensteinPath)
    sith._reference = refGeo
    ditto = deepcopy(refGeo)
    ditto.ric = 0.5 * ditto.ric
    sith._deformed = [ditto]
    sith._populateQ()
    coords2D = np.array(coords)[np.newaxis]
    assert np.array_equal(sith.q0, np.transpose(coords2D))
    assert np.array_equal(sith.qF, np.transpose(coords2D)*0.5)
    assert np.array_equal(sith.deltaQ, -np.transpose(coords2D)*0.5)


def test_energyAnalysis():
    sith = SITH(x0string, xFstring)
    sith.extractData()
    sith.deltaQ = dqSmall
    sith._reference.hessian = hessSmall
    sith._reference.dims = np.array([3, 1, 1, 1])
    sith.energyAnalysis()
    expectedDefE = np.transpose(dqSmall).dot(hessSmall).dot(dqSmall) / 2.
    assert sith.deformationEnergy == expectedDefE
    expectedEnergies = [(dqSmall[i] * 0.5 * hessSmall.dot(dqSmall))[i]
                        for i in range(3)]
    assert np.array_equal(sith.energies, expectedEnergies)
    blah = [sum(sith.pEnergies[:, i]) for i in range(len(sith.deformed))]
    assert all(x == 100 for x in [sum(sith.pEnergies[:, i])
               for i in range(len(sith.deformed))])


def test_energyAnalysis_Same():
    sith = SITH(frankensteinPath, frankensteinPath)
    with pytest.raises(Exception) as e:
        sith.energyAnalysis()
    assert str(e.value) == "Populate Q has not been executed so necessary data for analysis is lacking. This is likely due to not calling extractData()."
    sith.extractData()
    sith.energyAnalysis()
    assert np.array_equal(sith.energies, np.zeros((sith.reference.dims[0], 1)))
    assert sith.deformationEnergy[0, 0] == 0
    assert all([np.isnan(pE[0]) for pE in sith.pEnergies])


# region Integration Tests and Examples

def test_full_killed():  # full with no mismatched, kill valid, deformed directory, just check doesn't crash
    sith = SITH(x0string, deformedString)
    sith.setKillDOFs([(1, 2), (1, 3)])
    sith.extractData()
    sith.energyAnalysis()
    assert sith.energies.shape == (13, 5)
    assert sith.pEnergies.shape == (13, 5)
    blah = [sum(sith.pEnergies[:, i]) <=
            100.000001 for i in range(len(sith.deformed))]
    assert all(blah)
    assert sith.deformationEnergy.shape == (1, 5)


def test_multiDeformedGood():  # full no mismatched to remove, deformed directory
    sith = SITH(x0string, deformedString)
    sith.extractData()
    sith.energyAnalysis()
    assert sith.energies.shape == (15, 5)
    assert sith.pEnergies.shape == (15, 5)
    blah = [sum(sith.pEnergies[:, i]) <=
            100.000001 for i in range(len(sith.deformed))]
    assert all(blah)
    assert sith.deformationEnergy.shape == (1, 5)


def test_Glycine():  # full with mismatched to remove, deformed directory
    sith = SITH('tests/glycine-ds-test/Gly-x0.fchk',
                'tests/glycine-ds-test/deformed')
    sith.extractData()
    sith.energyAnalysis()
    assert sith.energies.shape == (80, 10)
    assert sith.pEnergies.shape == (80, 10)
    blah = [sum(sith.pEnergies[:, i]) <=
            100.000001 for i in range(len(sith.deformed))]
    assert all(blah)
    assert sith.deformationEnergy.shape == (1, 10)


def test_Alanine():  # full with valid kill invalid results
    sith = SITH('tests/Ala-stretched00.fchk',
                'tests/Ala-stretched10.fchk')
    sith.setKillDOFs([(1, 19)])
    with pytest.raises(Exception) as e:
        sith.extractData()
    assert str(
        e.value) == "Deformed geometry (Ala-stretched10) is missing reference DOF (4, 1, 5)."


def test_movedx0():
    sith = SITH('tests/moh-x0-1.7.fchk',
                'tests/deformed')
    sith.extractData()
    sith.energyAnalysis()
    assert sith.energies.shape == (15, 5)
    assert sith.pEnergies.shape == (15, 5)
    blah = [sum(sith.pEnergies[:, i]) <=
            100.000001 for i in range(len(sith.deformed))]
    assert all(blah)
    assert sith.deformationEnergy.shape == (1, 5)

# endregion
