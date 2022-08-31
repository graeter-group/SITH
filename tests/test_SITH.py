import os
import pytest
from pathlib import Path

from src.SITH.SITH import SITH
from src.SITH.Utilities import *
from tests.test_resources import *
from src.SITH.SithWriter import *


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
    sith.extract_data()
    assert sith.reference == refGeo


def test_deformedProp():
    sith = SITH(frankensteinPath, frankensteinPath)
    sith.extract_data()
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == refGeo


def test_hessianProp():
    sith = SITH(frankensteinPath, frankensteinPath)
    sith.extract_data()
    assert compare_arrays(sith.hessian, refGeo.hessian)


def test_validate_files():
    with pytest.raises(Exception) as e:
        sith = SITH(dnePath, xFstring)
    assert str(e.value) == "Path to reference geometry data does not exist."

    with pytest.raises(Exception) as e:
        sith = SITH(x0string, dePath=dnePath)
    assert str(e.value) == "Path to deformed geometry data does not exist."

# tests _get_contents


def test_emptyInput():
    with pytest.raises(Exception) as e:
        sith = SITH(emptyPath, xFstring)
        sith.extract_data()
    assert str(e.value) == "Reference data file is empty."

    with pytest.raises(Exception) as e:
        sith = SITH(x0string, dePath=emptyPath)
        sith.extract_data()
    assert str(e.value) == "One or more deformed files are empty."

    with pytest.raises(Exception) as e:
        sith = SITH(x0string, dePath=emptyDir)
        sith.extract_data()
    assert str(e.value) == "Deformed directory is empty."


def test_get_contents():
    sith = SITH(frankensteinPath, frankensteinPath)
    sith._get_contents()
    assert sith._rData == frankenLines
    assert len(sith._dData) == 1
    assert sith._dData[0] == (
        Path(os.getcwd()+'/'+frankensteinPath), frankenLines)


def test_get_contentsDir():
    sith = SITH(frankensteinPath, frankensteinDir)
    sith._get_contents()
    assert sith._rData == frankenLines
    assert len(sith._dData) == 2
    assert sith._dData[0][1] == frankenLines
    assert sith._dData[1][1] == frankenLines


def test_extract_dataFile():
    sith = SITH(frankensteinPath, frankensteinPath)
    sith.extract_data()
    assert sith.reference == refGeo
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == refGeo
    assert not sith._kill


def test_extract_dataDir():
    sith = SITH(frankensteinPath, frankensteinDir)
    sith.extract_data()
    assert sith.reference == refGeo
    assert len(sith.deformed) == 2
    refGeoCopy.name = 'frankenstein-1'
    assert sith.deformed[0] == refGeoCopy
    refGeoCopy.name = 'frankenstein-2'
    assert sith.deformed[1] == refGeoCopy
    assert not sith._kill


def test_set_kill_dofs():
    sith = SITH(frankensteinPath, frankensteinPath)
    killDOFs = [(1, 2), (2, 1, 3)]
    sith.set_kill_dofs(killDOFs)
    assert sith._kill
    assert np.array_equal(sith._killDOFs, killDOFs)


def test_setKillAtoms():
    sith = SITH(frankensteinPath, frankensteinPath)
    killAtoms = [1, 6]
    sith.set_kill_dofs(killAtoms)
    assert sith._kill
    assert np.array_equal(sith._killDOFs, killAtoms)


def test_kill():
    # Doesn't directly test SITH.__kill() or SITH.__kill_dofs(dofs) because they are private but ensures that they work
    # properly in cases where they are called
    sith = SITH(frankensteinPath, frankensteinPath)
    killDOFs = [(1, 2)]
    sith.set_kill_dofs(killDOFs)
    sith.extract_data()
    killedGeo = deepcopy(refGeo)
    killedGeo._killDOFs([0])
    assert sith.reference == killedGeo
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == killedGeo


def test_killAtom():
    # Doesn't directly test SITH.__kill() or SITH.__kill_dofs(dofs) because they are private but ensures that they work
    # properly in cases where they are called
    sith = SITH(frankensteinPath, frankensteinPath)
    killAtoms = [2]
    sith.set_kill_atoms(killAtoms)
    sith.extract_data()
    killedGeo = deepcopy(refGeo)
    killedGeo._killDOFs([0, 5, 6, 7, 12])
    assert sith.reference == killedGeo
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == killedGeo


def test_kill_bad():
    sith = SITH(frankensteinPath, frankensteinPath)
    killDOFs = [(1, 19)]
    sith.set_kill_dofs(killDOFs)
    sith.extract_data()
    assert sith.reference == refGeo
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == refGeo


def test_killDOFsBAD():
    sith = SITH(x0string, deformedString)
    sith.set_kill_dofs([(1, 6), (2, 1, 5, 6)])
    sith.extract_data()


def test_killDOFsBAD2():
    sith = SITH(x0string, deformedString)
    sith.set_kill_dofs([(1, 6)])
    sith.extract_data()


def test_remove_extra_dofs_noKill():
    # no atoms specified for kill and no mismatch
    sith = SITH(frankensteinPath, frankensteinPath)
    sith._reference = refGeo
    sith._deformed = [refGeo]
    sith.remove_extra_dofs()
    assert sith.reference == refGeo
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == refGeo

    # no atoms specified for kill but should kill DOF (1, 16) from deformed cus not in reference
    sith = SITH('tests/glycine-ds-test/Gly-x0.fchk',
                'tests/glycine-ds-test/deformed/Gly-streched4.fchk')
    sith._get_contents()
    extractor = Extractor(sith._referencePath, sith._rData)
    extractor._extract()
    sith._reference = extractor.getGeometry()
    extractor = Extractor(sith._deformedPath, sith._dData[0][1])
    extractor._extract()
    sith._deformed = [extractor.getGeometry()]
    ref = deepcopy(sith.reference)
    defd = deepcopy(sith.deformed[0])
    sith.remove_extra_dofs()
    defd._killDOFs([4])
    assert sith.reference == ref
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == defd


def test_remove_extra_dofs_kill():
    # atoms specified for kill but should kill DOF (1, 16) from deformed cus not in reference
    sith = SITH('tests/glycine-ds-test/Gly-x0.fchk',
                'tests/glycine-ds-test/deformed/Gly-streched4.fchk')
    sith._get_contents()
    extractor = Extractor(sith._referencePath, sith._rData)
    extractor._extract()
    ref = extractor.getGeometry()
    extractor = Extractor(sith._deformedPath, sith._dData[0][1])
    extractor._extract()
    defd = extractor.getGeometry()
    sith.set_kill_dofs([(1, 2)])
    sith.extract_data()
    ref._killDOFs([0])
    defd._killDOFs([0, 4])
    assert sith.reference == ref
    assert len(sith.deformed) == 1
    assert sith.deformed[0] == defd


def test_validate_geometries():
    ditto = deepcopy(refGeo)
    ditto._killDOFs([1])
    sith = SITH(x0string, xFstring)
    sith._reference = refGeo
    sith._deformed = [ditto]
    with pytest.raises(Exception) as e:
        sith._validate_geometries()
    assert str(
        e.value) == "Incompatible number of atoms or dimensions amongst input files."

    ditto = deepcopy(refGeo)
    ditto.nAtoms -= 1
    sith = SITH(x0string, xFstring)
    sith._reference = refGeo
    sith._deformed = [ditto]
    with pytest.raises(Exception) as e:
        sith._validate_geometries()
    assert str(
        e.value) == "Incompatible number of atoms or dimensions amongst input files."


def test_populate_q():
    sith = SITH(frankensteinPath, frankensteinPath)
    sith._reference = refGeo
    sith._deformed = [refGeo]
    sith._populate_q()
    coords2D = np.array(coords)[np.newaxis]
    assert compare_arrays(sith.q0, np.transpose(coords2D))
    assert compare_arrays(sith.qF, np.transpose(coords2D))
    assert compare_arrays(sith.deltaQ, np.zeros((len(coords), 1)))

    sith = SITH(frankensteinPath, frankensteinPath)
    sith._reference = refGeo
    ditto = deepcopy(refGeo)
    ditto.ric = 0.5 * ditto.ric
    sith._deformed = [ditto]
    sith._populate_q()
    coords2D = np.array(coords)[np.newaxis]
    assert compare_arrays(sith.q0, np.transpose(coords2D))
    assert compare_arrays(sith.qF, np.transpose(coords2D)*0.5)
    assert compare_arrays(sith.deltaQ, -np.transpose(coords2D)*0.5)


def test_analyze():
    sith = SITH(x0string, xFstring)
    sith.extract_data()
    sith.deltaQ = dqSmall
    sith._reference.hessian = hessSmall
    sith._reference.dims = np.array([3, 1, 1, 1])
    sith.analyze()
    expectedDefE = np.transpose(dqSmall).dot(hessSmall).dot(dqSmall) / 2.
    assert compare_arrays(sith.deformationEnergy, expectedDefE)
    expectedEnergies = np.array([(dqSmall[i] * 0.5 * hessSmall.dot(dqSmall))[i]
                        for i in range(3)])
    assert compare_arrays(sith.energies, expectedEnergies)
    assert all(x == 100 for x in [sum(sith.pEnergies[:, i])
               for i in range(len(sith.deformed))])
    summation = np.array([sum(sith.energies[:,i]) for i in range(len(sith.deformed))])
    assert compare_arrays(sith.deformationEnergy, summation)


def test_analyze_Same():
    sith = SITH(frankensteinPath, frankensteinPath)
    with pytest.raises(Exception) as e:
        sith.analyze()
    assert str(e.value) == "Populate Q has not been executed so necessary data for analysis is lacking. This is likely due to not calling extract_data()."
    sith.extract_data()
    sith.analyze()
    assert compare_arrays(sith.energies, np.zeros((sith.reference.dims[0], 1)))
    assert sith.deformationEnergy[0, 0] == 0
    assert all([np.isnan(pE[0]) for pE in sith.pEnergies])
    summation = np.array([sum(sith.energies[:,i]) for i in range(len(sith.deformed))])
    assert compare_arrays(sith.deformationEnergy, summation)


# region Integration Tests and Examples

def test_full_killed():  # full with no mismatched, kill valid, deformed directory, just check doesn't crash
    sith = SITH(x0string, deformedString)
    sith.set_kill_dofs([(1, 2), (1, 3)])
    sith.extract_data()
    sith.analyze()
    assert sith.energies.shape == (13, 5)
    assert sith.pEnergies.shape == (13, 5)
    blah = [sum(sith.pEnergies[:, i]) ==
            approx(100) for i in range(len(sith.deformed))]
    assert all(blah)
    assert sith.deformationEnergy.shape == (1, 5)


def test_multiDeformedGood():  # full no mismatched to remove, deformed directory
    sith = SITH(x0string, deformedString)
    sith.extract_data()
    sith.analyze()
    assert sith.energies.shape == (15, 5)
    assert sith.pEnergies.shape == (15, 5)
    blah = [sum(sith.pEnergies[:, i]) == approx(100) for i in range(len(sith.deformed))]
    assert all(blah)
    assert sith.deformationEnergy.shape == (1, 5)
    summation = np.array([sum(sith.energies[:,i]) for i in range(len(sith.deformed))])
    assert compare_arrays(sith.deformationEnergy, summation)
    write_summary(sith, 'moh-')


def test_Glycine():  # full with mismatched to remove, deformed directory
    sith = SITH('tests/glycine-ds-test/Gly-x0.fchk',
                'tests/glycine-ds-test/deformed')
    sith.extract_data()
    sith.analyze()
    assert sith.energies.shape == (80, 10)
    assert sith.pEnergies.shape == (80, 10)
    blah = [sum(sith.pEnergies[:, i]) == approx(100) for i in range(len(sith.deformed))]
    assert all(blah)
    assert sith.deformationEnergy.shape == (1, 10)
    write_summary(sith, 'gly-')


def test_Alanine():  # full with valid kill invalid results
    sith = SITH('tests/Ala-stretched00.fchk',
                'tests/Ala-stretched10.fchk')
    sith.set_kill_dofs([(1, 19)])
    with pytest.raises(Exception) as e:
        sith.extract_data()
    assert str(
        e.value) == "Deformed geometry (Ala-stretched10) is missing reference DOF (4, 1, 5)."


def test_movedx0():
    sith = SITH('tests/moh-x0-1.7.fchk',
                'tests/deformed')
    sith.extract_data()
    sith.analyze()
    assert sith.energies.shape == (15, 5)
    assert sith.pEnergies.shape == (15, 5)
    blah = [sum(sith.pEnergies[:, i]) == approx(100) for i in range(len(sith.deformed))]
    assert all(blah)
    assert sith.deformationEnergy.shape == (1, 5)
    write_summary(sith, 'moh-x0-moved-')

def test_glyGoof():  # full with mismatched to remove, deformed directory
    local_sith = SITH('tests/local-ref-test/Gly-opt08.fchk',
                '../../../../../basement/mbm/sucerquia/first_aminoacids/g09/3-OptStreched')
    local_sith.extract_data()
    local_sith.analyze()
    write_all(local_sith, filePrefix='local-ref-gly-')

    global_sith = SITH('tests/global-ref-test/Gly-streched00.fchk',
                '../../../../../basement/mbm/sucerquia/first_aminoacids/g09/2-OptStreched02/Glycine/')
    global_sith.extract_data()
    global_sith.analyze()
    write_all(global_sith, filePrefix='global-ref-gly-')

    blah = 2


# endregion
