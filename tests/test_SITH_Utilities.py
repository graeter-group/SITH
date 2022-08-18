from numpy import float32
import pytest

from src.SITH.Utilities import *
from src.SITH.SITH import SITH
from tests.test_variables import *

""" LTMatrix has already been tested by its creator on github,
 but should add in their testing just in case """


def test_atomCreation():
    atom = Atom('C', [2.3, 4.6, 7.8])
    assert atom.element is 'C'
    assert atom.coords == [2.3, 4.6, 7.8]

# region Geometry Tests


def test_geometry():
    geo = Geometry('testName', 'blah', 3)
    assert geo.name == 'testName'
    assert geo._path == 'blah'
    assert geo.nAtoms == 3
    assert geo.energy == None


def test_geo_energy():
    geo = Geometry('blah', 'blah', 6)
    geo.energy = 42
    assert geo.name == 'blah'
    assert geo.nAtoms == 6
    assert geo.energy == 42


def test_buildRICGood():
    geo = Geometry('methanol-test', 'blah', 6)
    geo.buildRIC(dims, dimIndicesGoodInput, coordLinesGoodInput)
    assert geo.dims == dims
    assert geo.dimIndices == dimIndices
    assert all([geo.ric[i] == coords[i] for i in range(len(coords))])


def test_equals():
    geoCopy = deepcopy(refGeo)
    assert geoCopy == refGeo
    assert refGeo != Geometry('test', 'test', 6)
    geo = Geometry('methanol-test', 'blah', 6)
    geo.buildRIC(dims, dimIndicesGoodInput, coordLinesGoodInput)
    assert refGeo != geo
    geoCopy.name = 'blah'
    assert geoCopy != refGeo
    geoCopy.name = refGeo.name
    geoCopy.hessian = geoCopy.hessian[1:]
    assert geoCopy != refGeo
    geoCopy.hessian = refGeo.hessian
    geoCopy.atoms[3] = Atom('C', (1., 1., 1.))
    assert geoCopy != refGeo
    geoCopy.atoms = refGeo.atoms
    assert geoCopy == refGeo
    geoCopy.energy = 0
    assert geoCopy != refGeo
    geoCopy.energy = refGeo.energy
    geoCopy.ric[2] = 26
    assert geoCopy != refGeo
    geoCopy.ric = refGeo.ric
    geoCopy.dimIndices[2] = (1, 2)
    assert geoCopy != refGeo


# region bad coordinates


def test_letterCoord():
    letterCoord = coordLinesGoodInput + ['blah']
    geo = Geometry('methanol-test', 'blah', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndicesGoodInput, letterCoord)
    assert str(
        e.value) == "Redundant internal coordinates contains invalid values, such as strings."


def test_moreCoords():
    coordsMore = coordLinesGoodInput + ['100.78943']
    geo = Geometry('methanol-test', 'blah', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndicesGoodInput, coordsMore)
    assert str(e.value) == "Mismatch between the number of degrees of freedom expected (" + \
        str(dims[0])+") and number of coordinates given ("+str(dims[0]+1)+")."


def test_lessCoords():
    coordsLess = coordLinesGoodInput[1:]
    geo = Geometry('methanol-test', 'blah', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndicesGoodInput, coordsLess)
    assert str(e.value) == "Mismatch between the number of degrees of freedom expected (" + \
        str(dims[0])+") and number of coordinates given ("+str(dims[0]-5)+")."


# endregion

# region bad Indices


def test_riciBad():
    geo = Geometry('methanol-test', 'blah', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndices59, coordLinesGoodInput)
    assert str(
        e.value) == "One or more redundant internal coordinate indices are missing or do not have the expected format. Please refer to documentation"
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndices59[1:], coordLinesGoodInput)
    assert str(
        e.value) == "One or more redundant internal coordinate indices are missing or do not have the expected format. Please refer to documentation"


def test_riciLetters():
    geo = Geometry('methanol-test', 'blah', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndicesLetters, coordLinesGoodInput)
    assert str(
        e.value) == "Invalid atom index given as input."


def test_riciNumIndices():
    geo = Geometry('methanol-test', 'blah', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndicesNumI, coordLinesGoodInput)
    assert str(
        e.value) == "Mismatch between given 'RIC dimensions' and given RIC indices."


def test_riciInvalid():
    dimIndicesInvalid = [
        '           1           7           0           0           1           3']+dimIndicesGoodInput[1:]
    geo = Geometry('methanol-test', 'blah', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndicesInvalid, coords)
    assert str(e.value) == "Invalid atom index given as input."


def test_buildRICIBad():
    geo = Geometry('methanol-test', 'blah', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndicesGoodInput[2:], coordLinesGoodInput)
    assert str(
        e.value) == "One or more redundant internal coordinate indices are missing or do not have the expected format. Please refer to documentation"

# endregion


def test_buildRIC_badDims():
    geo = Geometry('methanol-test', 'blah', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC([16, 5, 7, 3], dimIndicesGoodInput, coordLinesGoodInput)
    assert str(
        e.value) == "Invalid quantities of dimension types (bond lengths, angles, dihedrals) given in .fchk."
    with pytest.raises(Exception) as e:
        geo.buildRIC([16, 'h', 7, 3], dimIndicesGoodInput, coordLinesGoodInput)
    assert str(
        e.value) == "Invalid input given for Redundant internal dimensions."

# region Cartesian


def test_buildCartesian():
    geo = Geometry('methanol-test', 'blah', 6)
    geo.buildCartesian(cartesianLines)
    assert geo.nAtoms == 6 == len(geo.atoms)
    assert all([geo.atoms[i] == refAtoms[i] for i in range(len(refAtoms))])


def test_buildCartesian_integrated():
    assert refGeo.nAtoms == 6 == len(refGeo.atoms)
    assert all([refGeo.atoms[i] == refAtoms[i] for i in range(len(refAtoms))])


# endregion


def test_killDOF():
    sith = SITH(x0string, deformedString)
    sith.extractData()
    a = sith._reference.hessian == eHessFull
    assert (sith._reference.hessian == eHessFull).all()
    sith._reference._killDOFs([0])
    b = sith._reference.dimIndices == dimIndices[1:]
    assert all(sith._reference.dimIndices == dimIndices[1:])
    c = sith._reference.dims == array('i', [14, 4, 7, 3])
    assert sith._reference.dims == array('i', [14, 4, 7, 3])
    d = sith._reference.hessian == eHessKill0
    assert (sith._reference.hessian == eHessKill0).all()


def test_killDOFs():
    sith = SITH(x0string, deformedString)
    sith.extractData()
    a = sith._reference.hessian == eHessFull
    assert (sith._reference.hessian == eHessFull).all()
    sith._reference._killDOFs([0, 14])
    b = sith._reference.dimIndices == dimIndices[1:14]
    assert all(sith._reference.dimIndices == dimIndices[1:14])
    c = sith._reference.dims == array('i', [13, 4, 7, 2])
    assert sith._reference.dims == array('i', [13, 4, 7, 2])
    d = sith._reference.hessian == eHessKill0_14
    assert (sith._reference.hessian == eHessKill0_14).all()

# endregion


# region Extractor Tests


def test_creationEmptyList():
    extractor = Extractor(testPath, [])
    assert extractor._path == testPath
    assert extractor._name == testPath.stem


def test_extract():
    extractor = Extractor(testPath, frankenNoLines)
    extractor._extract()
    assert extractor.hRaw == ehRaw


def test_extractedGeometry():
    extractor = Extractor(Path(
        '/hits/fast/mbm/farrugma/sw/SITH/tests/frankenTest-methanol.fchk'), frankenNoLines)
    extractor._extract()
    geo = extractor.getGeometry()
    egeo = Geometry('frankenTest-methanol', 'blah', 6)
    egeo.energy = energy
    egeo.buildRIC(dims, dimIndicesGoodInput, coordLinesGoodInput)
    egeo.buildCartesian(cartesianLines)
    egeo.hessian = eHessFull
    assert geo == egeo


def test_buildHessian():
    extractor = Extractor(testPath, frankenNoLines)
    extractor._extract()
    geo = extractor.getGeometry()
    egeo = Geometry(testPath.stem, 'blah', 6)
    egeo.energy = energy
    egeo.buildRIC(dims, dimIndicesGoodInput, coordLinesGoodInput)
    egeo.buildCartesian(cartesianLines)
    hess = extractor.hessian

    assert (eHessFull == hess).all()


def test_getGeometry():
    extractor = Extractor(testPath, frankenNoLines)
    egeo = Geometry('testName', 'blah', 3)
    with pytest.raises(Exception) as e:
        geo = extractor.getGeometry()
    assert str(e.value) == "There is no geometry."
    extractor.geometry = Geometry('testName', 'blah', 3)
    geo = extractor.getGeometry()
    assert geo == egeo


# endregion

def test_units():
    assert UnitConverter.angstromToBohr(1.3) == float32(2.456644)
    assert UnitConverter.bohrToAngstrom(1.3) == float32(0.68793035)
    assert UnitConverter.radianToDegree(1.3) == float32(74.48451)
