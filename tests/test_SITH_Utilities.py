from genericpath import exists
import pytest
from SITH_Utilities import *

""" LTMatrix has already been tested by its creator on github,
 but should add in their testing just in case """


def test_atomCreation():
    atom = Atom('C', [2.3, 4.6, 7.8])
    assert atom.element is 'C'

# region Geometry Tests


def test_geometryName():
    geo = Geometry('testName', 3)
    assert geo.name == 'testName'


def test_geometryNAtoms():
    geo = Geometry('testName', 3)
    assert geo.nAtoms == 3


def test_dumbGeoCreationErrors():
    geo = Geometry('blah', 6)
    assert geo.energy == np.inf


def test_getEnergyGood():
    geo = Geometry('blah', 6)
    geo.energy = 42
    assert geo.energy == 42
    assert geo.getEnergy() == 42


def test_getEnergyBad():
    geo = Geometry('blah', 6)
    with pytest.raises(Exception) as e_info:
        blah = geo.getEnergy()


dims = [15, 5, 7, 3]
dimIndicesGoodInput = ['           1           2           0           0           1           3',
                       '           0           0           1           4           0           0',
                       '           1           5           0           0           5           6',
                       '           0           0           2           1           3           0',
                       '           2           1           4           0           2           1',
                       '           5           0           3           1           4           0',
                       '           3           1           5           0           4           1',
                       '           5           0           1           5           6           0',
                       '           2           1           5           6           3           1',
                       '           5           6           4           1           5           6']
coordLinesGoodInput = ['  2.06335755E+00  2.07679249E+00  2.07679461E+00  2.73743812E+00  1.83354933E+00',
                       '  1.90516186E+00  1.90518195E+00  1.84167462E+00  1.91434964E+00  1.94775283E+00',
                       '  1.94775582E+00  1.96310537E+00 -3.14097002E+00 -1.07379153E+00  1.07501112E+00']
coords = [float(2.06335755E+00), 2.07679249, 2.07679461, 2.73743812, 1.83354933, 1.90516186E+00,
          1.90518195,  1.84167462,  1.91434964,  1.94775283, 1.94775582,  1.96310537, -3.14097002,
          -1.07379153,  1.07501112E+00]
bonds = [float(2.06335755E+00), 2.07679249, 2.07679461, 2.73743812, 1.83354933]
angles = [1.90516186E+00, 1.90518195,  1.84167462,
          1.91434964,  1.94775283, 1.94775582,  1.96310537]
diheds = [-3.14097002, -1.07379153,  1.07501112E+00]

dimIndices = [(1, 2), (1, 3), (1, 4), (1, 5), (5, 6), (2, 1, 3), (2, 1, 4), (2, 1, 5),
              (3, 1, 4), (3, 1, 5), (4, 1, 5), (1, 5, 6), (2, 1, 5, 6), (3, 1, 5, 6), (4, 1, 5, 6)]


def test_buildRICGood():
    geo = Geometry('methanol-test', 6)
    geo.buildRIC(dims, dimIndicesGoodInput, coordLinesGoodInput)
    assert geo.dims == dims
    assert geo.dimIndices == dimIndices
    assert geo.rawRIC == coords
    assert geo.lengths == bonds
    assert geo.angles == angles
    assert geo.diheds == diheds


# region bad coordinates

def test_moreCoords():
    coordsMore = coordLinesGoodInput + ['100.78943']
    geo = Geometry('methanol-test', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndicesGoodInput, coordsMore)
    assert str(e.value) == "Mismatch between the number of degrees of freedom expected (" + \
        str(dims[0])+") and number of coordinates given ("+str(dims[0]+1)+")."


def test_lessCoords():
    coordsLess = coordLinesGoodInput[1:]
    geo = Geometry('methanol-test', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndicesGoodInput, coordsLess)
    assert str(e.value) == "Mismatch between the number of degrees of freedom expected (" + \
        str(dims[0])+") and number of coordinates given ("+str(dims[0]-5)+")."


# endregion

dimIndices59 = ['           1           2           0           0           1           3',
                '           0           0           1           4           0           0',
                '           1           5           0           0           5           6',
                '           0           0           2           1           3           0',
                '           2           1           4           0           2           1',
                '           5           0           3           1           4           0',
                '           3           1           5           0           4           1',
                '           5           0           1           5           6           0',
                '           2           1           5           6           3           1',
                '           5           6           4           1           5']

dimIndicesLetters = ['           1           2           k           0           1           3',
                     '           0           0           1           4           0           0',
                     '           1           l           0           0           5           6',
                     '           0           0           2           1           3           0',
                     '           2           1           A           0           2           1',
                     '           5           0           t           1           4           0',
                     '           3           1           5           0           4           1',
                     '           5           0           1           5           6           0',
                     '           2           1           5           6           3           1',
                     '           5           6           4           1           5           6']

dimIndicesNumI = ['           1           2           0           0           1           3',
                  '           0           0           1           4           0           0',
                  '           1           5           0           0           5           6',
                  '           0           0           2           1           3           0',
                  '           2           1           4           0           2           1',
                  '           5           0           3           1           4           0',
                  '           3           1           5           0           4           1',
                  '           5           0           1           5           6           0',
                  '           2           1           5           6           3           1',
                  '           5           0           4           1           5           6']


def test_riciBad():
    geo = Geometry('methanol-test', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndices59, coordLinesGoodInput)
    assert str(
        e.value) == "Redundant internal coordinate indices input has invalid dimensions."

#! Modify to have a specific flag for this potentially


def test_riciLetters():
    geo = Geometry('methanol-test', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndicesLetters, coordLinesGoodInput)
    assert str(
        e.value) == "Invalid atom index given as input."


def test_riciNumIndices():
    geo = Geometry('methanol-test', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndicesNumI, coordLinesGoodInput)
    assert str(
        e.value) == "Mismatch between given 'RIC dimensions' and given RIC indices."


def test_riciInvalid():
    dimIndicesInvalid = [
        '           1           7           0           0           1           3']+dimIndicesGoodInput[1:]
    geo = Geometry('methanol-test', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndicesInvalid, coords)
    assert str(e.value) == "Invalid atom index given as input."


def test_buildRICBad():
    geo = Geometry('methanol-test', 6)
    with pytest.raises(Exception) as e:
        geo.buildRIC(dims, dimIndicesGoodInput[2:], coordLinesGoodInput)
    assert str(
        e.value) == "Redundant internal coordinate indices input has invalid dimensions."


# region Cartesian


def test_buildCartesian():
    pass


def test_getAtoms():
    pass

# endregion

# endregion

# region Extractor Tests


# region Cartesian
def test_writeXYZ():
    pass

# endregion


def test_extract():
    pass


def test_creation():
    pass


def test_buildHessian():
    pass


def test_getGeometry():
    pass


def test_getHessian():
    pass


# endregion
