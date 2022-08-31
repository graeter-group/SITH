from src.SITH.Utilities import *
from src.SITH.SITH import SITH
from src.SITH.SithWriter import *
from tests.test_resources import *

"""There aren't necessarily very many options for testing here which are not unnecessarily detailed so 
the testing of SithWriter is kept to a minimum of simply 'it works' and manually checking the output files. 
This can be expanded upon later if necessary."""


def test_buildDQ():
    sith = SITH(x0string, xFstring)
    sith.extract_data()
    sith.analyze()

    blah = build_delta_q(sith)


def test_buildDQ2():
    sith = SITH(x0string, deformedString)
    sith.extract_data()
    sith.analyze()

    blah = build_delta_q(sith)


def test_compareEnergies():
    sith = SITH(x0string, deformedString)
    sith.extract_data()
    sith.analyze()

    assert write_delta_q(sith)
    assert write_error(sith)
    assert write_dof_energies(sith)
    assert write_summary(sith)


def test_writeAll():
    sith = SITH(x0string, deformedString)
    sith.extract_data()
    sith.analyze()

    assert write_all(sith)


def test_writeXYZ():
    sith = SITH(x0string, deformedString)
    sith.extract_data()
    sith.analyze()

    write_xyz(sith.reference)


def test_writeSummaryXYZ():
    sith = SITH(x0string, deformedString)
    sith.extract_data()
    sith.analyze()

    assert write_summary(sith, "xyz", includeXYZ=True)
