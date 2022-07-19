from genericpath import exists
from numpy import extract
import pytest
from SITH_Utilities import *
from JEDI import JEDI
import pathlib
from JediResults import JediResults

def test_buildDQ():
    jedi = JEDI()
    jedi.energyAnalysis()
    jp = JediResults()
    blah = jp.buildDeltaQString(jedi)
    print(blah)

def test_buildDQ2():
    jedi = JEDI('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk', '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    jedi.energyAnalysis()
    jp = JediResults()
    blah = jp.buildDeltaQString(jedi)
    print(blah)

def test_compareEnergies():
    jedi = JEDI('/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk', '/hits/fast/mbm/farrugma/sw/SITH/tests/deformed')
    jedi.energyAnalysis()
    jp = JediResults()
    blah = jp.buildDeltaQString(jedi)
    jp.writeDeltaQ(jedi)
    jp.compareEnergies(jedi)
