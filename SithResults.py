from typing import Tuple
from SITH import SITH
from SITH_Utilities import UnitConverter
import numpy as np


#TODO: sandbox the file access with try catch dum dum

class SithResults:
    """Contains methods which format, organize, and output data from SITH objects"""

    def writeFiles(self, sith: SITH) -> bool:
        pass

    @staticmethod
    def writeSummary(sith: SITH):
        totE = SithResults.buildTotEnergiesString(sith)
        dq = SithResults.buildDeltaQString(sith)
        ric = SithResults.buildInternalCoordsString(sith)
        expectedDE, errorDE, pErrorDE = SithResults.compareEnergies(sith)
        energies = SithResults.buildEnergyMatrix(sith)

        with open(sith._relaxedPath.parent.as_posix()+sith._relaxedPath.root+'summary.txt', "w") as s:
            s.write("Summary of SITH analysis\n")
            s.write(
                "Redundant Internal Coordinate Definitions\n**Defined by indices of involved atoms**\n")
            s.writelines('\n'.join(ric))
            s.write(
                "\nChanges in internal coordinates (Delta q)\n**Distances given in Angstroms, angles given in degrees**\n")
            s.writelines('\n'.join(dq))
            s.write(
                "\n\n***********************\n**  Energy Analysis  **\n***********************\n")
            s.write("Overall Structural Energies\n")
            s.write(
                "Deformation        \u0394E          \u0025Error          Error\n")
            for i in range(len(sith._deformed)):
                s.write("{: <12s}{: >16.6E}{: >12.2%}{: >16.6E}\n".format(
                    sith._deformed[i].name, sith.deformationEnergy[0, i], pErrorDE[0, i], errorDE[0, i]))

            s.write("Energy per DOF (RIC)\n")
            s.writelines("\n".join(energies))

    @staticmethod
    def buildTotEnergiesString(sith: SITH) -> list:
        pass

    @staticmethod
    def buildDeltaQString(sith: SITH) -> list:
        """
        Returns a list of strings containing the change in internal coordinates in each degree of freedom 
        per deformed geometry. Data is in Angstroms and degrees of the format:
        DOF Index       Deformation 1       Deformation 2       ...
        1               change              change              ...
        2               change              change              ...
        ...
        """
        uc = UnitConverter()
        dqAngstroms = list()
        header = "DOF         "
        for deformation in sith._deformed:
            header += "{: ^12s}".format(deformation.name)
        dqAngstroms.append(header)
        dqAng = [uc.bohrToAngstrom(dq)
                 for dq in sith.deltaQ[0:sith._relaxed.dims[1], :]]
        dqAng = np.asarray(dqAng)
        for dof in range(sith._relaxed.dims[1]):
            if len(sith._deformed) > 1:
                line = "{: <12}".format(
                    dof+1) + ''.join(["{: >12.8f}".format(dqa) for dqa in dqAng[dof, :]])
                dqAngstroms.append(line)
            else:
                line = "{: <12}{: >12.8f}".format(dof+1, dqAng[dof][0])
                dqAngstroms.append(line)
        dqDeg = np.degrees(sith.deltaQ[sith._relaxed.dims[1]:, :])
        dqDeg = np.asarray(dqDeg)
        for dof in range(sith._relaxed.dims[2]+sith._relaxed.dims[3]):
            if len(sith._deformed) > 1:
                line = "{:< 12}".format(dof+1+sith._relaxed.dims[1]) + ''.join(
                    ["{: >12.8f}".format(dqd) for dqd in dqDeg[dof, :]])
                dqAngstroms.append(line)
            else:
                line = "{:< 12}{: >12.8f}".format(
                    dof+1+sith._relaxed.dims[1], dqDeg[dof][0])
                dqAngstroms.append(line)

        return dqAngstroms

    @staticmethod
    def writeDeltaQ(sith: SITH) -> bool:
        dqPrint = SithResults.buildDeltaQString(sith)
        with open('delta_q.txt', "w") as dq:
            dq.writelines('\n'.join(dqPrint))

    @staticmethod
    def buildInternalCoordsString(sith: SITH) -> list:
        """
        Returns a list of strings containing the atom indices involved in each degree of freedom.
        """
        return ["{: <12}".format(dof+1) + str(sith._relaxed.dimIndices[dof]) for dof in range(sith._relaxed.dims[0])]

    @staticmethod
    def buildEnergyMatrix(sith: SITH) -> list:
        """
        Returns a list of strings containing the energy in each degree of freedom per deformed geometry.
        Data is in Hartrees and of the format:
        DOF Index       Deformation 1       Deformation 2       ...
        1               stress E            stress E            ...
        2               stress E            stress E            ...
        ...             ...                 ...                 ...
        """
        uc = UnitConverter()
        eMat = list()
        header = "DOF         "
        for deformation in sith._deformed:
            header += "{: ^16s}".format(deformation.name)
        eMat.append(header)
        for dof in range(sith._relaxed.dims[0]):
            line = "{: <12}".format(
                dof+1) + ''.join(["{: >16.6E}".format(e) for e in sith.energies[dof, :]])
            eMat.append(line)
        return eMat

    @staticmethod
    def writeEnergyMatrix(sith: SITH) -> bool:
        ePrint = SithResults.buildEnergyMatrix(sith)
        with open('E_RICS.txt', "w") as dq:
            dq.writelines('\n'.join(ePrint))

    @staticmethod
    def compareEnergies(sith: SITH) -> Tuple:
        expectedDE = np.zeros((1, len(sith._deformed)))
        for i in range(len(sith._deformed)):
            expectedDE[0, i] = sith._deformed[i].energy - sith._relaxed.energy
        errorDE = sith.deformationEnergy - expectedDE
        pErrorDE = errorDE / expectedDE
        return (expectedDE, errorDE, pErrorDE)

    @staticmethod
    def writeComparison(sith: SITH):
        expectedDE, errorDE, pErrorDE = SithResults.compareEnergies(sith)
        with open('pError.txt', "w") as dq:
            dq.writelines('\n'.join(pErrorDE.astype(str)))

    @staticmethod
    def buildAtomList(sith: SITH):
        """Builds strings for indicating the atom represented by each index."""
        pass

    @staticmethod
    def writeAtomList(sith: SITH):
        pass
