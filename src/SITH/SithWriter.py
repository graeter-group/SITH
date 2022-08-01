from typing import Tuple

import numpy as np
import openbabel as ob

from src.SITH import SITH
from src.SITH.Utilities import Geometry, UnitConverter


class SithWriter:
    """Contains methods which format, organize, and output data from SITH objects"""

#region: Write

    @staticmethod
    def writeAll(sith: SITH) -> bool:
        return SithWriter.writeSummary(sith) and SithWriter.writeDeltaQ(sith) and SithWriter.writeEnergyMatrix(sith) and SithWriter.writeError(sith)

    @staticmethod
    def writeSummary(sith: SITH, fileName='summary.txt') -> bool:
        """
        Takes in SITH object sith, Writes summary.txt file of sith data, Returns True if successful

        -----
        File is written to sith input's parent directory.
        """

        totE = SithWriter.buildTotEnergiesString(sith)
        dq = SithWriter.buildDeltaQString(sith)
        ric = SithWriter.buildInternalCoordsString(sith)
        expectedDE, errorDE, pErrorDE = SithWriter.compareEnergies(sith)
        energies = SithWriter.buildEnergyMatrix(sith)

        try:
            with open(sith._referencePath.parent.as_posix()+sith._referencePath.root+sith._referencePath.stem+fileName, "w") as s:
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
                return True
        except IOError as e:
            print(e)
            return False
        except e:
            print("Non-IO Exception encountered:")
            print(e)
            return False

    @staticmethod
    def writeTotEnergiesString(sith: SITH, fileName="totalStressEnergy.txt") -> True:
        """
        Takes in SITH object sith, Writes the change in RICs per structure, Returns true if successful.

        -----
        Data is in Angstroms and Radians and of the format row:DOF column:deformation structure.
        """
        try:
            lines = SithWriter.buildTotEnergiesString(sith)
            with open(sith._referencePath.parent.as_posix()+sith._referencePath.root+fileName, "w") as dq:
                dq.writelines('\n'.join(lines))
            return True
        except IOError as e:
            print(e)
            return False
        except e:
            print("Non-IO Exception encountered:")
            print(e)
            return False

    @staticmethod
    def writeDeltaQ(sith: SITH, fileName="delta_q.txt") -> bool:
        """
        Takes in SITH object sith, Writes the change in RICs per structure, Returns true if successful.

        -----
        Data is in Angstroms and Radians and of the format row:DOF column:deformation structure.
        """
        try:
            dqPrint = SithWriter.buildDeltaQString(sith)
            with open(sith._referencePath.parent.as_posix()+sith._referencePath.root+fileName, "w") as dq:
                dq.writelines('\n'.join(dqPrint))
            return True
        except IOError as e:
            print(e)
            return False
        except e:
            print("Non-IO Exception encountered:")
            print(e)
            return False

    @staticmethod
    def writeError(sith: SITH, fileName='Error.txt') -> bool:
        """Takes in SITH object sith, Writes error data, Returns True if successful

        -----
        Writes to .txt file in directory of sith input files' parent"""
        try:
            lines = SithWriter.buildErrorStrings(sith)
            with open(fileName, "w") as dq:
                dq.writelines('\n'.join(lines))
            return True
        except IOError as e:
            print(e)
            return False
        except e:
            print("Non-IO Exception encountered:")
            print(e)
            return False

    @staticmethod
    def writeEnergyMatrix(sith: SITH, fileName='E_RICs.txt') -> bool:
        """
        Takes in SITH object sith, Writes the energy in each degree of freedom per deformed geometry, Returns True if successful

        -----
        Data is in Hartrees and of the format row:DOF column:deformation structure.
        """
        try:
            ePrint = SithWriter.buildEnergyMatrix(sith)
            with open(sith._referencePath.parent.as_posix()+sith._referencePath.root+fileName, "w") as dq:
                dq.writelines('\n'.join(ePrint))
            return True
        except IOError as e:
            print(e)
            return False
        except e:
            print("Non-IO Exception encountered:")
            print(e)
            return False

    @staticmethod
    def writeXYZ(geometry: Geometry):
        """
        Writes a .xyz file of the geometry using OpenBabel and the initial SITH input .fchk file
        """
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("fchk", "xyz")

        mol = ob.OBMol()
        assert geometry._path.exists(), "Path to fchk file does not exist"
        assert obConversion.ReadFile(mol, geometry._path.as_posix(
        )), "Reading fchk file with openbabel failed."
        assert obConversion.WriteFile(mol, str(
            geometry._path.parent.as_posix()+geometry._path.root+geometry._path.stem+".xyz")), "Could not write XYZ file."

# endregion

#region: Build

    @staticmethod
    def buildTotEnergiesString(sith: SITH) -> list:
        """
        Takes in SITH object sith, Returns a list of strings containing error informationper deformed geometry.
        Data is in Hartrees and percentages.
        """
        assert sith.deformationEnergy is not None, "SITH.energyAnalysis() has not been performed yet, no information available."

        lines = list()
        header = "            "
        for deformation in sith._deformed:
            header += "{: ^16s}".format(deformation.name)
        lines.append(header)
        lines.append("Stress Energy   " +
                     ''.join(["{: >16.6E}".format(e) for e in sith.deformationEnergy[0]]))
        return lines

    @staticmethod
    def buildDeltaQString(sith: SITH) -> list:
        """
        Takes in SITH object sith, Returns a list of strings containing the change in internal coordinates in each degree of freedom 
        per deformed geometry. Data is in Angstroms and degrees of the format row:DOF column: deformation
        """
        """
        DOF Index       Deformation 1       Deformation 2       ...
        1               change              change              ...
        2               change              change              ...
        ...
        """
        assert sith.deltaQ is not None, "SITH.extractData() has not been performed yet, no deltaQ information available."

        uc = UnitConverter()
        dqAngstroms = list()
        header = "DOF         "
        for deformation in sith._deformed:
            header += "{: ^12s}".format(deformation.name)
        dqAngstroms.append(header)
        dqAng = [uc.bohrToAngstrom(dq)
                 for dq in sith.deltaQ[0:sith._reference.dims[1], :]]
        dqAng = np.asarray(dqAng)
        for dof in range(sith._reference.dims[1]):
            if len(sith._deformed) > 1:
                line = "{: <12}".format(
                    dof+1) + ''.join(["{: >12.8f}".format(dqa) for dqa in dqAng[dof, :]])
                dqAngstroms.append(line)
            else:
                line = "{: <12}{: >12.8f}".format(dof+1, dqAng[dof][0])
                dqAngstroms.append(line)
        dqDeg = np.degrees(sith.deltaQ[sith._reference.dims[1]:, :])
        dqDeg = np.asarray(dqDeg)
        for dof in range(sith._reference.dims[2]+sith._reference.dims[3]):
            if len(sith._deformed) > 1:
                line = "{:< 12}".format(dof+1+sith._reference.dims[1]) + ''.join(
                    ["{: >12.8f}".format(dqd) for dqd in dqDeg[dof, :]])
                dqAngstroms.append(line)
            else:
                line = "{:< 12}{: >12.8f}".format(
                    dof+1+sith._reference.dims[1], dqDeg[dof][0])
                dqAngstroms.append(line)

        return dqAngstroms

    @staticmethod
    def buildInternalCoordsString(sith: SITH) -> list:
        """
        Takes in SITH object sith, Returns a list of strings containing the atom indices involved in each degree of freedom.
        """
        assert sith._reference.dimIndices is not None, "SITH.extractData() has not been performed yet, no summary information available."
        return ["{: <12}".format(dof+1) + str(sith._reference.dimIndices[dof]) for dof in range(sith._reference.dims[0])]

    @staticmethod
    def buildEnergyMatrix(sith: SITH) -> list:
        """
        Takes in SITH object sith, Returns a list of strings containing the energy in each degree of freedom
        per deformed geometry. Data is in Hartrees and of the format row:DOF column:deformation
        """
        """
        DOF Index       Deformation 1       Deformation 2       ...
        1               stress E            stress E            ...
        2               stress E            stress E            ...
        ...             ...                 ...                 ...
        """
        assert sith.energies is not None, "SITH.energyAnalysis() has not been performed yet, no summary information available."

        uc = UnitConverter()
        eMat = list()
        header = "DOF         "
        for deformation in sith._deformed:
            header += "{: ^16s}".format(deformation.name)
        eMat.append(header)
        for dof in range(sith._reference.dims[0]):
            line = "{: <12}".format(
                dof+1) + ''.join(["{: >16.6E}".format(e) for e in sith.energies[dof, :]])
            eMat.append(line)
        return eMat

    @staticmethod
    def buildErrorStrings(sith: SITH):
        """
        Takes in SITH object sith, Returns a list of strings containing error informationper deformed geometry.
        Data is in Hartrees and percentages.
        """
        expected, error, pError = SithWriter.compareEnergies(sith)
        lines = list()
        header = "            "
        for deformation in sith._deformed:
            header += "{: ^16s}".format(deformation.name)
        lines.append(header)
        lines.append("{: <12}".format('Expected Energy') +
                     ''.join(["{: >16.6E}".format(e) for e in expected[0]]))
        lines.append("{: <12}".format('Signed Error') +
                     ''.join(["{: >16.6E}".format(e) for e in error[0]]))
        lines.append("{: <12}".format("\u0025Error") +
                     ''.join(["{: >16.2%}".format(e) for e in pError[0]]))
        return lines

# endregion

    @staticmethod
    def compareEnergies(sith: SITH) -> Tuple:
        """
        Takes in SITH object sith, Returns Tuple of expected stress energy, stress energy error, and %Error

        -----
        Expected Stress Energy: Total E deformed structure from input .fchk - total E reference structure from input .fchk
        Stress Energy Error: calculated stress energy - Expected Stress Energy
        %Error: Stress Energy Error / Expected Stress Energy"""
        assert sith.deformationEnergy is not None, "SITH.energyAnalysis() has not been performed yet, no summary information available."
        expectedDE = np.zeros((1, len(sith._deformed)))
        for i in range(len(sith._deformed)):
            expectedDE[0, i] = sith._deformed[i].energy - \
                sith._reference.energy
        errorDE = sith.deformationEnergy - expectedDE
        pErrorDE = errorDE / expectedDE
        return (expectedDE, errorDE, pErrorDE)
