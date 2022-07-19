from re import L
from JEDI import JEDI
from SITH_Utilities import UnitConverter
import numpy as np


class JediResults:

    def writeFiles(self, jedi: JEDI) -> bool:
        pass

    def buildEnergiesString(self, jedi: JEDI) -> list:
        pass

    def buildDeltaQString(self, jedi: JEDI) -> list:
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
        header = "DOF \t"
        for deformation in jedi.deformed:
            header += str(deformation.name)+"\t"
        dqAngstroms.append(header)
        dqAng = [uc.bohrToAngstrom(dq)
                 for dq in jedi.delta_q[0:jedi.relaxed.dims[1], :]]
        dqAng = np.asarray(dqAng)
        dqAng = dqAng.astype(str)
        for dof in range(jedi.relaxed.dims[1]):
            if len(jedi.deformed) > 1:
                line = str(dof+1) + "\t" + '\t'.join(dqAng[dof, :])
                dqAngstroms.append(line)
            else:
                line = str(dof+1) + "\t" + dqAng[dof][1:-2]
                dqAngstroms.append(line)
        dqDeg = np.degrees(jedi.delta_q[jedi.relaxed.dims[1]:, :])
        dqDeg = np.asarray(dqDeg)
        dqDeg = dqDeg.astype(str)
        for dof in range(jedi.relaxed.dims[2]+jedi.relaxed.dims[3]):
            if len(jedi.deformed) > 1:
                line = str(
                    dof+1+jedi.relaxed.dims[1]) + "\t" + '\t'.join(dqDeg[dof, :])
                dqAngstroms.append(line)
            else:
                line = str(
                    dof+1+jedi.relaxed.dims[1]) + "\t" + str(dqDeg[dof][0])
                dqAngstroms.append(line)

        return dqAngstroms

    def writeDeltaQ(self, jedi: JEDI) -> bool:
        dqPrint = self.buildDeltaQString(jedi)
        with open('delta_q.txt', "w") as dq:
            dq.writelines('\n'.join(dqPrint))

    def buildInternalCoordsString(self, jedi: JEDI) -> list:
        pass

    def buildEnergyMatrix(self, jedi: JEDI) -> list:
        """
        Returns a list of strings containing the energy in each degree of freedom per deformed geometry.
        Data is of the format:
        DOF Index       Deformation 1       Deformation 2       ...
        1               stress E            stress E            ...
        2               stress E            stress E            ...
        ...
        """
        uc = UnitConverter()
        dqAngstroms = list()
        header = "DOF \t"
        for deformation in jedi.deformed:
            header += str(deformation.name)+"\t"
        dqAngstroms.append(header)
        dqAng = [uc.bohrToAngstrom(dq)
                 for dq in jedi.delta_q[0:jedi.relaxed.dims[1], :]]
        dqAng = np.asarray(dqAng)
        dqAng = dqAng.astype(str)
        for dof in range(jedi.relaxed.dims[1]):
            if len(jedi.deformed) > 1:
                line = str(dof+1) + "\t" + '\t'.join(dqAng[dof, :])
                dqAngstroms.append(line)
            else:
                line = str(dof+1) + "\t" + dqAng[dof][1:-2]
                dqAngstroms.append(line)
        dqDeg = np.degrees(jedi.delta_q[jedi.relaxed.dims[1]:, :])
        dqDeg = np.asarray(dqDeg)
        dqDeg = dqDeg.astype(str)
        for dof in range(jedi.relaxed.dims[2]+jedi.relaxed.dims[3]):
            if len(jedi.deformed) > 1:
                line = str(
                    dof+1+jedi.relaxed.dims[1]) + "\t" + '\t'.join(dqDeg[dof, :])
                dqAngstroms.append(line)
            else:
                line = str(
                    dof+1+jedi.relaxed.dims[1]) + "\t" + str(dqDeg[dof][0])
                dqAngstroms.append(line)

        return dqAngstroms

    def compareEnergies(self, jedi: JEDI):
        self.expectedDE = np.zeros((1,len(jedi.deformed)))
        for i in range(len(jedi.deformed)):
            self.expectedDE[0,i] = jedi.deformed[i].energy - jedi.relaxed.energy
        self.errorDE = jedi.deformationEnergy - self.expectedDE
        self.pErrorDE = self.errorDE / self.expectedDE
