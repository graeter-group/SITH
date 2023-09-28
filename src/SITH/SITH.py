"""Calculates & houses SITH analysis data."""
import sys
from typing import Tuple
import pathlib
from pathlib import Path
import numpy as np
from SITH.Utilities import Extractor, Geometry


class SITH:
    """A class to calculate & house SITH analysis data."""
    """Attributes:
        q0 (numpy.ndarray):
            RIC values of reference geometry [DOF index, deformation index]
        qF (numpy.ndarray[float]):
            RIC value matrix  of deformed geometries [DOF index,
            deformation index]
        deltaQ (numpy.ndarray[float]):
            RIC value matrix  of deformed -reference
            [DOF index, deformation index]
        hessian (numpy.ndarray[float]):
            Hessian matrix of reference geometry
        energies (numpy.ndarray[float]):
            stress energy per DOF per deformation
            [DOF index, deformation index]
        pEnergies (numpy.ndarray[float]):
            Proportion of total stress energy [DOF index, deformation index]
        deformationEnergy (numpy.ndarray[float]):
            Total stress energy per deformation [deformation index]
    """

    def __init__(self, rePath='', dePath=''):
        """Initializes a SITH object

        Args:
            rePath (str, optional):
                reference geometry .fchk file path. Defaults to ''.
            dePath (str, optional):
                deformed geometry .fchk file path or path to directory
                of deformed geometries .fchk files. Defaults to ''.
        """
        self._workingPath = Path.cwd()

        self._referencePath = None
        """Path to reference geometry, specified on SITH construction"""

        self._deformedPath = None
        """Path to deformed geometry or directory of deformed geometries,
        specified on SITH construction"""

        if (rePath == ''):
            self._referencePath = self._workingPath / 'x0.fchk'
        else:
            self._referencePath = self._workingPath / rePath
        if (dePath == ''):
            self._deformedPath = self._workingPath / 'xF.fchk'
        else:
            self._deformedPath = self._workingPath / dePath

        # region variable documentation

        self.energies = None
        """numpy.ndarray[float]: Stress energy associated with a DOF for each
        deformation

        Organized in the form of [DOF index, deformation index].

        Note:
            While this may be publicly accessed, please refrain from manually
            setting this value as it is an analysis result."""
        self.deformationEnergy = None
        """np.ndarray[float]: Total stress energy associated with going from
        the reference structure to the ith deformed structure.

        Note:
            While this may be publicly accessed, please refrain from manually
            setting this value as it is an analysis result."""
        self.pEnergies = None
        """np.ndarray[float]: Percentage of stress energy in each degree of
        freedom to the each structure's total stress energy in the form
        [DOF index, deformation index]

        Note:
            While this may be publicly accessed, please refrain from manually
            setting this value as it is an analysis result."""

        self._reference = None
        """SITH.Utilities.Geometry: Reference Geometry object

        Note:
            Please refrain from manually setting this value as other analysis
            variables depend upon it.

        If you would like to manually change this value for a SITH object,
        instead implement the method set_reference() and the associated
        refactoring recommended in its documentation."""

        self._deformed = None
        """list[SITH.Utilities.Geometry]: List of deformed Geometry objects

        Note:
            While this may be publicly accessed, please refrain from manually
            setting this value as other analysis variables depend upon it."""

        # qF columns correspond to each deformed geometry, the rows correspond
        # to the degrees of freedom
        # qF[DOF index, row or deformed geometry index] -> value of DOF for
        # deformed geometry at index of self.deformed
        self.q0 = None
        """np.ndarray[float]: Vector of RIC values of reference geometry

        q[DOF index, 0] -> value of DOF for reference geometry

        Note:
            Publicly accessable for retrieval and reference, please refrain
            from manually setting this value as other analysis variables
            depend upon it."""

        self.qF = None
        """np.ndarray[float]: RIC values of deformed geometries

        qF[DOF index, row or deformed geometry index] -> value of DOF
        for deformed geometry at index of self.deformed

        Note:
            Publicly accessable for retrieval and reference, please refrain
            from manually setting this value as other analysis variables
            depend upon it."""

        self.deltaQ = None
        """np.ndarray[float]: Changes in RIC values from reference geometry to
        each deformed geometry.

        deltaQ[DOF index, row or deformed geometry index] -> value of DOF
        for deformed geometry at index of self.deformed

        Note:
            Publicly accessable for retrieval and reference, please
            refrain from manually setting this value as other analysis
            variables depend upon it and it is a result of calculations in
            _populate_q()."""

        self._kill = False
        """bool: Sentinel value indicating if any atoms of DOFs present in the
        input files should be removed from geometries, hessian, and analysis.
        """
        self._killAtoms = list()
        """list[int]: atom indices of atoms to be removed from reference
        geometry, hessian, and analysis"""
        self._killDOFs = list()
        """list[Tuple]: Tuples of DOFs (by involved atom indices to be removed)
        from reference geometry, hessian, and analysis"""
        # endregion
        self._validate_files()
        print("Successfully initialized SITH object with given input files...")

# region Atomic Homicide

    def __kill(self):
        """
        Executes the removal of degrees of freedom (DOFs) specified and any
        associated with specified atoms.

        Warning:
            This is a private method to limit user error. Specification of
            these DOFs is made by the user programmatically with the public
            functions set_kill_atoms(atoms: list) and set_kill_dofs(dofs: list)
            prior to data extraction by calling SITH.extract_data(). If no
            mismatch between number of DOFs in each geometry's coordinates and
            Hessian, this can be manually called after extract_data() as well
            but is not recommended.
        """
        print("Killing atoms and degrees of freedom...")
        self.__kill_dofs(self._killDOFs)
        dimsToKill = list()
        for atom in self._killAtoms:
            dimsToKill.extend(
                [dim for dim in self._reference.dim_indices if atom in dim])
        self.__kill_dofs(dimsToKill)
        print("Atoms and DOFs killed...")

    def __kill_dofs(self, dofs: list[Tuple]):
        """
        Removes the indicated degrees of freedom from the JEDI analysis, as
        such it removes them from the geometries' RICs as well as from the
        Hessian matrix.
        """
        rIndices = list()
        for dof in dofs:
            rIndices.extend([i for i in range(self._reference.dims[0])
                            if self._reference.dim_indices[i] == dof])
        # remove repetition
        rIndices = list(set(rIndices))
        self._reference._kill_DOFs(rIndices)

# endregion

# region Public Functions meant to be used by novice-intermediate user

    @property
    def reference(self) -> Geometry:
        """reference geometry"""
        return getattr(self, '_reference')

    @property
    def deformed(self) -> list[Geometry]:
        """deformed geometries"""
        return getattr(self, '_deformed')

    @property
    def hessian(self) -> np.ndarray:
        """
        Hessian matrix used to calculate the change in stress energy during
        SITH.analysis().

        Default value is that of the reference Geometry's Hessian.

        Note:
            Hessian matrix is the analytical gradient of the harmonic potential
            energy surface at a reference geometry as calculated by a frequency
            analysis at the level of DFT or higher.

        Warning:
            While this value is accessable through the reference Geometry for
            retrieval purposes, please refrain from manually setting it.
            If you would like to manually change this value for a SITH object,
            instead implement the method set_reference() and the associated
            refactoring recommended in its documentation.
        """

        return self.reference.hessian

    def set_kill_atoms(self, atoms: list):
        """
        Marks which atoms in the reference geometry should be removed during
        data extraction.

        Args:
            atoms (list): atoms which should be removed from the reference
            geometry

        Note:
            Any atoms in the deformed geometries not present
            in the reference geometry are automatically removed during
            extraction.

        Warning:
            This must be set prior to calling SITH.extract_data().
        """

        self._killAtoms = atoms
        self._kill = True
        print("Atoms to be killed are set...")

    def set_kill_dofs(self, dofs: list[Tuple]):
        """
        Marks which DOFs in the reference geometry should be removed during
        data extraction.

        Args:
            dofs (list[Tuple]): DOFs to be removed from the reference geometry.

        Note:
            Any atoms in the deformed geometries not present in the reference
            geometry are automatically removed during extraction.

        Warning:
            This must be set prior to calling SITH.extract_data().
        """
        self._killDOFs = dofs
        self._kill = True
        print("DOFs to be killed are set...")

    def remove_extra_dofs(self):
        """
        Removes any DOFs in the deformed geometry which are not in the
        reference geometry.

        Ensures that deformed geometry data is compatible with analysis based
        on the reference geometry and its Hessian.

        Warning:
            If there are DOFs in the reference geometry that do not exist in
            the deformed geometry, it will mess up this method and indicates
            an issue with the calculation which must be fixed in the generation
            of input.

            Solution: https://gaussian.com/gic/ specify Active GIC in deformed
            opt
        """
        print("Removing DOFs in the deformed geometries which are not present"
              "in the reference geometry...")
        for deformation in self._deformed:
            dofsToRemove = list()
            for j in range(max(deformation.dims[0], self._reference.dims[0])):
                if j < deformation.dims[0]:
                    # deformed not in reference
                    if deformation.dim_indices[j] not in \
                       list(self._reference.dim_indices):
                        dofsToRemove.append(j)
                # reference not in deformed
                if j < self._reference.dims[0]:
                    assert self._reference.dim_indices[j] in \
                        deformation.dim_indices, "Deformed geometry (" + \
                        deformation.name+") is missing reference DOF " + \
                        str(self._reference.dim_indices[j])+"."
            # Remove repetition
            dofsToRemove = list(set(dofsToRemove))
            deformation._kill_DOFs(dofsToRemove)

    def extract_data(self):
        """
        Extracts, validates, and organizes data from input files. Removes any
        marked atoms and DOFs.

        Note:
            Input files must be specified in SITH constructor, atoms and DOFs
            to remove previously must be specified by the user with
            set_kill_atoms or set_kill_dofs.
        Warning:
            This method must always be called prior to analyze() to extract and
            set upthe relevant data.
        """
        print("Beginning data extraction...")
        self._get_contents()

        rExtractor = Extractor(self._referencePath, self._rData)
        rExtractor._extract()
        # Create Geometry objects from reference and deformed data
        self._reference = rExtractor.get_geometry()
        self._deformed = list()
        for dd in self._dData:
            dExtractor = Extractor(dd[0], dd[1])
            dExtractor._extract()
            self._deformed.append(dExtractor.get_geometry())

        print("Finished data extraction...")

        # Defaults to the reference geometry Hessian, it is recommended to make
        # new SITH objects for each new analysis for the sake of clearer output
        # files but implementation of SITH.SetReference() as a public function
        # would enable the user to manually swap the reference geometry with
        # that of another geometry in the deformd list and then re-run
        # analysis.

        # Killing of atoms should occur here prior to validation for the sake
        # of DOF # atoms consistency, as well as before populating the q
        # vectors to ensure that no data which should be ignored leaks into the
        # analysis
        if self._kill:
            self.__kill()

        self.remove_extra_dofs()

        self._validate_geometries()

        self._populate_q()
        print("Finished setting up for energy analysis...")

    def analyze(self):
        """
        Performs the SITH energy analysis, populates energies,
        and pEnergies.

        Notes
        -----
        Consists of the dot multiplication of the deformation vectors and the
        Hessian matrix (analytical gradient of the harmonic potential energy
        surface) to produce both the total calculated change in energy between
        the reference structure and each deformed structure
        (SITH.deformationEnergy) as well as the subdivision of that energy into
        each DOF (SITH.energies).
        """
        print("Performing energy analysis...")
        if self.deltaQ is None or self.q0 is None or self.qF is None:
            raise Exception(
                "Populate Q has not been executed so necessary data for " +
                "analysis is lacking. This is likely due to not calling " +
                "extract_data().")
        self.energies = np.zeros(
            (self.reference.dims[0], len(self._deformed)))
        self.deformationEnergy = np.zeros((1, len(self._deformed)))
        self.pEnergies = np.zeros(
            (self._reference.dims[0], len(self._deformed)))

        for i in range(len(self._deformed)):
            # scalar 1x1 total Energy
            self.deformationEnergy[0, i] = 0.5 * np.transpose(self.deltaQ[:, i]
                                                              ).dot(
                self._reference.hessian).dot(self.deltaQ[:, i])
            for j in range(self._reference.dims[0]):
                isolatedDOF = np.hstack((np.zeros(j), self.deltaQ[j, i],
                                         np.zeros(
                    self._reference.dims[0]-j-1)))
                self.energies[j, i] = 0.5 * \
                    (isolatedDOF).dot(self._reference.hessian).dot(
                        self.deltaQ[:, i])
            self.pEnergies[:, i] = float(
                100) * self.energies[:, i] / self.deformationEnergy[0, i]

        print("Execute Order 67. Successful energy analysis completed.")

    def set_reference(self, geometryName: str):
        """
        UNIMPLEMENTED

        Replaces the current reference geometry used for calculation of stress
        energy with the specified geometry, pushing the current reference
        geometry to the deforation list.
        If the specified geometry does not exist or has no Hessian, the
        reference geometry will simply not be changed. Deformation energy is
        still calculated for all geometries aside from the new reference
        geometry.

        Use Case Note:
            This could be useful for cases where the error increases
            unacceptably and multiple reference points along a deformation are
            needed in order to have a more accurate energy calculation as the
            stretching coordinate progresses.

        Implementation Instructions:
            To implement this without re-extracting the data (which is
            basically the current workflow of creating a new SITH instead), you
            must set the SITH._reference geometry to that with the
            new geometryName, move the old reference geometry into the list of
            deformed geometries, and then _validate_geometries() _populate_q().
            This seems unnecessary as the cost which it saves is minimal.
        """
        raise NotImplementedError(
              "Unimplemented due to current lack of necessity, contact"
              "@mmfarrugia on github for more info.")
        """Use Case Note:
        This can be useful for cases where the error increases unacceptably
        and multiple reference points along a deformation are needed in order
        to have a more accurate energy calculation as the stretching coordinate
        progresses."""

# endregion

# region Validation

    def _validate_files(self):
        """
        Check that all files exist and whether the deformed path is a directory
        """
        print("Validating input files...")
        assert self._referencePath.exists(), \
               "Path to reference geometry data does not exist."
        assert self._deformedPath.exists(), \
               "Path to deformed geometry data does not exist."

        self.__deformedIsDirectory = self._deformedPath.is_dir()

    def _validate_geometries(self):
        """
        Ensure that the reference and deformed geometries are
        compatible(# atoms, # dofs, etc.)
        """
        print("Validating geometries...")
        assert all([deformn.n_atoms == self._reference.n_atoms and
                   np.array_equal(deformn.dims, self._reference.dims) and
                   np.array_equal(deformn.dim_indices,
                                  self._reference.dim_indices)
                   for deformn in self._deformed]), \
               "Incompatible number of atoms or dimensions amongst input " +\
               "files."

# endregion

    def _get_contents(self):
        """
        Gets the contents of the input files, including those in a deformed
        directory and verifies they are not empty.
        """
        print("Retrieving file contents...")
        try:
            with self._referencePath.open() as rFile:
                self._rData = rFile.readlines()
                assert len(self._rData) > 0, "Reference data file is empty."

            self._dData = list()
            if self.__deformedIsDirectory:
                dPaths = list(sorted(self._deformedPath.glob('*.fchk')))
                dPaths = [pathlib.Path(dp) for dp in dPaths]
            else:
                dPaths = [self._deformedPath]

            assert len(dPaths) > 0, "Deformed directory is empty."
            for dp in dPaths:
                with dp.open() as dFile:
                    dLines = dFile.readlines()
                    assert len(
                        dLines) > 0, "One or more deformed files are empty."
                    self._dData.append((dp, dLines))

        except AssertionError:
            # This exception catch is specific so that the AssertionErrors are
            # not caught and only "raise" is called so as to maintain the
            # original stack trace
            raise
        except Exception as e:
            print("An exception occurred during the extraction of the input " +
                  "files' contents.")
            print(e)
            sys.exit(sys.exc_info()[0])

    def _populate_q(self):
        """
        Populates the reference RIC vector q0, deformed RIC matrix qF, and a
        matrix deltaQ containing the changes in RICs.
        """

        print("Populating RIC vectors and calculating \u0394q...")
        self.q0 = np.zeros((self._reference.dims[0], 1))
        self.qF = np.zeros((self._reference.dims[0], 1))
        self.q0[:, 0] = np.transpose(np.asarray(self._reference.ric))
        # qF columns correspond to each deformed geometry, the rows correspond
        # to the degrees of freedom
        # qF[d.o.f. index, row or deformed geometry index] -> value of d.o.f.
        # for deformed geometry at index of self.deformed
        self.qF[:, 0] = np.transpose(np.asarray(self._deformed[0].ric))
        if len(self._deformed) > 1:
            for i in range(1, len(self._deformed)):
                deformation = self._deformed[i]
                temp = np.transpose(np.asarray(deformation.ric))
                self.qF = np.column_stack((self.qF, temp))
            # delta_q is organized in the same shape as qF
        self.deltaQ = np.subtract(self.qF, self.q0)

        """ This adjustment is to account for cases where dihedral angles
        oscillate about 180 degrees or pi and, since the coordinate system in
        Gaussian for example is from pi to -pi, where k and l are small, it
        shows up as

        -   -(pi-k) - (pi - l) = -2pi + (k+l) should be: -(k + l)
                --> -(result + 2pi)
        +   (pi - k) - -(pi - l) = 2pi - (k+l) should be (k+l)
                --> -(result - 2pi) = 2pi - result
        """
        with np.nditer(self.deltaQ, op_flags=['readwrite']) as dqit:
            for dq in dqit:
                dq[...] = dq - 2*np.pi if dq > np.pi \
                    else (dq + 2*np.pi if dq < -np.pi else dq)

# Error
    def compareEnergies(self):
        """
        Takes in SITH object sith, Returns Tuple of expected stress energy,
        stress energy error, and %Error

        Notes
        -----
        Expected Stress Energy: Total E deformed structure from input .fchk -
        total E reference structure from input .fchk
        Stress Energy Error: calculated stress energy - Expected Stress Energy
        %Error: Stress Energy Error / Expected Stress Energy
        """

        assert self.deformationEnergy is not None, \
            "SITH.energyAnalysis() has not been performed yet, no summary " +\
            "information available."
        obtainedDE = self.deformationEnergy
        expectedDE = np.zeros((1, len(self._deformed)))
        for i in range(len(self._deformed)):
            expectedDE[0, i] = self._deformed[i].energy - \
                self._reference.energy
        errorDE = self.deformationEnergy - expectedDE
        pErrorDE = (errorDE / expectedDE) * 100
        if not np.isfinite(pErrorDE[0][0]):
            pErrorDE[0][0] = -1
        return np.array([obtainedDE[0], expectedDE[0],
                         errorDE[0], pErrorDE[0]])
