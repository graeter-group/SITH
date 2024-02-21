import numpy as np
import importlib
from typing import Union
from SITH.energy_analysis.sith_analysis import SithAnalysis
from SITH.energy_analysis.jedi_analysis import JediAnalysis


class SITH:
    """A class to calculate and house SITH or JEDI analysis data.

    Attributes
    ==========
    reference: int
        Index of the structure to have as reference as a minimim energy.
    structures: list
        :mom:`SITH.Utilities.Geometry` s objects for each structure.
    n_structures: list
        Number of structures to include in the analysis.
    dims: np.array
        counter of the DOFs, organized as:
        [0]: total dimensions/DOFs
        [1]: bond lengths
        [2]: bond angles
        [3]: dihedral angles
    structures_scf_energies: np.array
        energies computed by the external software, e.g. DFT energies.
    all_dofs: np.array
        DOF values of the structures geometry with shape (#structures, #DOFs)
    delta_q: np.array
        Changes in DOF values computed and defined according to the analysis
        executed.
    all_hessians: np.array[float]
        Hessian matrixes of all the structures geometries with shape
        (#strucures, #DOFs, #DOFs)
    dof_energies: np.array
        Stress energy associated to all DOFs for all structures with shape
        (#structures, #DOFs)
    structure_energies: np.array
        Energy of each structure according to the distribution of energies with
        shape (#structures)
    energies_percentage: np.array
        Percentage of stress energy in each degree of freedom respect to the
        structure energy with shape (#structures, #DOFs)
    removed_dofs: dict
        Tracking of the removed DOFs. keys are '<structure index>_<DOF index>',
        values are the arrays of indices that define the removed DOF.
    dim_indices: np.array
        Indices that define each DOF with shape (#DOFs, 4). These indices start
        in one. Index zero means None for DOFs with less than 4 indices, e.g.
        distances.
    """

    def __init__(self, inputfiles: Union[list, str],
                 reader: str = 'G09Reader', reference: int = 0,
                 **kwargs):
        """Initializes a SITH object

        Parameters
        ==========
        inputfiles: list or str
            list of paths to the input files (as strings or paths) for the
            readers. The order of the elements matters according to the energy
            anaylis aplied. This parameter could also be a string of the path
            to the a directory containing the fchk files in alphabetic order.
        reader: str
            name of the reader. The options so far are:
            -- G09Reader
        reference: int (optional)
            index of the structure to have as a reference for the energy
            distribution analysis. Default=0
        **kwargs
            additional arguments for the selected reader.
        """
        self.removed_dofs = {}
        self.reference = reference
        reader = self._get_reader(reader)(inputfiles, **kwargs)
        self.structures = reader.structures

        # check compatibility between strucures
        self.remove_extra_dofs()
        self._validate_geometries()
        self.dim_indices = self.structures[0].dim_indices

        self.n_structures = len(self.structures)
        # structures geometries must be the equal.
        self.dims = self.structures[0].dims
        self.structures_scf_energies = np.array([defo.scf_energy
                                                 for defo in
                                                 self.structures]) - \
            self.structures[self.reference].scf_energy
        self.all_dofs = np.array([defo.dof for defo in self.structures])
        self.all_forces = np.array([defo.internal_forces
                                    for defo in self.structures])
        self.dofs_energies = None
        self.structure_energy = None
        self.energies_percertage = None
        self.delta_q = None

    def _get_reader(self, reader: str) -> object:
        """Import the class of the reader. It depends on the software used by
        the user to compute the energies.

        Parameters
        ==========
        reader: str
            name of the reader. check SITH.readers

        Returns
        =======
        (object) reader to be initialized.
        """
        # {name: modulus}
        readers_objs = {'G09Reader': 'SITH.readers.g09_reader'}
        module = importlib.import_module(readers_objs[reader])
        return getattr(module, reader)

    # region Validate
    def _validate_geometries(self):
        """Ensure that all structures are compatible(# atoms, # dofs, etc.)"""
        # TODO: Replace print by logging
        # print("Validating geometries...")
        ref = self.structures[0]
        assert all([deformn.n_atoms == ref.n_atoms
                    for deformn in self.structures]), \
            "Incompatible number of atoms"
        assert all([(deformn.dims == ref.dims).all()
                    for deformn in self.structures]), \
            f"Incompatible dimensions {ref.dims} " +\
            f"{self.structures[-1].dims}"
        assert all([(deformn.dim_indices == ref.dim_indices).all()
                    for deformn in self.structures]), "Incompatible dimensions"
    # endregion

    # Error
    def energies_error(self) -> np.ndarray:
        """Computes the error of the structure energy computed by the energy
        distribution and the energy obtained from the external software.

        Returns
        =======
        (np.array) array of the error given by:
        [0] absolute difference
        [1] percentage of error
        """

        assert self.structure_energy is not None, \
            "Energy distribution not computed yet."
        obtainedDE = self.structure_energy
        expectedDE = self.structures_scf_energies

        errorDE = obtainedDE - expectedDE
        pErrorDE = (errorDE / expectedDE) * 100
        if not np.isfinite(pErrorDE[0]):
            pErrorDE[0] = 100
        return np.array([errorDE[0], pErrorDE[0]])

    # region DOFsHomicide
    def remove_extra_dofs(self) -> dict:
        """Takes the structure with the lowest number of DOFs and removes the
        the DOFs in the other structures until get the same number of DOFs in
        all structures. Ensures that all structures data are compatible.
        """
        # TODO: change prints for logging
        # print("Removing extra DOFs in the structures... " +
        #      "(check SITH.removed_dofs)")
        n_dimensions = [defo.dims[0] for defo in self.structures]
        ref_index = n_dimensions.index(min(n_dimensions))
        min_n_dim = n_dimensions[ref_index]
        ref = self.structures[ref_index].dim_indices
        # stores the removed dofs with the keys <index_defo>_<index-dof>
        for i, defo in enumerate(self.structures):
            j = 0
            while j < defo.dims[0] and min_n_dim < defo.dims[0]:
                if not any(np.all(ref == defo.dim_indices[j],
                                  axis=1)):
                    self.removed_dofs[f'{i}_{j}'] = defo.dim_indices[j]
                    defo.kill_dofs([j])
                    j -= 1
                j += 1

        self.dim_indices = self.structures[0].dim_indices

        return self.removed_dofs

    def killer(self, killAtoms: list = None, killDOFs: list = None,
               killElements: list = None) -> dict:
        """
        Removes all DOFs that contain atoms, DOFs and elements to be removed.

        Parameters
        ==========
        killAtoms:
            list of indexes of atoms to be killed
        killDOFs:
            list of tuples with the DOFs to be killed
        killElements:
            list of strings with the elements to be killed. So, if you want to
            remove all hydrogens and carbons, use killElements=['H', 'C'].

        Return
        ======
        (list) [int] indexes of the removed DOFs.
        """
        if killAtoms is None:
            killAtoms = []
        if killDOFs is None:
            killDOFs = []
        if killElements is None:
            killElements = []

        # concatenate elements in atoms to be killed
        molecule = np.array(self.structures[0].atoms.get_chemical_symbols())

        for element in killElements:
            indexes_element = np.where(molecule == element)[0] + 1
            killAtoms.extend(indexes_element)

        # concatenate atoms in DOFs to be killed
        for atom in killAtoms:
            killDOFs.extend([i for i, dim in
                             enumerate(self.structures[0].dim_indices)
                             if atom in dim])

        indexes2kill = list(set(killDOFs))
        for index in indexes2kill:
            dim_indices = self.structures[0].dim_indices[index]
            self.removed_dofs[f'a_{index}'] = dim_indices

        for defo in self.structures:
            defo.kill_dofs(indexes2kill)

        # kill DOFs in sith
        self.dims = self.structures[0].dims
        self.all_dofs = np.delete(self.all_dofs, indexes2kill, axis=1)
        if self.delta_q is not None:
            self.delta_q = np.delete(self.delta_q, indexes2kill, axis=1)
        if self.all_forces[0] is not None:
            self.all_forces = np.delete(self.all_forces, indexes2kill, axis=1)

        e_ref = self.structures[self.reference].scf_energy
        self.structures_scf_energies = np.array([defo.scf_energy
                                                 for defo in
                                                 self.structures]) - e_ref

        self.dim_indices = self.structures[0].dim_indices
        # TODO: use logging instead of print
        # print("Atoms and DOFs killed...")

        return self.removed_dofs

    def rem_first_last(self, rem_first_def=0, rem_last_def=0,
                       from_last_minimum: bool = False) -> list:
        """Removes first and last structure configs and data from all the
        attributes of the Sith object.

        Parameters
        ==========
        rem_first_def: int (optional)
            number of configuration to remove in the first stretched
            configuration. Default=0
        rem_last_def: int (optional)
            number of configuration to remove in the last stretching
            configuration. Default=0
        from_last_minimum: bool (optional)
            set the rem_first_def to the structure of the last minium in the
            total energy if it is larger than current value of rem_first_def.
            Default=False

        Return
        ======
        (list) Deformed Geometry objects. SITH.structures.
        """
        if from_last_minimum:
            try:
                dif_ener = self.structures_scf_energies[1:] - \
                    self.structures_scf_energies[:-1] < 0
                i_min = (np.where(dif_ener)[0] + 1)[-1]
                ini_index = max(rem_first_def, i_min)
            except IndexError:
                ini_index = rem_first_def
                pass
        else:
            ini_index = rem_first_def

        last_index = self.n_structures - rem_last_def

        self.structures = self.structures[ini_index: last_index]
        self.n_structures = len(self.structures)
        self.structures_scf_energies = self.structures_scf_energies[
            ini_index: last_index]
        self.structures_scf_energies -= self.structures_scf_energies[
            self.reference]
        self.all_dofs = self.all_dofs[ini_index: last_index]
        if self.all_forces is not None:
            self.all_forces = self.all_forces[ini_index: last_index]
        if self.dofs_energies is not None:
            self.structure_energy = np.sum(self.dofs_energies, axis=1)

        return self.structures
    # endregion

    # region Energy Analysis
    def jedi_analysis(self):
        """Uses the values in 'structures' (:mod:`~SITH.Utilities.Geometry`
        of each configuration) and computes the distribution of energies
        using JEDI (Harmonic approximation)

        Return
        (float, np.array) energy predicted by the method summing up the
        energies of each DOF
        """
        # Jedi Analysis (ja)
        ja = JediAnalysis(self)
        self.dofs_energies, self.structure_energy = ja.jedi_analysis()
        return self.structure_energy, self.dofs_energies

    def sith_analysis(self, integration_method: str = 'trapezoid_integration'):
        """Uses the values in 'structures' (:mod:`~SITH.Utilities.Geometry`
        of each configuration) and computes the distribution of energies
        using SITH (Numerical integration).

        Parameters
        ==========
        integration_method: str
           choose one of the next integration methods:
               - trapezoid_integration (Default)
               - simpson_integration
               - rectangle_integration

        Return
        (float, np.array) energy predicted by the method summing up the
        energies of each DOF
        """
        # SITH Analysis (sa)
        sa = SithAnalysis(self)
        # transform integration method from string to method
        integration_method = getattr(sa, integration_method)
        self.dofs_energies, self.structure_energy = integration_method()
        return self.structure_energy, self.dofs_energies
    # endregion
