import numpy as np


class Geometry:
    """
    Houses data associated with a molecular structure, all public variables
    are intended for access not modification.
    """

    def __init__(self, name: str, path: pathlib.Path, n_atoms: int) -> None:
        self.name = name
        """Name of geometry, based off of stem of .fchk file path unless
           otherwise modified."""
        self._path = path
        """Path of geometry .fchk file."""
        self.ric = array('f')
        """Redundant Internal Coordinates of geometry in atomic units (Bohr
        radius)"""
        self.energy = None
        """Energy associated with geometry based on the DFT or higher level
        calculations used to generate the .fchk file input (Hatrees)"""
        self.atoms = list()
        """<ase.Atoms> object associated with geometry."""
        self.n_atoms = n_atoms
        """Number of atoms"""
        self.dims = array('i')
        """Array of number of dimensions of DOF type
        [0]: total dimensions/DOFs
        [1]: bond lengths
        [2]: bond angles
        [3]: dihedral angles
        """
        self.dim_ndices = list()
        """List of Tuples referring to the indices of the atoms involved in
        each dimension/DOF in order of DOF index in ric"""

        self.hessian = None
        """
        Hessian matrix associated with the geometry. If 'None', then the
        associated fchk file did not contain any Hessian, in the case of
        Gaussian the Hessian is generated when a freq analysis is performed.
        """

    def build_atoms(self, raw_coords: list, atomic_num: list):
        assert len(raw_coords) == len(atomic_num) * 3, \
               f"{len(raw_coords)} cartesian coordinates given, incorrect " +\
               f"for {len(atomic_num)} atoms."
        atomic_coord = [Bohr * float(raw_coord) for raw_coord in raw_coords]
        atomic_coord = np.reshape(atomic_coord, (self.n_atoms, 3))
        molecule = ''.join([chemical_symbols[int(i)] for i in atomic_num])
        self.atoms = Atoms(molecule, atomic_coord)

    def build_RIC(self, dims: list, dim_indices_lines: list,
                  coord_lines: list):
        """
        Takes in lists of RIC-related data, Populates

            dims: quantities of each RIC dimension type
            dim_indices_lines: list of strings of each line of RIC Indices
            coord_lines: list of strings of each line of RICs
        """
        try:
            self.dims = array('i', [int(d) for d in dims])
        except ValueError:
            raise Exception(
                "Invalid input given for Redundant internal dimensions.")
        assert self.dims[0] == self.dims[1] + self.dims[2] + self.dims[3] and \
               len(dims) == 4, \
               "Invalid quantities of dimension types (bond lengths, " +\
               "angles, dihedrals) given in .fchk."

        # region Indices
        # Parses through the 'dim_indices_lines' input which indicates which
        # atoms (by index) are involved in each RIC degree of freedom

        raw_indices = list()
        for indices_line in dim_indices_lines:
            indices_line_split = indices_line.split()
            try:
                raw_indices.extend([int(i) for i in indices_line_split])
            except ValueError as ve:
                print(ve)
                raise Exception("Invalid atom index given as input.")

        # Check that # indices is divisible by 4
        assert len(raw_indices) % 4 == 0 and \
               len(raw_indices) == self.dims[0] * 4, \
               "One or more redundant internal coordinate indices are " +\
               "missing or do not have the expected format. Please refer " +\
               "to documentation"

        # Parse into sets of 4, then into tuples of the relevant number of
        # values
        lengths_count = 0
        angles_count = 0
        diheds_count = 0
        for i in range(0, len(raw_indices), 4):
            a1 = raw_indices[i]
            a2 = raw_indices[i+1]
            a3 = raw_indices[i+2]
            a4 = raw_indices[i+3]
            # Check that the number of values in each tuple matches the
            # dimension type (length, angle, dihedral) for that dim index
            # These should line up with self.dims correctly
            assert all([(x <= self.n_atoms and x >= 0)
                        for x in raw_indices[i: i + 4]]), \
                   "Invalid atom index given as input."
            assert a1 != a2 and a1 != a3 and \
                   a1 != a4 and a2 != a3 and \
                   a2 != a4 and (a3 != a4 or a3 == 0), \
                   "Invalid RIC dimension given, atomic indices cannot " +\
                   "repeat within a degree of freedom."
            assert a1 != 0 and \
                   a2 != 0, \
                   "Mismatch between given 'RIC dimensions' and given " +\
                   "RIC indices."
            # bond lengths check
            if i < self.dims[1]*4:
                assert a3 == 0 and \
                       a4 == 0, \
                       "Mismatch between given 'RIC dimensions' and given " +\
                       "RIC indices."
                self.dim_indices.append((a1, a2))
                lengths_count += 1
            # bond angles check
            elif i < (self.dims[1] + self.dims[2])*4:
                assert a3 != 0 and \
                       a4 == 0, \
                       "Mismatch between given 'RIC dimensions' and given " +\
                       "RIC indices."
                self.dim_indices.append((a1, a2, a3))
                angles_count += 1
            # dihedral angles check
            elif i < (self.dims[1] + self.dims[2] + self.dims[3])*4:
                assert a3 != 0 and \
                       a4 != 0, \
                       "Mismatch between given 'RIC dimensions' and given " +\
                       "RIC indices."
                self.dim_indices.append((a1, a2, a3, a4))
                diheds_count += 1

        assert lengths_count == self.dims[1] and \
               angles_count == self.dims[2] and \
               diheds_count == self.dims[3], \
               "Redundant internal coordinate indices given inconsistent " +\
               "with Redundant internal dimensions given."

        # endregion

    def kill_dofs(self, dof_indices: list[int]) -> np.array:
        """Takes in list of indices of degrees of freedom and removes DOFs from
        dof, dim_indices, internal_forces, and hessian; updates dims

        Parameters
        ==========
        dof_indices: list
            list of indices to remove.

        Returns
        =======
        (numpy.array) dim_indices of removed dofs.
        """
        # remove repetition
        dof_indices = list(set(dof_indices))

        self.dof = np.delete(self.dof, dof_indices)

        # counter of the number of DOF removed arranged in types (lenght,
        # angle, dihedral)
        tdofremoved = [0, 0, 0]
        for index in sorted(dof_indices, reverse=True):
            tdofremoved[np.count_nonzero(self.dim_indices[index]) - 2] += 1
        dof_indices2remove = self.dim_indices[dof_indices]
        self.dim_indices = np.delete(self.dim_indices, dof_indices, axis=0)

        self.dims[0] -= len(dof_indices)
        self.dims[1] -= tdofremoved[0]
        self.dims[2] -= tdofremoved[1]
        self.dims[3] -= tdofremoved[2]

        if self.internal_forces is not None:
            self.internal_forces = np.delete(self.internal_forces, dof_indices)

        if (self.hessian is not None):
            self.hessian = np.delete(self.hessian, dof_indices, axis=0)
            self.hessian = np.delete(self.hessian, dof_indices, axis=1)

        return dof_indices2remove
