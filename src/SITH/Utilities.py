import numpy as np


class Geometry:
    """Houses data associated with a molecular structure, all public variables
    are intended for access not modification.

    Every sith.readers.<reader> must assing the values of Geometry attributes.

    Attributes
    ==========
    name: str
        Name of geometry, based off of stem of .fchk file path unless
        otherwise modified.
    n_atoms: int
        Number of atoms
    scf_energy: float
        Potential energy associated with geometry.
    dims: np.array
        Array of number of dimensions of DOF type with shape (4)
        [0]: total dimensions/DOFs
        [1]: bond lengths
        [2]: bond angles
        [3]: dihedral angles
    dof: np.array
        values of the DOFs for the given configuration with shape (#DOFs).
    dim_indices: np.array
        Indices that define each DOF with shape (#DOFs, 4). These indices start
        in one. Indices zero means None because of DOFs with less than 4
        indices, e.g. distances.
    hessian: np.array
        Hessian matrix associated with the geometry in units of
    internal_forces: np.array
        Forces in DOFs.
    atoms: ase.Atoms
        Atoms object associated with geometry.

    Note: All quantities are in units of Hartrees, Angstrom, radians
    """

    def __init__(self, name: str = ''):
        """Geometry object that SITH would take to compute the energy
        distribution analysis.

        Parameters
        ==========
        name: str (optional)
            name of the Geometry object, it is arbitrary. Default: ''
        """
        # region Basic Attributes

        # Note: All the attributes are marked as "mandatory", "optional". This
        # refers to the task the reader has to do.

        self.name = name  # optional
        self.n_atoms = None  # mandatory
        self.scf_energy = None  # optional
        self.dims = None  # mandatory
        self.dim_indices = None  # mandatory
        self.dof = None  # mandatory
        self.hessian = None  # mandatory for jedi_analysis
        self.internal_forces = None  # mandatory for sith_analysis
        self.atoms = None  # mandatory
        # endregion

    def __eq__(self, __o: object) -> bool:
        """Basic method used to compare two Geometry objects

        Parameters
        ==========
        __o: obj
            object to compare

        Returns
        =======
        (bool) True if Geometry objects have the same values on the attributes.
        """
        if not isinstance(__o, Geometry):
            return False
        b = True
        b = b and self.name == __o.name
        b = b and self.n_atoms == __o.n_atoms
        b = b and self.scf_energy == __o.scf_energy
        b = b and (self.dims, __o.dims).all()
        b = b and (self.dim_indices, __o.dim_indices).all()
        b = b and ((self.hessian is None and __o.hessian is None) or
                   (self.hessian == __o.hessian).all())
        b = b and ((self.internal_forces is None and 
                    __o.internal_forces is None) or
                   (self.internal_forces == __o.internal_forces).all())
        b = b and (self.dof, __o.dof).all()
        b = b and self.atoms == __o.atoms

        return b

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
