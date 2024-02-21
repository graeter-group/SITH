from ipywidgets import Output
import numpy as np
from SITH.SITH import SITH
from matplotlib.colors import BoundaryNorm, Colormap
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.transforms import Bbox


class Geometry:
    """Houses data associated with a molecular structure, all public variables
    are intended for access, not modification.

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
        Array of number of DOFs types in 4 components
        [0]: total dimensions/DOFs
        [1]: bond lengths
        [2]: bond angles
        [3]: dihedral angles
    dim_indices: np.array
        Indices that define each DOF with shape (#DOFs, 4). These indices start
        in one. Index zero means None for DOFs with less than 4 indices, e.g.
        distances.
    dof: np.array
        values of the DOFs for the given configuration with shape (#DOFs).
    hessian: np.array
        Hessian matrix associated with the geometry in units of
    internal_forces: np.array
        Forces in DOFs.
    atoms: ase.Atoms
        Atoms object associated with geometry.

    Note: All quantities are in units of Hartrees, Angstrom, radians
    """
    def __init__(self, name: str = ''):
        """Geometry object that stores the geometric information of each
        structure and sith takes to compute the energy distribution analysis.

        Parameters
        ==========
        name: str (optional)
            name of the Geometry object, it is arbitrary. Default defined by
            reader.
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


def color_distribution(sith: SITH, dofs: np.ndarray,
                       idef: int,
                       cmap: Colormap,
                       absolute: bool = False,
                       div: int = 5):
    """"
    Extract the energies of the specified DOFs and deformation structure, and
    the normalization according to a cmap.

    Parameters
    ==========
    sith: SITH
        sith object with the distribution of energies and structures
        information.
    dofs: array
        specific set of dofs to extract the energies.
    idef: int
        index of the structure to extract the energies.
    cmap: Colormap
        colormap to normalize according the number of divisions.
    absolute: bool
        True to define the color bar based on the maximum energy of the all
        the DOFS in all the stretching confs. False to define the color bar
        based on the maximum energy of the all the DOFS in the present
        stretching conf.
    div: int
        number of sets of colors in which the colorbar is divided.

    Returns
    =======
    (np.array, BoundaryNorm) set of energies of DOFs and deformation structure
    and the BoundaryNorm object to get the distribution of colors.
    """
    energies = []
    dof_ind = sith.dim_indices
    components = np.full(sith.dims[0], False, dtype=bool)

    for dof in dofs:

        dofindofs = np.all(dof_ind == dof, axis=1)
        components = np.logical_or(dofindofs, components)

    energies = sith.dofs_energies[:, components]

    if absolute:
        minval = min(energies.flatten())
        maxval = max(energies.flatten())
    else:
        minval = min(energies[idef].flatten())
        maxval = max(energies[idef].flatten())

    # In case of all the energies are the same (e.g. 0 stretching)
    if minval == maxval:
        minval = 0
        maxval = 1

    assert len(dofs) == len(energies[idef]), "The number of DOFs " + \
        f"({len(dofs)}) does not correspond with the number of " + \
        f"energies ({len(energies)}). This could happen because " + \
        "some DOFs that you are trying to map does not beling to" +\
        "sith.dim_indices"

    boundaries = np.linspace(minval, maxval, div + 1)
    normalize = BoundaryNorm(boundaries, cmap.N)

    return energies, normalize


def create_colorbar(normalize: BoundaryNorm, cmap: Colormap, deci: int,
                    label: str, labelsize: float, orientation: str,
                    width: int, height: int, dpi: int = 100) -> None:
    """
    Plot the colorbar according to BoundaryNorm.

    Parameters
    ==========
    normalize: BoundaryNorm
        normalization function.
    cmap: Colormap
        colormap to normalize according the number of indexes.
    deci: int
        number of decimals if the ticks in the colorbar.
    label: str
        label of the color bar.
    labelsize: int
        size of the labels of the colorbar.
    orientation: "vertical" or "horizontal".
            orientation of the color bar. Default: "vertical"
    width: int
        width (in pixels) of the space that will contain the scene and the
        color bar. Deault=700
    height: int
        height (in pixels) of the space that will contain the scene and the
        color bar. Deault=700

    Returns
    =======
    (plt.fig, widget.Output) Figure containing the colorbar and the widget of
    the figure.
    """
    if orientation == 'v' or orientation == 'vertical':
        rotation = 0
    else:
        rotation = 90

    # labelsize is given in points, namely 1/72 inches
    # the width is here defined as 1.16 times the space occupied by
    # the ticks and labels(see below)
    width_inches = 0.07*labelsize
    height_inches = height / dpi  # Convert pixels to inches

    fig, ax = plt.subplots(figsize=(width_inches, height_inches),
                           dpi=dpi)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=normalize,
                                              cmap=cmap),
                        cax=ax, orientation='vertical',
                        format='%1.{}f'.format(deci))
    cbar.set_label(label=label,
                   fontsize=labelsize,
                   labelpad=0.5 * labelsize)
    cbar.ax.tick_params(labelsize=0.8*labelsize,
                        length=0.2*labelsize,
                        pad=0.2*labelsize,
                        rotation=rotation)
    ax.set_position(Bbox([[0.01, 0.1],
                          [0.99-0.06*labelsize/width_inches,
                           0.9]]),
                    which='both')

    out = Output()
    with out:
        plt.show()

    return fig, out
