from pathlib import Path
from glob import glob
import numpy as np
from ase import Atoms
from ase.units import Bohr
from ase.data import chemical_symbols
from SITH.Utilities import Geometry
from typing import Union


class FileReader:
    """Used to read info in a .fchk file and store it into a
    :mod:`~SITH.Utilities.Geometry` object.

    Attributes
    ==========
    dims: np.array
        counter of the DOFs, organized as:
        [0] Total number of DOFs
        [1] Number of distances
        [2] Number of angles
        [3] Number of dihedrals
    geometry: SITH.Utilities.Geometry
        object where the info is stored.
    block_readers: dict
        dictionary with the header and the respective reader that stores the
        values in :mod:`~SITH.Utilities.Geometry`.

    Note:
        This class should be used only for developers. Use this as a template
        to create other readers. Readers should assign the values to the
        attributes of the class :mod:`~SITH.Utilities.Geometry`.
    """
    def __init__(self, path: Union[Path, str], extract_data: bool = True):
        """Initializes g09 reader

        Parameters
        ==========
        path: Path or str
            Path to the fchk file to extract into a
            :mod:`~SITH.Utilities.Geometry` object.
        extract_data: bool
            Automatically try to extract the data from the .fchk file. If
            False, the user should run
            :mod:`~SITH.readers.g09_reader.FileReader._extract`. Defatult=True
        """
        if isinstance(path, str):
            path = Path(path)
        self._path = path
        self._name = path.stem

        # Geometry object into which the Extractor loads the extracted fchk
        # data:
        self.geometry = Geometry(self._name)

        # dims of DOFS [#DOFS, #lenghts, #angles, #dihedrals]
        self.dims = None

        # Read lines in file
        self.__lines = list()
        with path.open() as dFile:
            lines = dFile.readlines()
            assert len(lines) > 0, "One or more gaussian fchk files are empty."
            self.__lines = lines

        # False mainly for debug and tests
        if extract_data:
            # Add info to geometry object
            self._extract()

    def _extract(self) -> bool:
        """Extracts and populates Geometry information from self.__lines

        Returns
        =======
        (bool) True if extracting data is successful.
        """
        #  headers in fchk files from g09
        self.__energy_header = "Total Energy"
        self.__hessian_header = "Internal Force Constants"
        self.__coords_header = "Current cartesian coordinates"
        self.__DOF_header = "Redundant internal coordinates"
        self.__DOF_dim_header = "Redundant internal dimensions"
        self.__DOF_indices_header = "Redundant internal coordinate indices"
        self.__atomic_nums_header = "Atomic numbers"
        self.__internal_forces = "Internal Forces"

        # The next dict contains the headers of the blocks. The corresponding
        # values are the methods to read each block. all of them has the
        # current line as a parameter.
        self.block_readers = {self.__atomic_nums_header: self._anum_reader,
                              self.__coords_header: self._apos_reader,
                              self.__DOF_dim_header: self._dim_reader,
                              self.__DOF_indices_header: self._indices_reader,
                              self.__DOF_header: self._dof_values_reader,
                              self.__energy_header: self._scf_energy_reader,
                              self.__internal_forces: self._iforces_reader,
                              self.__hessian_header: self._hessian_reader}

        i = 0  # counter of lines
        while i < len(self.__lines):
            self.__lines[i]
            for block in self.block_readers.keys():
                if block in self.__lines[i]:
                    i, _ = self.block_readers[block](i)
                    del self.block_readers[block]
                    break
            i += 1

        self.build_atoms()

        return True

    def build_atoms(self) -> Atoms:
        """Creates a ase.Atoms object of the structure in the fchk file.

        Returns
        =======
        (ase.Atoms) Atoms object with the molecular information of the
        structure.

        Note: Before using this method, FileReader.cartesian_coordinates and
        FileReader.atomic_nums must have an assigned value.
        """
        assert self.cartesian_coord is not None, "Cartesian coordinates " +\
            "were not read properly"
        assert self.atomic_nums is not None, "Atomic numbers were not read " +\
            " properly"
        atomic_coord = self.cartesian_coord
        molecule = ''.join([chemical_symbols[int(i)]
                            for i in self.atomic_nums])
        self.geometry.atoms = Atoms(molecule, atomic_coord)

        return self.geometry.atoms

    # region readers
    def _fill_array(self, ith_line: int, dtype=None) -> tuple[np.ndarray, int]:
        """Fill an array after a given the header of a g09 block.

        Parameters
        ==========
        ith_line: int
            Index of the line where the header of the block is.
        dtype: data-type
            The desired data-type for the array. Default= Numpy guess.

        Returns
        =======
        (np.array, int) read data, the jth-line where the data
        finished in the file.

        Note: This method is useful because all the arrays in fchk files
        starts with a title and the numbers of elements in the end of the line.
        """
        # reads the number of elements
        len_list = int(self.__lines[ith_line].split()[-1])

        filling_list = list()
        while len(filling_list) < len_list:
            ith_line += 1
            filling_list.extend(self.__lines[ith_line].split())

        return np.array(filling_list, dtype=dtype), ith_line

    def _anum_reader(self, ith_line: int) -> tuple[int, np.ndarray]:
        """Read the atomic numbers

        Parameters
        ==========
        ith_line: int
            line index where atomic numbers header was found.

        Returns
        =======
        (int, np.array) index of the last line of the data block in the file,
        atomic numbers of the molecule with shape (#atoms).
        """
        self.atomic_nums, ith_line = self._fill_array(ith_line, dtype=int)
        self.geometry.n_atoms = len(self.atomic_nums)
        return ith_line, self.atomic_nums

    def _scf_energy_reader(self, ith_line: int) -> tuple[int, float]:
        """Read DFT energy

        Parameters
        ==========
        ith_line: int
            line index where energy header was found.

        Returns
        =======
        (int, float) same index line of the header, DFT energy of the structure
        in Hartrees.
        """
        self.geometry.scf_energy = float(self.__lines[ith_line].split()[-1])
        return ith_line, self.geometry.scf_energy

    def _dim_reader(self, ith_line: int) -> tuple[int, np.ndarray]:
        """Read DOFs dimensions

        Parameters
        ==========
        ith_line: int
            line index where dimensions header was found.

        Returns
        =======
        (int, np.array) index of the last line of the data block in the file,
        dimensions given as [#DOFS, #lenghts, #angles, #dihedrals].
        """
        ith_line += 1

        try:
            dims = np.array(self.__lines[ith_line].split(), dtype=int)
        except ValueError:
            raise Exception("Invalid input given for Redundant " +
                            "internal dimensions.")

        assert dims[0] == dims[1] + dims[2] + dims[3] and \
            len(dims) == 4, \
            "Invalid quantities of dimension types (bond " +\
            "lengths, angles, dihedrals) given in .fchk."
        self.dims = dims
        self.geometry.dims = dims

        return ith_line, dims

    def _indices_reader(self, ith_line: int) -> tuple[int, np.ndarray]:
        """Read DOFs indices

        Parameters
        ==========
        ith_line: int
            line index where indices definition header was found.

        Returns
        =======
        (int, np.array) index of the last line of the data block in the file,
        indices that define each DOF with shape (#DOFs, 4)

        Note: These indices start in one. Indices zero means None value in
        this array.
        """
        r_indices, ith_line = self._fill_array(ith_line, dtype=int)
        assert self.dims is not None, "The dimensions were not properly read"
        assert len(r_indices) == self.geometry.dims[0] * 4, \
            "Missing Redundant internal coordinate indices."
        self.geometry.dim_indices = r_indices.reshape((self.dims[0], 4))

        return ith_line, self.geometry.dim_indices

    def _dof_values_reader(self, ith_line: int) -> tuple[int, np.ndarray]:
        """Read DOFs values

        Parameters
        ==========
        ith_line: int
            line index where DOF values header was found.

        Returns
        =======
        (int, np.array) index of the last line of the data block in the file,
        values of the DOFs for the given configuration with shape (#DOFs).
        """
        dofs, ith_line = self._fill_array(ith_line, dtype=float)
        assert len(dofs) == self.dims[0], "unexpected number of DOFs while " +\
            "trying to extract DOFs values"
        dofs[:self.dims[1]] *= Bohr
        self.geometry.dof = dofs

        return ith_line, dofs

    def _apos_reader(self, ith_line: int) -> tuple[int, np.ndarray]:
        """Read atomic positions (cartesian coordinates).

        Parameters
        ==========
        ith_line: int
            line index where coordinates header was found.

        Returns
        =======
        (int, np.array) index of the last line of the data block in the file,
        atomic coordinates with shape (#atoms, 3).
        """
        r_coord, ith_line = self._fill_array(ith_line, dtype=float)
        assert len(r_coord) == self.geometry.n_atoms * 3, \
            "Missing coordinates components."
        self.cartesian_coord = r_coord.reshape((self.geometry.n_atoms, 3))
        self.cartesian_coord *= Bohr

        return ith_line, self.cartesian_coord

    def _hessian_reader(self, ith_line: int) -> tuple[int, np.ndarray]:
        """Read hessian elements and build the matrix

        Parameters
        ==========
        ith_line: int
            line index where hessian header was found.

        Returns
        =======
        (int, np.array) index of the last line of the data block in the file,
        hessian matrix with shape (#DOFs, #DOFs)
        """
        # Note: Gaussian stores the lower triangular matrix. This method builds
        # the complete matrix.

        assert self.dims is not None, "The dimensions were not properly read"

        r_hess, ith_line = self._fill_array(ith_line, dtype=float)
        n = self.dims[0]
        assert len(r_hess) == n * (n + 1) / 2, \
            "Missing Hessian elements."
        # Build Hessian matrix
        hessian = np.zeros((n, n))
        row, col = np.tril_indices(n)
        for i_elem in range(len(r_hess)):
            hessian[row[i_elem]][col[i_elem]] = r_hess[i_elem]
            # As the hessian is symmetric:
            hessian[col[i_elem]][row[i_elem]] = r_hess[i_elem]

        # fchk files of g09 store distances in Bohr, so, those DOFs defined as
        # distances must be transformed
        hessian[:, :self.dims[1]] /= Bohr
        hessian[:self.dims[1]] /= Bohr

        self.geometry.hessian = hessian

        return ith_line, self.geometry.hessian

    def _iforces_reader(self, ith_line: int) -> tuple[int, np.ndarray]:
        """Read internal forces.

        Parameters
        ==========
        ith_line: int
            line index where internal forces header was found.

        Returns
        =======
        (int, np.array) index of the last line of the data block in the file,
        generalized forces in DOFs with shape (#DOFs)
        """
        forces, ith_line = self._fill_array(ith_line, dtype=float)
        assert len(forces) == self.dims[0], "unexpected number of DOFs " +\
            "while trying to extract DOFs values"
        self.geometry.internal_forces = forces
        self.geometry.internal_forces[:self.dims[1]] /= Bohr

        return ith_line, self.geometry.internal_forces
    # endregion


class G09Reader:
    """Tool to read a set of fchk files of g09 corresponding to stretched
    molecules.

    Atributes
    =========
    inputfiles: list[Path]
        paths to the fchk files.
    structures: list[Geometry]
        list of :mod:`SITH.Utilities.Geometry`s for each deformed
        configuration.
    """
    def __init__(self, inputfiles: Union[list, str]):
        """"Object that creates the geometries for each stretched
        configuration.

        Parameters
        ==========
        inputfiles: list or str
            list of paths to the fchk files (as strings or paths). The order of
            the elements matters according to the energy anaylis applied. This
            parameter could be also a string of the path to the a directory
            containing the fchk files in alphabetic order.
        """
        if isinstance(inputfiles, str):
            master_path = Path(inputfiles)
            assert master_path.is_dir(), "Master directory does not exist."
            inputfiles = glob(inputfiles + '/*.fchk')
            inputfiles.sort()

        if isinstance(inputfiles[0], str):
            inputfiles = [Path(fil) for fil in inputfiles]

        self.inputfiles = inputfiles
        self._validate_files(self.inputfiles)
        self.structures = list()
        for fil in self.inputfiles:
            read_file = FileReader(fil)
            self.structures.append(read_file.geometry)

    def _validate_files(self, files_paths: list) -> bool:
        """Check that all files exist.

        Parameters
        ==========
        files_paths: list[Path]
            list of paths that you want to check if they exist. It raises an
            error in case any of the files does not exist.

        Returns
        =======
        (bool) True if it works.
        """
        # TODO: change all prints for logging
        # print("Validating input files...")
        for file in files_paths:
            assert file.exists(), \
               "Path to reference geometry data does not exist."

        return True
