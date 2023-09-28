from array import array
from pathlib import Path
import pathlib
from ase import Atoms, Atom
from ase.data import chemical_symbols
from ase.units import Bohr
import numpy as np


class LTMatrix(list):
    """LTMatrix class and code adapted to PEP8 style but from https://github.com/ruixingw/rxcclib/blob/dev/utils/my/LTMatrix.py"""

    def __init__(self, L):
        """
        Accept a list of elements in a lower triangular matrix.
        """
        list.__init__(self, L)
        self.list = L
        i, j = LTMatrix.get_row_column(len(L) - 1)
        assert i == j, "Not a LTMatrix"
        self.dimension = i + 1

    def __getitem__(self, key):
        """
        Accept one or two integers.
        ONE: get item at the given position (count from zero)
        TWO: get item at the given (row, column) (both counted from zero)
        """
        if type(key) is tuple:
            return self.list[LTMatrix.get_position(*key)]
        else:
            return self.list[key]

    @staticmethod
    def get_row_column(N):
        """
        Return the row and column number of the Nth entry  of a lower triangular matrix.
        N, ROW, COLUMN are counted from ZERO!
        Example:
           C0 C1 C2 C3 C4 C5
        R0 0
        R1 1  2
        R2 3  4  5
        R3 6  7  8  9
        R4 10 11 12 13 14
        R5 15 16 17 18 19 20
        >>> LTMatrix.getRowColumn(18)
        (5, 3)
        18th element is at row 5 and column 3. (count from zero)
        """
        N += 1
        y = int((np.sqrt(1 + 8 * N) - 1) / 2)
        b = int(N - (y**2 + y) / 2)
        if b == 0:
            return (y - 1, y - 1)
        else:
            return (y, b - 1)

    @staticmethod
    def get_position(i, j):
        """
        Return the number of entry in the i-th row and j-th column of a symmetric matrix.
        All numbers are counted from ZERO.
        >>> LTMatrix.getPosition(3, 4)
        13
        """
        i += 1
        j += 1
        if i < j:
            i, j = j, i
        num = (i * (i - 1) / 2) + j
        num = int(num)
        return num - 1

    def build_full_mat(self):
        """
        build full matrix (np.ndarray).
        """
        L = []
        for i in range(self.dimension):
            L.append([])
            for j in range(self.dimension):
                L[-1].append(self[i, j])
        self._fullmat = np.array(L)
        return self._fullmat

    @property
    def full_mat(self):
        """
        Return the full matrix (np.ndarray)
        """
        return getattr(self, '_fullmat', self.build_full_mat())

    def inverse(self):
        return np.linalg.inv(self.full_mat)

    def uppermat(self):
        up = []
        for i, row in enumerate(self.full_mat):
            up.extend(row[i:])
        return up

    @staticmethod
    def new_from_upper_mat(uppermat):
        """
        Example:
           C0 C1 C2 C3 C4 C5
        R0  0  1  2  3  4  5
        R1     6  7  8  9 10
        R2       11 12 13 14
        R3          15 16 17
        R4             18 19
        R5                20
        """
        dimension = LTMatrix.get_row_column(len(uppermat)-1)[0] + 1

        def getUpperPosition(i, j, dimension):
            firstnum = dimension
            if i == 0:
                return j
            lastnum = firstnum - i + 1
            countnum = i
            abovenums = (firstnum + lastnum)*countnum / 2
            position = abovenums + j - i
            return int(position)
        L = []
        for i in range(dimension):
            for j in range(i+1):
                L.append(uppermat[getUpperPosition(j, i, dimension)])
        return LTMatrix(L)


class Geometry:
    """Houses data associated with a molecular structure, all public variables are intended for access not modification."""

    def __init__(self, name: str, path: pathlib.Path, n_atoms: int) -> None:
        self.name = name
        """Name of geometry, based off of stem of .fchk file path unless otherwise modified."""
        self._path = path
        """Path of geometry .fchk file."""
        self.ric = array('f')
        """Redundant Internal Coordinates of geometry in atomic units (Bohr radius)"""
        self.energy = None
        """Energy associated with geometry based on the DFT or higher level calculations used to generate the .fchk file input"""
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
        self.dim_indices = list()
        """List of Tuples referring to the indices of the atoms involved in each dimension/DOF in order of DOF index in ric"""

        self.hessian = None
        """Hessian matrix associated with the geometry. If 'None', then the associated fchk file did not contain any Hessian,
        in the case of Gaussian the Hessian is generated when a freq analysis is performed."""

    def build_atoms(self, raw_coords:list, atomic_num:list):
        assert len(raw_coords) == len(atomic_num) * 3, str(len(raw_coords))+" cartesian coordinates given, incorrect for "+str(len(atomic_num))+" atoms."
        atomic_coord = [Bohr * float(raw_coord) for raw_coord in raw_coords]
        atomic_coord = np.reshape(atomic_coord, (self.n_atoms, 3))
        molecule = ''.join([chemical_symbols[int(i)] for i in atomic_num])
        self.atoms = Atoms(molecule, atomic_coord)

    def build_RIC(self, dims: list, dim_indices_lines: list, coord_lines: list):
        """
        Takes in lists of RIC-related data, Populates 

            dims: quantities of each RIC dimension type
            dimILines: list of strings of each line of RIC Indices 
            coordLines: list of strings of each line of RICs
        """
        try:
            self.dims = array('i', [int(d) for d in dims])
        except ValueError:
            raise Exception(
                "Invalid input given for Redundant internal dimensions.")
        assert self.dims[0] == self.dims[1] + self.dims[2] + self.dims[3] and len(
            dims) == 4, "Invalid quantities of dimension types (bond lengths, angles, dihedrals) given in .fchk."

        # region Indices
        # Parses through the 'dimILines' input which indicates which atoms (by index)
        # are involved in each RIC degree of freedom

        raw_indices = list()
        for indices_line in dim_indices_lines:
            indices_line_split = indices_line.split()
            try:
                raw_indices.extend([int(i) for i in indices_line_split])
            except ValueError as ve:
                print(ve)
                raise Exception("Invalid atom index given as input.")

        # Check that # indices is divisible by 4
        assert len(raw_indices) % 4 == 0 and len(
            raw_indices) == self.dims[0] * 4, "One or more redundant internal coordinate indices are missing or do not have the expected format. Please refer to documentation"

        # Parse into sets of 4, then into tuples of the relevant number of values
        lengths_count = 0
        angles_count = 0
        diheds_count = 0
        for i in range(0, len(raw_indices), 4):
            a1 = raw_indices[i]
            a2 = raw_indices[i+1]
            a3 = raw_indices[i+2]
            a4 = raw_indices[i+3]
            # Check that the number of values in each tuple matches the dimension type (length, angle, dihedral) for that dim index
            # These should line up with self.dims correctly
            assert all([(x <= self.n_atoms and x >= 0)
                       for x in raw_indices[i:i+4]]), "Invalid atom index given as input."
            assert a1 != a2 and a1 != a3 and a1 != a4 and a2 != a3 and a2 != a4 and (
                a3 != a4 or a3 == 0), "Invalid RIC dimension given, atomic indices cannot repeat within a degree of freedom."
            assert a1 != 0 and a2 != 0, "Mismatch between given 'RIC dimensions' and given RIC indices."
            # bond lengths check
            if i < self.dims[1]*4:
                assert a3 == 0 and a4 == 0, "Mismatch between given 'RIC dimensions' and given RIC indices."
                self.dim_indices.append((a1, a2))
                lengths_count += 1
            # bond angles check
            elif i < (self.dims[1] + self.dims[2])*4:
                assert a3 != 0 and a4 == 0, "Mismatch between given 'RIC dimensions' and given RIC indices."
                self.dim_indices.append((a1, a2, a3))
                angles_count += 1
            # dihedral angles check
            elif i < (self.dims[1] + self.dims[2] + self.dims[3])*4:
                assert a3 != 0 and a4 != 0, "Mismatch between given 'RIC dimensions' and given RIC indices."
                self.dim_indices.append((a1, a2, a3, a4))
                diheds_count += 1

        assert lengths_count == self.dims[1] and angles_count == self.dims[2] and diheds_count == self.dims[
            3], "Redundant internal coordinate indices given inconsistent with Redundant internal dimensions given."

        # endregion

        try:
            for line in coord_lines:
                self.ric.extend([float(ric) for ric in line.split()])
        except ValueError:
            raise(Exception(
                "Redundant internal coordinates contains invalid values, such as strings."))

        self.ric = np.asarray(self.ric, dtype=np.float32)
        assert len(self.ric) == self.dims[0], "Mismatch between the number of degrees of freedom expected ("+str(
            dims[0])+") and number of coordinates given ("+str(len(self.ric))+")."

        # Angles are moved (-pi, pi)
        for i in range(self.dims[1], self.dims[0]):
            self.ric[i] = self.ric[i]

    def _kill_DOFs(self, dof_indices: list[int]):
        """Takes in list of indices of degrees of freedom to remove, Removes DOFs from ric, dim_indices, and hessian, updates dims"""
        self.ric = np.delete(self.ric, dof_indices)
        self.internal_forces = np.delete(self.internal_forces, dof_indices)
        tdofremoved = [0, 0, 0]  # counter of the number of DOF removed 
                                 # arranged in types (lenght, angle, dihedral)
        for index in sorted(dof_indices, reverse=True):
            tdofremoved[len(self.dim_indices[index]) - 2] += 1
            del self.dim_indices[index]

        self.dims[0] -= len(dof_indices)
        self.dims[1] -= tdofremoved[0]
        self.dims[2] -= tdofremoved[1]
        self.dims[3] -= tdofremoved[2]

        if(self.hessian is not None):
            self.hessian = np.delete(self.hessian, dof_indices, axis=0)
            self.hessian = np.delete(self.hessian, dof_indices, axis=1)

    def __eq__(self, __o: object) -> bool:
        b = True
        b = b and self.name == __o.name
        b = b and np.array_equal(self.ric, __o.ric)
        b = b and self.energy == __o.energy
        b = b and self.atoms == __o.atoms
        b = b and self.n_atoms == __o.n_atoms
        b = b and np.array_equal(self.dims, __o.dims)
        b = b and np.array_equal(self.dim_indices, __o.dim_indices)
        b = b and ((self.hessian is None and __o.hessian is None)
                   or np.array_equal(self.hessian, __o.hessian))
        return b


class UnitConverter:
    """
    Class to convert units utilizing Atomic Simulation Environment (ase) constants
    xyz standard input is in Angstrom
    RIC are in atomic units
    internal Hessian is in atomic units of length: Ha/Bohr^2 angle: Hartree/radian^2
    Note: values in all Gaussian version 3 formatted checkpoint files are in atomic units
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def angstrom_to_bohr(a: float) -> np.float32:
        return np.float32(a / Bohr)

    @staticmethod
    def bohr_to_angstrom(b: float) -> np.float32:
        return np.float32(b * Bohr)

    @staticmethod
    def radian_to_degree(r: float) -> np.float32:
        return np.float32(r / np.pi * 180)


class Extractor:
    """Used on a per .fchk file basis to organize lines from the fchk file into a Geometry

    Note:
        The user really shouldn't be using this class 0.0 unless they perhaps want a custom one for
        non-fchk files, in which case they should make a new class inheriting from and overriding methods in this one."""

    def __init__(self, path: Path, lines_list: list) -> None:
        """Initializes an Extractor

        Args:
            path (Path): path to the fchk file to extract into a SITH.Utilities.Geometry object
            lines_list (list): content of the input fchk file as a list of the lines of the file.
        """        
        self.__energy_header = "Total Energy"
        self.__hessian_header = "Internal Force Constants"
        self.__cartesian_coords_header = "Current cartesian coordinates"
        self.__RIC_header = "Redundant internal coordinates"
        self.__RIC_dim_header = "Redundant internal dimensions"
        self.__RIC_indices_header = "Redundant internal coordinate indices"
        self.__num_atoms_header = "Number of atoms"
        self.__atomic_nums_header = "Atomic numbers"
        self.__internal_forces = "Internal Forces                            R"

        self._path = path
        self._name = path.stem

        self.geometry = None
        "Geometry object into which the Extractor loads the extracted fchk data"

        self.__lines = lines_list

    def _extract(self) -> bool:
        """Extracts and populates Geometry information from self.__lines

        Returns:
            bool: True if successful
        """        

        try:
            i = 0
            atomic_nums = list()
            while i < len(self.__lines):
                line = self.__lines[i]

                # This all must be in order because of the way things show up in the fchk file, it makes no sense to build
                # these out separately to reduce dependencies because it will just increase the number of times to traverse the data.
                if self.__num_atoms_header in line:
                    split_line = line.split()
                    num_atoms = int(split_line[len(split_line)-1])
                    self.geometry = Geometry(self._name, self._path, num_atoms)

                if self.__atomic_nums_header in line:
                    i += 1
                    stop = int(line.split()[-1])
                    count = 0
                    while count < stop:
                        atomic_nums.extend(self.__lines[i].split())
                        count += len(self.__lines[i].split())
                        i += 1

                elif self.__energy_header in line:
                    split_line = line.split()
                    self.geometry.energy = float(split_line[len(split_line)-1])

                elif self.__RIC_dim_header in line:
                    i = i+1
                    lin = self.__lines[i]
                    r_dims = lin.split()

                elif self.__RIC_indices_header in line:
                    rdiStart = i+1
                    stop = int(line.split()[-1])
                    count = 0
                    while count < stop:
                        count += len(self.__lines[i].split())
                        i += 1
                    xrDims = self.__lines[rdiStart:i+1]
                    assert len(
                        xrDims) > 0, "Missing Redundant internal coordinate indices."

                if self.__RIC_header in line:
                    i = i+1
                    xrStart = i
                    stop = int(line.split()[-1])
                    count = 0
                    while count < stop:
                        count += len(self.__lines[i].split())
                        i += 1
                    xrRaw = self.__lines[xrStart:i]
                    self.geometry.build_RIC(r_dims, xrDims, xrRaw)

                if self.__cartesian_coords_header in line:
                    i = i+1
                    stop = int(line.split()[-1])
                    count = 0
                    c_raw = list()
                    while count < stop:
                        c_raw.extend(self.__lines[i].split())
                        count += len(self.__lines[i].split())
                        i += 1
                    assert len(atomic_nums) == num_atoms, "Mismatch between length of atomic numbers and number of atoms specified."
                    self.geometry.build_atoms(c_raw, atomic_nums)

                elif self.__hessian_header in line:
                    i = i+1
                    stop = int(line.split()[-1])
                    self.h_raw = list()
                    while len(self.h_raw) < stop:
                        row = self.__lines[i]
                        row_split = row.split()
                        self.h_raw.extend([float(i) for i in row_split])
                        i = i+1

                elif self.__internal_forces in line:
                    i += 1
                    stop = int(line.split()[-1])
                    count = 0
                    self.internal_forces = list()

                    while count < stop:
                        row = self.__lines[i]
                        rowsplit = row.split()
                        forces = [float(force) for force in rowsplit]
                        self.internal_forces.extend(forces)
                        count += len(forces)
                        i += 1
                    i -= 1
                    # g09 stores forces in chk using [eV]/[A ]
                    self.geometry.internal_forces = np.array(self.internal_forces)
                i = i + 1

            # Prints H when starts to read a hessian and adds * when is done.
            print("H", end='')
            self.build_hessian()
            print("* ", end='')
            return True
        except Exception as e:
            print(e)
            print("Data extraction failed.")
            return False

    def build_hessian(self):
        """Properly formats the Hessian matrix from the lower triangular matrix given by the .fchk data"""
        lt_mat = LTMatrix(self.h_raw)
        self.hessian = lt_mat.full_mat
        self.geometry.hessian = self.hessian

    def get_geometry(self) -> Geometry:
        """Gets the geometry as extracted from the .fchk file given to the Extractor.

        Raises:
            Exception: if `geometry` is None, implying it hasn't been populated by the Extractor

        Returns:
            Geometry: geometry populated by the Extractor based on the input lines from a .fchk file
        """        
        if self.geometry is not None:
            return self.geometry
        else:
            raise Exception("There is no geometry.")


class void:
    pass


class SummaryReader:
    def __init__(self, file:str):
        """Extract the data in a summary file

        Args:
            file (str): summary file from which to extract data
        """        
        with open(file) as data:
            self.info = data.readlines()

        self._reference = void()
        self._deformed = list()
        self.read_all()

    def read_section(self, header, tail, iplus=0, jminus=0) -> list[str]:       
        """
        Take any block of a set of lines separated by header
        and tail as references

        Args:
            lines (list[str]): set of lines from which to extract the blocks.
            header (str): line that marks the start of the block. It is included
            in the block.
            tail (str): line that marks the end of the block. It is not included
            in the block.
            iplus (int): number of lines to ignore after header. Defaults to 0.
            jminus (int): number of lines to ignore before the tail. Defaults to 0.

        Returns:
            data (list[str]): set of lines extracted
        """
        i = self.info.index(header)
        j = self.info.index(tail)
        data = self.info[i+iplus:j-jminus]
        return data

    def read_dofs(self) -> list[tuple]:
        """Read in the degrees of freedom in the summary file

        Returns:
            list[tuple]: degrees of freedom as tuples of the atomic indices involved
        """        

        lines = self.read_section('Redundant Internal Coordinate' +
                                  ' Definitions\n',
                                  'Changes in internal coordinates' +
                                  ' (\u0394q)\n',
                                  iplus=2)
        return [tuple(np.fromstring(line.replace('(',
                                                 '').replace(',',
                                                             '').replace(')',
                                                                         ''),
                sep=' ', dtype=int)[1:])
                for line in lines]

#TODO make this a 2D array with row column maybe? for ease bc lists take up more than arrays
    def read_changes(self) -> list[list]:
        """
        Read the changes in the DOFs (\u0394q) in the summary file

        Returns:
            list[list]: where the i-th list is the changes in the
        i-th deformed config.
        """
        lines = self.read_section('Changes in internal coordinates' +
                                  ' (\u0394q)\n',
                                  '**  Energy Analysis  **\n',
                                  iplus=3, jminus=2)
        return np.array([np.fromstring(line, sep=' ')[1:] for line in lines])

#TODO finish making documentation conform with Google style
    def read_accuracy(self) -> np.ndarray:     
        """
        read the accuracy in the total difference of energy.

        return a list lists where each element corresponds with the
        energy-analysis of each deformed config saved as
        [energy diff predicted with harmonic approx, percentaje_error, Error]
         """
        lines = self.read_section('Overall Structural Energies\n',
                                  'Energy per DOF (RIC)\n',
                                  iplus=2)

        return np.array([np.array(line.split()[1:],
                                  dtype=float)
                         for line in lines]).T

    def read_energies(self, ndofs):
        """
        read the energies in each degree of freedom

        return a list lists where each element corresponds to the
        distribution of energies of each deformed config
        """
        lines = self.read_section('Energy per DOF (RIC)\n',
                                  'Energy per DOF (RIC)\n',
                                  iplus=2, jminus=-ndofs-2)

        return np.array([np.array(line.split(),
                                  dtype=float)[1:]
                         for line in lines])

    def read_structures(self):
        """
        Creates the ase.Atoms objects of each structure.
        """
        init = self.info.index("XYZ FILES APPENDED\n") + 1
        n_configs = len(self.deltaQ[0]) + 1  # deformed plus reference
        length = int(len(self.info[init:])/n_configs)
        n_atoms = int(self.info[init])

        configs = [self.info[init+i*(length):init+(i+1)*length]
                   for i in range(n_configs)]
        configs = [[atom.split() for atom in config] for config in configs]
        molecule = ''.join([atom[0] for atom in configs[0][-n_atoms:]])
        positions = np.array([[np.array(atom[1:], dtype=float)
                              for atom in config[-n_atoms:]]
                              for config in configs])

        atoms = [Atoms(molecule, config) for config in positions]

        return atoms[0], atoms[1:]

    def read_all(self):
        """
        Read all summary file and save the info in instances
        """
        self.dim_indices = self.read_dofs()
        dims = [len(self.dim_indices), 0, 0, 0]
        for dof in self.dim_indices:
            dims[len(dof)-1] += 1
        self._reference.dims = np.array(dims, dtype=int)
        self.deltaQ = self.read_changes()
        self.accuracy = self.read_accuracy()
        assert len(self.dim_indices) == len(self.deltaQ), \
            f'{len(self.dim_indices)} DOF' +\
            f'but {len(self.deltaQ[0])} changes are reported'

        self.energies = self.read_energies(len(self.deltaQ))
        self._reference.atoms, deformed = self.read_structures()
        [self._deformed.append(void()) for _ in range(len(deformed))]

        for i in range(len(deformed)):
            self._deformed[i].atoms = deformed[i]

        self._reference.dim_indices = self.dim_indices
