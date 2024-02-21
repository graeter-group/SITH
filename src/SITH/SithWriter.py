from pathlib import Path
from ase.units import Bohr
import numpy as np
import os


class WriteSITH:
    def __init__(self, geometry):
        """Object that set the variables used in sith as g09 fchk format.
        That helps the user to reduce the amount of data store for sith
        analysis, while still having the posibility to read it and reuse it
        again.
        """
        self.geometry = geometry

        # Units in fchk are different:
        self.geometry.dof[:self.geometry.dims[1]] /= Bohr
        
        if self.geometry.hessian is not None:
            self.geometry.hessian[:, : self.geometry.dims[1]] *= Bohr
            self.geometry.hessian[: self.geometry.dims[1]] *= Bohr
        if self.geometry.internal_forces is not None:
            self.geometry.internal_forces[:self.geometry.dims[1]] *= Bohr

        self._transform_hessian()

        self.lines = f'This file was created using SITH.SithWriter. All ' +\
            'the units of quantities are in Hartrees, Angstroms or Radians.\n'

        #  headers in fchk files in g09 format
        self.__energy_header = "Total Energy"
        self.__hessian_header = "Internal Force Constants"
        self.__coords_header = "Current cartesian coordinates"
        self.__DOF_header = "Redundant internal coordinates"
        self.__DOF_dim_header = "Redundant internal dimensions"
        self.__DOF_indices_header = "Redundant internal coordinate indices"
        self.__atomic_nums_header = "Atomic numbers"
        self.__internal_forces = "Internal Forces"

        self.headers_scalars = {self.__energy_header: 'scf_energy'}

        self.headers_vectors = {self.__DOF_dim_header: 'dims',
                                self.__DOF_indices_header: 'dim_indices',
                                self.__DOF_header: 'dof',
                                self.__internal_forces: 'internal_forces',
                                self.__hessian_header: 'hessian'}

    def _transform_hessian(self):
        if self.geometry.hessian is not None:
            n = self.geometry.dims[0]
            row, col = np.tril_indices(n)
            tril = np.zeros(len(row))
            for i in range(len(row)):
                tril[i] = self.geometry.hessian[row[i]][col[i]]
            self.geometry.hessian = tril
            return tril
        else:
            return None

    def _write_atoms(self):
        """Writes atoms coordinates and atomic numbers"""
        cs = self.geometry.atoms.get_atomic_numbers()
        pos = self.geometry.atoms.positions / Bohr
        lines = self._write_array(self.__atomic_nums_header, cs)
        lines += self._write_array(self.__coords_header, pos)
        self.lines += lines

        return lines

    def _write_array(self, header, array):
        """Writes any array or list in fchk format, namely, flattened
        and with the description on the header specifying the number of
        elements
        
        Parameters
        ==========
        header: str
            name of the header
        array:
            array with with any shape to be written
        
        Return
        ======
        (str) lines to be printed
        """
        flatten = array.flatten()
        n_elements = len(flatten)

        if 'int' in array.dtype.name:
            valuetype = 'I'
        elif 'float' in array.dtype.name:
            valuetype = 'R'
        else:
            raise ValueError("values in writer only accept integers or reals")

        lines = header.ljust(44) + valuetype + "   N=" + \
            str(n_elements).rjust(12) + "\n"

        if valuetype == 'R':
            i = 0
            while i*5 < len(flatten):
                for value in flatten[5 * i: 5 * i + 5]:
                    lines += "  {:.8E}".format(value)
                lines += "\n"
                i += 1

        elif valuetype == 'I':
            i = 0
            while i*6 < len(flatten):
                for value in flatten[6 * i: 6 * i + 6]:
                    lines += str(value).rjust(12)
                lines += "\n"
                i += 1

        self.lines += lines

        return lines
    
    def _write_scalar(self, header, value):
        """Writes any scalar in fchk format, namely, in the header
        
        Parameters
        ==========
        header: str
            name of the header
        value:
            value to be written
        
        Return
        ======
        (str) lines to be printed
        """
        if isinstance(value, int):
            line = header.ljust(43) + 'I' + str(value).rjust(17) + '\n'
        elif isinstance(value, float):
            valuetype = 'R'
            line = header.ljust(43) + valuetype + str(value).rjust(27) + '\n'
        else:
            raise ValueError("values in writer only accept integers or reals but the variable is " + str(type(value)))

        self.lines += line

        return line

    def write_file(self, outputfile: str = './geometry_data.fchk'):
        """fills lines attribute and writes the output file
        
        Parameters
        ==========
        outputfile: str (optional)
            name of the output file. It should have fchk extension.
            Default=./geometry_data.fchk
        
        Return
        ======
        (str) lines to be printed in the output file.
        """
        self._write_atoms()

        for header, variable in self.headers_scalars.items():
            if getattr(self.geometry, variable) is not None:
                self._write_scalar(header,
                                getattr(self.geometry, 
                                        variable))
            else:
                print(f'Geometry object does not have a defined {variable}')

        for header, variable in self.headers_vectors.items():
            if getattr(self.geometry, variable) is not None:
                self._write_array(header,
                                getattr(self.geometry, 
                                        variable))
            else:
                print(f'Geometry object does not have a defined {variable}')


        with open(outputfile, 'w') as output:
            output.write(self.lines)

        return self.lines

def write_sith_data(sith_obj, outdir: str = './'):
    """Creates all the fchk files for each structure in a sith object
    
    Parameters
    ==========
    sith_obj: sith
        sith object with all the information stored in the attributes.
    outdir: str (optional)
        directory where to create the subdirectory "sith_data". Default=./
    
    Return
    ======
    (str) name of the subdir
    """
    if isinstance(outdir, (Path, str)):
            outdir = Path(outdir)
    assert outdir.exists(), "The path to the directory you " +\
        "specified does not exist does not exist"
    # avoid deleting previous data existing in the output directory
    subdir = outdir / 'sith_data'
    i = 1
    final_output = subdir
    while os.path.exists(final_output):
        final_output = Path(str(subdir) + '_' + "{:03d}".format(i))
        i += 1

    os.makedirs(final_output)

    for j, geometry in enumerate(sith_obj.structures):
        geowriter = WriteSITH(geometry)
        geowriter.write_file(outputfile=final_output / 'structure_{:03d}.fchk'.format(j))

    return final_output
