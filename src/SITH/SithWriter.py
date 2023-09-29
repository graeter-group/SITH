"""Contains methods which format, organize, and output data from SITH
objects"""
from typing import Tuple
import numpy as np
from ase.io import write
from SITH.SITH import SITH
from SITH.Utilities import Geometry, UnitConverter


# region: Write
def write_all(sith: SITH, filePrefix='') -> bool:
    """Write all SITH output files: summary, energies, error, and \u0394q in
    the parent directory of the reference structure.

    This function simply calls all SithWriter.write methods which take a SITH
    object as input. Units are Angstroms, degrees, and Hartrees.

    Args:
        sith (SITH):
            analyzed SITH object
        filePrefix (str, optional):
            prefix for file output. Defaults to ''.

    Returns:
        bool: If all files were successfully written.
    """
    return write_summary(sith, filePrefix) and \
        write_delta_q(sith, filePrefix) and \
        write_dof_energies(sith, filePrefix) and \
        write_error(sith, filePrefix)


def write_summary(sith: SITH, filePrefix='', includeXYZ=False) -> bool:
    """Write 'summary.txt' file of sith analysis  in the parent directory of
    the reference structure.

    Units are Angstroms, degrees, and Hartrees.

    Args:
        sith (SITH):
            analyzed SITH object
        filePrefix (str, optional):
            prefix for file output. Defaults to ''.
        includeXYZ (bool, optional):
            _description_. Defaults to False.

    Returns:
        bool: True if successful
    """
    totE = build_tot_energies(sith)
    dq = build_delta_q(sith)
    ric = build_dof_indices(sith)
    error = build_error(sith)
    expectedDE, errorDE, pErrorDE = calculate_error(sith)
    energies = build_dof_energies(sith)
    try:
        with open(sith._referencePath.parent.as_posix() +
                  sith._referencePath.root+filePrefix + "summary.txt",
                  "w") as s:
            s.write("Summary of SITH analysis\n")
            s.write("Redundant Internal Coordinate Definitions\n**" +
                    "Defined by indices of involved atoms**\n")
            s.writelines('\n'.join(ric))
            s.write("\nChanges in internal coordinates (Delta q)\n**" +
                    "Distances given in Angstroms, angles given in degrees**\n"
                    )
            s.writelines('\n'.join(dq))
            s.write("\n\n***********************\n" +
                    "**  Energy Analysis  **" +
                    "\n***********************\n")
            s.write("Overall Structural Energies\n")
            s.writelines('\n'.join(error))
            s.write("\nEnergy per DOF (RIC)\n")
            s.writelines("\n".join(energies))
            s.write("\nXYZ FILES APPENDED\n")
        if includeXYZ:
            write(sith._referencePath.parent.as_posix() +
                  sith._referencePath.root + filePrefix + "summary.txt",
                  sith.reference.atoms,
                  format='xyz', append=True, comment=sith.reference.name)
            for geometry in sith.deformed:
                write(sith._referencePath.parent.as_posix() +
                      sith._referencePath.root + filePrefix + "summary.txt",
                      geometry.atoms, format='xyz', append=True,
                      comment=geometry.name)
        return True
    except IOError as e:
        print(e)
        return False
    except Exception as e:
        print("Non-IO Exception encountered:")
        print(e)
        return False


def writeTotEnergiesString(sith: SITH, filePrefix="") -> True:
    """
    Writes the change in energy per structure to 'total_energies.txt'
     in the parent directory of the reference structure.

    Units are Hartrees. Format is (row:DOF, column:deformation structure).

    Args:
        sith (SITH):
            analyzed SITH object
        filePrefix (str, optional):
            prefix for file output. Defaults to ''.

    Returns:
        bool: True if successful.
    """
    try:
        lines = buildTotEnergiesString(sith)
        with open(sith._referencePath.parent.as_posix() +
                  sith._referencePath.root + filePrefix +
                  "totalStressEnergy.txt", "w") as dq:
            dq.writelines('\n'.join(lines))
            dq.write('\n')
        return True
    except IOError as e:
        print(e)
        return False
    except Exception as e:
        print("Non-IO Exception encountered:")
        print(e)
        return False


def write_delta_q(sith: SITH, filePrefix='') -> bool:
    """Writes the change in Redundant Internal Coordinates (RICs) per structure
    to 'delta_q.txt' in the parent directory of the reference structure.

    Units are Angstroms and degrees. Format is
    (row:DOF, column:deformation structure)

    Args:
        sith (SITH): extracted SITH object
        filePrefix (str, optional): prefix for file output. Defaults to ''.

    Returns:
        bool: True if successful.
    """
    try:
        dqPrint = build_delta_q(sith)
        with open(sith._referencePath.parent.as_posix() +
                  sith._referencePath.root + filePrefix+"delta_q.txt",
                  "w") as dq:
            dq.writelines('\n'.join(dqPrint))
            dq.write('\n')
        return True

    except IOError as e:
        print(e)
        return False
    except Exception as e:
        print("Non-IO Exception encountered:")
        print(e)
        return False


def write_error(sith: SITH, filePrefix='') -> bool:
    """Writes error information to 'error.txt' in the parent directory of the
    reference structure.

    Units are Hartrees and percentages.

    Args:
        sith (SITH): analyzed SITH object
        filePrefix (str, optional): prefix for file output. Defaults to ''.

    Returns:
        bool: True if successful
    """
    try:
        lines = build_error(sith)
        with open(sith._referencePath.parent.as_posix() +
                  sith._referencePath.root+filePrefix+'Error.txt',
                  "w") as dq:
            dq.writelines('\n'.join(lines))
            dq.write('\n')
        return True
    except IOError as e:
        print(e)
        return False
    except Exception as e:
        print("Non-IO Exception encountered:")
        print(e)
        return False


def write_dof_energies(sith: SITH, filePrefix='') -> bool:
    """Writes the energy in each degree of freedom (RIC) per deformed geometry.

    Writes to 'dof_energies.txt' in the parent directory of the reference
    geometry.

    Units are Hartrees.  Form is (row:DOF, column:deformation).

    Args:
        sith (SITH): extracted SITH object
        filePrefix (str, optional): prefix for file output. Defaults to ''.

    Returns:
        bool: True if successful
    """
    try:
        ePrint = build_dof_energies(sith)
        with open(sith._referencePath.parent.as_posix() +
                  sith._referencePath.root + filePrefix + 'dof_energies.txt',
                  "w") as dq:
            dq.writelines('\n'.join(ePrint))
            dq.write('\n')
        return True
    except IOError as e:
        print(e)
        return False
    except Exception as e:
        print("Non-IO Exception encountered:")
        print(e)
        return False


def write_xyz(geometry: Geometry):
    """Writes a .xyz file of the geometry

    Args:
        geometry (Geometry): Geometry to write
    """
    write(str(geometry._path.parent.as_posix()+geometry._path.root +
          geometry._path.stem+".xyz"), geometry.atoms, format='xyz',
          append=False)


def write_all_xyz(sith: SITH):
    """Writes .xyz files of all geometries

    Args:
        sith (SITH): SITH object from which to write all geometries
    """
    write_xyz(sith.reference)
    for deformation in sith.deformed:
        write_xyz(deformation)

# endregion


# region: Build
def build_tot_energies(sith: SITH) -> list:
    """Gathers and formats total stress energy of deformed geometries

    Units are Hartrees and percentages.

    Args:
        sith (SITH): analyzed SITH object

    Returns:
        list: strings containing stress energy of each deformed geometry.
    """
    assert sith.deformationEnergy is not None, \
        "SITH.analyze() has not been performed yet, no information available."
    lines = list()
    header = "            "
    for deformation in sith._deformed:
        header += "{: ^16s}".format(deformation.name)
    lines.append(header)
    lines.append("Stress Energy   " +
                 ''.join(["{: >16.6E}".format(e)
                          for e in sith.deformationEnergy[0]]))
    return lines


def build_delta_q(sith: SITH) -> list:
    """Gathers and formats the change in RICs (\u0394q) per deformed geometry

    Units are Angstroms and degrees. Format is (row:DOF, column: deformation).

    Args:
        sith (SITH): extracted SITH object

    Returns:
        list: strings containing the change in each degree of freedom (RIC) per
        deformed geometry.
    """
    """
    DOF Index       Deformation 1       Deformation 2       ...
    1               change              change              ...
    2               change              change              ...
    ...
    """
    assert sith.deltaQ is not None, \
           "SITH.extractData() has not been performed yet, no deltaQ " +\
           "information available."
    uc = UnitConverter()
    dqAngstroms = list()
    header = "DOF         "
    for deformation in sith._deformed:
        header += "{: ^16s}".format(deformation.name)
    dqAngstroms.append(header)
    dqAng = [uc.bohr_to_angstrom(dq)
             for dq in sith.deltaQ[0:sith._reference.dims[1], :]]
    dqAng = np.asarray(dqAng)
    for dof in range(sith._reference.dims[1]):
        if len(sith._deformed) > 1:
            line = "{: <12}".format(
                dof+1) + ''.join(["{: >16.6e}".format(dqa)
                                  for dqa in dqAng[dof, :]])
            dqAngstroms.append(line)
        else:
            line = "{: <12}{: >16.6e}".format(dof+1, dqAng[dof][0])
            dqAngstroms.append(line)
    dqDeg = np.degrees(sith.deltaQ[sith._reference.dims[1]:, :])
    dqDeg = np.asarray(dqDeg)
    for dof in range(sith._reference.dims[2]+sith._reference.dims[3]):
        if len(sith._deformed) > 1:
            line = "{:< 12}".format(dof+1+sith._reference.dims[1]) + ''.join(
                ["{: >16.6e}".format(dqd) for dqd in dqDeg[dof, :]])
            dqAngstroms.append(line)
        else:
            line = "{:< 12}{: >16.6e}".format(
                dof+1+sith._reference.dims[1], dqDeg[dof][0])
            dqAngstroms.append(line)
    return dqAngstroms


def build_dof_indices(sith: SITH) -> list:
    """Gathers and formats the DOF atom indices.

    Args:
        sith (SITH): extracted SITH object

    Returns:
        list: strings containing the atom indices involved in each degree of
        freedom.
    """
    """
    Takes in SITH object sith, Returns a list of strings containing the atom
    indices involved in each degree of freedom (DOF).
    """
    assert sith._reference.dimIndices is not None, \
           "SITH.extractData() has not been performed yet, " +\
           "no summary information available."
    return ["{: <12}".format(dof+1) + str(sith._reference.dimIndices[dof])
            for dof in range(sith._reference.dims[0])]


def build_dof_energies(sith: SITH) -> list:
    """Gathers and formats the energy in each DOF per deformed geometry.

    Units are Hartrees. Format is (row:DOF, column:deformation).

    Args:
        sith (SITH): analyzed SITH object

    Returns:
        list: strings containing the energy in each DOF (degree of freedom) per
        deformed geometry
    """
    """
    DOF Index       Deformation 1       Deformation 2       ...
    1               stress E            stress E            ...
    2               stress E            stress E            ...
    ...             ...                 ...                 ...
    """
    assert sith.energies is not None, \
           "SITH.energyAnalysis() has not been performed yet, no summary " +\
           "information available."
    eMat = list()
    header = "DOF         "
    for deformation in sith._deformed:
        header += "{: ^16s}".format(deformation.name)
    eMat.append(header)
    for dof in range(sith._reference.dims[0]):
        line = "{: <12}".format(dof+1) + \
               ''.join(["{: >16.6E}".format(e)
                       for e in sith.energies[dof, :]])
        eMat.append(line)
    return eMat


def build_error(sith: SITH) -> list:
    """Gathers and formats error data per deformed geometry.

    Units are Hartrees and percentages.

    Args:
        sith (SITH): analyzed SITH object

    Returns:
        list: strings containing error information per deformed geometry.
    """
    expected, error, pError = calculate_error(sith)
    lines = list()
    lines.append("{: <12s}{: ^16s}{: ^16s}{: ^12s}{: ^16s}".format(
        'Deformation', "\u0394E", "Expected \u0394E", "\u0025Error", "Error"))
    # "Deformation  \u0394E_Expected  \u0394E  \u0025Error  Error")
    for i in range(len(sith._deformed)):
        lines.append("{: <12s}{: >16.6E}{: >16.6E}{: >12.2f}{: >16.6E}".format(
            sith._deformed[i].name,
            sith.deformationEnergy[0, i],
            expected[0, i],
            pError[0, i],
            error[0, i]))
    return lines
#  endregion
