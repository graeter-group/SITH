# SITH
Stress Induced Traversal of Hessian analysis code

A redesign and simplification of the JEDI approach outlined in https://aip.scitation.org/doi/abs/10.1063/1.4870334.

API is documented and examples are attached, but full documentation of the SITH package is still active, albeit slowly.
Thank you for your patience and please do not hesitate to direct any questions to the maintainer at mfarrugi@nd.edu, daniel.sucerquia@h-its.org.

## Version 2.0
- Every energy distribution analysis method is a different module in the package.
- using sith is now divided in two steps. Reading and computing QM data by SITH.readers. Reading is done to allow any DFT software to do the computation. So far, only g09 files are included, but the idea is to include other sofwares (QChem, ORCA and GPAW).
- fchk reader methods are now in SITH.g09_reader. This will allow users of others softwares add their own readers.
- in SITH.g09_reader.FileReader, every block defined by ..._header is read by _fill_aray method or directly in one line (this avoided long if else blocks).
- SITH.Geometry.dim_indices.shape = (#dofs, 4) and dim_indices is now an numpy array. Distances, that only have two indices, the rest are zeros (remember g09 counts from 1). indeed all the arrays have that shape.
- Geometry.build_RIC was completely removed. This is done by readers now.
- Lmatrix class is not necessary anymore, completely removed and replaced by _hessian_reader in g09_reader
- UnitConverter class in .Utilities is not necessary, completely removed and replaced by ase.units
- all attributes called "._deformed" renamed as "structures"

Possibility:
change energy units from Hartrees to eV

# TODO
remove print and replace with logging