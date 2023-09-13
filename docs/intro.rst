.. SITH Modules documentation

Introduction
================================

SITH, or Stress Induced Traversal of Hessian, is an analysis package for python intended
for the decomposition of a molecular geometry's stress energy into its respective internal degrees
of freedom.


Abbreviation Glossary
================================
RIC
   Redundant Internal Coordinate
   
DOF
   Degree of Freedom
   In SITH, DOFs are represented by their respective atomic indices.
   Bond Lengths:    (atom1, atom2)
   Bond Angles:     (atom1, atom2, atom3)
   Dihedral Angles: (atom1, atom2, atom3, atom4)
   where atomX is the index of a particular atom in a geometry.
