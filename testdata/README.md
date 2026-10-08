# Test data

## nickel/

`patterns_v2.h5` is kikuchipy's `nickel_ebsd_large`: 55 × 75 patterns of 60 × 60 pixels (8 bit) of
polycrystalline nickel, acquired on a NORDIF UF-1100 detector at 20 kV with the sample tilted 70°,
in kikuchipy's h5ebsd format, with a static background and the pattern centres and orientations
kikuchipy refined for it.

- Source: <https://github.com/pyxem/kikuchipy-data/tree/master/nickel_ebsd_large>
- Authors: Håkon Wiik Ånes and the kikuchipy developers (the scan was acquired at NTNU)
- License: Creative Commons Attribution 4.0 International ([LICENSE](nickel/LICENSE)); the file is unchanged

`reference_quaternions.f32` holds the file's own refined orientations (its Bunge angles) as
4125 × 4 little-endian float32 quaternions (w, x, y, z), so the tests and the browser test can
compare against them without an HDF5 library.

`Ni_master_20kV.npz` is a dynamical master pattern of nickel at 20 kV simulated with
[ebsdsim](https://github.com/ZacharyVarley/ebsdsim) from its bundled `Ni.cif` (501 × 501 per
hemisphere, stored as the fundamental sector of m-3m).

In both master patterns' metadata the path of the CIF, which ebsdsim records in full, is cut down to the file's name.

The app's Example button loads this scan and this master pattern.

## gan/

`GaN_master_20kV.npz` is a master pattern of wurtzite GaN at 20 kV simulated with ebsdsim from its
bundled `GaN.cif` (301 × 301 per hemisphere, point group 6mm): a hexagonal crystal without an
inversion centre, as the second candidate phase in the tests and the second tutorial.
