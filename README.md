# Algebraic Representations for Volumetric Frame Fields

## Introduction
This code includes algorithms for computing volumetric (octahedral and odeco) frame fields, described in detail in our paper:

> Palmer, D., Bommes, D., & Solomon, J. (2020). Algebraic Representations for Volumetric Frame Fields. ACM Transactions on Graphics (TOG), 39(2), 1-17.

## External Dependencies
- [Manopt](https://www.manopt.org) 5.0
- [Mosek](https://www.mosek.com) 9.0 ([C++ Fusion API](https://docs.mosek.com/9.0/cxxfusion/index.html#))
- Intel [TBB](https://github.com/intel/tbb)
- [MatPlotLib Colormaps](https://www.mathworks.com/matlabcentral/fileexchange/62729-matplotlib-2-0-colormaps-perceptually-uniform-and-beautiful)
- (Optional) [Eigen 3](https://eigen.tuxfamily.org)

## Installation
First, remember to build and install the Mosek Fusion API as described
[here](https://docs.mosek.com/9.0/cxxfusion/install-interface.html).

The following commands will compile all MEX files and add the code
to the MATLAB path.
```matlab
cd src/batchop
mexbuild /path/to/tbb/include
cd ../sdp
mexbuild /path/to/tbb/include /path/to/mosek/9.0
cd ../../ext/ray
mexbuild /path/to/eigen3 /path/to/tbb/include
cd ../..
install
```
## Usage
The main commands for computing fields are `MBO`, `OctaManopt`,
and `OdecoManopt`.

### Loading Models
Some tetrahedral meshes in `Medit` format
are included in the `meshes` directory for convenience.
To load a mesh, use
```matlab
mesh = ImportMesh('meshes/rockerarm_91k.mesh'); % Medit format
```
We also support meshes in `Tetgen` format:
```matlab
mesh = ImportMesh('path/to/file.node'); % Tetgen .node/.ele format
```

### Computing Frame Fields
The following commands compute octahedral and odeco fields by MBO
with random initialization:
```matlab
qOcta = MBO(mesh, OctaMBO, [], 1, 0);
qOdeco = MBO(mesh, OdecoMBO, [], 1, 0);
```
For modified MBO as described in our paper,
set the diffusion time multiplier and exponent as follows:
```matlab
qOcta = MBO(mesh, OctaMBO, [], 50, 3);
qOdeco = MBO(mesh, OdecoMBO, [], 50, 3);
```
The following lines compute octahedral and odeco fields by RTR with specified
initial fields. Drop the second argument for random initialization.
```matlab
qOcta = OctaManopt(mesh, qOcta);
qOdeco = OdecoManopt(mesh, qOdeco);
```
We have also included an implementation of the method of Ray et al. [2016]
in the `ext/ray` directory. To use it, invoke
```matlab
qRay = Ray(mesh);
```

### Visualization
To visualize an octahedral or odeco field, use `VisualizeResult`, which
plots the integral curves and singular structure, e.g.,
```matlab
VisualizeResult(mesh, qOdeco);
```
`PlotInterpolatedFrames` plots field-oriented cubes at specified sample
points:
```matlab
PlotInterpolatedFrames(q, mesh.tetra, samples)
```
where `samples` is a $k \times 3$ matrix of sample positions.

## Figures
We have included scripts for generating (MATLAB versions of) figures that appear in
the paper in the `figures/` directory.

- `EnergyTest` compares energy divergence behavior of octahedral and odeco fields,
as in Figure 12 in the paper.

- `PrismFigures` generates a figure similar to Figure 1 in the paper, showing scaling
behavior of an odeco field.

- `ConvergenceComparisons` generates figures like Figures 5 and 6 in the paper:
  ```matlab
  ConvergenceComparisons('../meshes', 'path/to/output/');
  ```

- `GenerateComparisons` generates a table like that in our supplemental document:
  ```matlab
  GenerateComparisons('../meshes', 'path/to/output/');
  ```

- To verify the exactness of SDP projection into the octahedral and odeco varieties,
respectively, execute
    ```matlab
    OctaExactnessTest(n);
    OdecoExactnessTest(n);
    ```
    for a sufficiently large value of `n`.
