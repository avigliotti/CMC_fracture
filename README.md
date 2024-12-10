### Finite Element Program for Fracture Mechanics of Ceramic Matrix Composites

This static repository contains the Julia source code for a Finite Element program designed to analyze the fracture mechanics of ceramic matrix composites.

The code produced the results discussed in the paper:  
**"On the role of the matrix in the strength of carbon fiber-reinforced ceramics"**

If you use or find these codes useful, please cite this repository and the paper as:

```
@article{VIGLIOTTI2024105227,
  title = {On the role of the matrix in the strength of carbon fiber-reinforced ceramics},
  journal = {Mechanics of Materials},
  pages = {105227},
  year = {2024},
  issn = {0167-6636},
  doi = {https://doi.org/10.1016/j.mechmat.2024.105227},
  url = {https://www.sciencedirect.com/science/article/pii/S0167663624003193},
  author = {Andrea Vigliotti and Ferdinando Auricchio and Damiano Pasini},
  keywords = {Ceramics, Composites, Multiscale mechanics, Fracture, Phase field},
}
```

---

#### Description of Files

- **`fe_tools.jl`**  
  Includes the module for handling the phase field model.
  
- **`helper_funcs.jl`**  
  Contains helper functions for solving the Finite Element (FE) problem.
  
- **`solver.jl`**  
  Contains the main solver.
  
- **`mesh_files/`**  
  Directory containing the meshes of the models.
  
- **`AD4SM/`**  
  Directory containing the package for handling automatic differentiation and element generation.  
  To add this package via Julia's package manager, run:  
  ```
  ]add AD4SM
  ```  
  or  
  ```
  ]add https://github.com/avigliotti/AD4SM.jl
  ```
  
- **`qs_cm_hex1x1x1nL20_r500L1100lc0125lcf0500AT1nhl0500*.jl`**  
  Sample scripts to launch simulations for specified boundary conditions and models.

---

#### Solver Expectations

1. **Mesh Directory:**  
   The solver expects to find the mesh files in the directory named `mesh_files`. This can be changed by passing the relevant parameter to the solver.
   
2. **Output Directories:**  
   - **JLD2 Files:** The solver writes output files in JLD2 format to the `jld2_files` directory.  
   - **Paraview Files:** The solver writes Paraview-compatible files to the `vtk_files` directory.  
     
   If these directories are not present, they will be created automatically. Ensure the program has write privileges to these directories.

---

#### Julia Installation

Julia can be downloaded and installed from the [official website](https://julialang.org/downloads/).

---

