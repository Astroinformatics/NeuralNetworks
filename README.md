# Introduction to Neural Networks Lab
### led by [Prof. Hyungsuk Tak](https://hyungsuktak.github.io/)
### Astroinformatics Summer School 2022 
### Organized by [Penn State Center for Astrostatistics](https://sites.psu.edu/astrostatistics/)

-----
This repository contains one computational notebook: 
- neuralnetwork_intro.jl ([Pluto notebook](https://astroinformatics.github.io/NeuralNetworks/neuralnetwork_intro.html)):  Provides an introduction to neural networks using classification of high-redshift quasars as an example

The lab does not assume familiarity with Julia.  While it can be useful to "read" selected portions of the code, the lab tutorials aim to emphasize understanding how algorithms work, while minimizing need to pay attention to a language's syntax.

---

## Running Labs
Instructions will be provided for students to run labs on AWS severs during the summer school.  
Others may install Julia and Pluto on their local computer with the following steps:
1.  Download and install current version of Julia from [julialang.org](https://julialang.org/downloads/).
2.  Run julia
3.  From the Julia REPL (command line), type
```julia
julia> using Pkg
julia> Pkg.add("Pluto")
```
(Steps 1 & 3 only need to be done once per computer.)

4.  Start Pluto
```julia
julia> using Pluto
julia> Pluto.run()
```
5.  Open the Pluto notebook for your lab

---
## Additional Links
- [GitHub respository](https://github.com/Astroinformatics/SummerSchool2022) for all of Astroinformatics Summer school
- Astroinformatics Summer school [Website & registration](https://sites.psu.edu/astrostatistics/astroinfo-su22/)

## Contributing
We welcome people filing issues and/or pull requests to improve these labs for future summer schools.
