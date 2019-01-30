# MPI-GLTextureCube
MPI Enabled, Distributed Rendering Demo App

### Dependencies

* MPI ([MPICH](https://www.mpich.org/), [OpenMPI](https://www.open-mpi.org/), or [MS-MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi))
* [GLFW](https://www.glfw.org/)
* [glad](https://github.com/Dav1dde/glad/)

### Running

`mpiexec -np <N> ./bin/texturecube [imagecapture] [width] [height]`

* 1st command line option: `imagecapture` will flip view frustum of each rank and perform `glReadPixels()` to create a pixel buffer of the rendered image starting in the top-left corner. Any other value will result in normal rendering.
* 2nd command line option: overall width of rendered output. Default value is 1280.
* 3rd command line option: overall height of rendered output. Default value is 720.

### Example

`mpiexec -np 4 ./bin/texturecube NA 512 512`

![MPI GL Texture Cube with 4 ranks](https://tmarrinan.github.io/MPI-GLTextureCube/docs/MPI_GLTextureCube.png "MPI GL Texture Cube with 4 ranks")
