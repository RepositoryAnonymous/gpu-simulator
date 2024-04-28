
The source code for the simulator presented in our paper (Submission #49) at the "MICRO 2024" conference.

## Paper Details

- Submission ID: #49
- Title: 
- Conference: MICRO 2024

## Prerequisites

- C++ Compiler (g++), the version we use is `g++ (GCC) 7.3.1 20180303 (Red Hat 7.3.1-5)`.
  All compilers that support the C++11 standard will do well. You can install by this way:
  ```shell
  yum install g++
  ```
- Make, the version we use is `GNU Make 4.2.1`. You can install by this way:
  ```shell
  yum install make
  ```
- Python, the version we use is `Python 3.7.0`. We recommend you to use Anaconda to manage
  Python packages, and the guide to install Anaconda is available
  [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installing-conda).
- MPICH, the version of MPICH we use is `v3.3.2` and it is available
  [here](https://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz).
  Download the source code, uncompress the folder, and change into the MPICH directory.
  ```shell
  wget https://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz
  tar -xzf mpich-3-3.2.tar.gz
  cd mpich-3-3.2
  ```
  After doing this, you should be able to configure your installation by performing `./configure`.
  ```shell
  ./configure
  ```
  If you need to install MPICH to a local directory (for example, if you don't have root access
  to your machine), type `./configure --prefix=/path/to/your/path`. 
  When the configuration is done, build and install MPICH with `make && sudo make install`.
  After installing MPICH, you also need to set the environment variable:
  ```shell
  export MPI_HOME=/path/to/mpich-3.3.2
  export PATH=$PATH:$MPI_HOME/bin
  export MANPATH=$MANPATH:$MPI_HOME/man
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_HOME/lib
  ```
  After this, you should be able to type `mpiexec --version` or `mpirun --version` and see the
  version information of MPICH.
- Boost Library, the version we use is `v1.83.0`, and it is available
  [here](https://boostorg.jfrog.io/artifactory/main/release/1.83.0/source/boost_1_83_0.tar.gz).
  Download the source code, uncompress the folder, and change into the Boost directory.
  ```shell
  wget https://boostorg.jfrog.io/artifactory/main/release/1.83.0/source/boost_1_83_0.tar.gz
  tar -xzf boost_1_83_0.tar.gz
  cd boost_1_83_0
  ```
  After doing this, you should be able to compile the Boost library.
  ```shell
  ./bootstrap.sh –prefix=/path/to/boost
  ```
  Where `/path/to/boost` is the path you want to install Boost in.
  When the compilation is done, install Boost with `sudo ./b2 install`， and this will allow you
  to install the Boost library in the `/path/to/boost` directory.
  After installing, you also need to set the environment variable:
  ```shell
  export BOOST_HOME=/path/to/boost
  export CPLUS_INCLUDE_PATH=$BOOST_HOME/include:$CPLUS_INCLUDE_PATH
  export C_INCLUDE_PATH=$BOOST_HOME/include:$C_INCLUDE_PATH
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BOOST_HOME/lib
  ```

## Building the Simulator

1. Clone the repository:
   ```shell
   git clone https://github.com/RepositoryAnonymous/gpu-simulator.git gpu-simulator
   ```

2. Change to the project directory:
   ```shell
   cd gpu-simulator
   ```

3. Build the simulator:
   ```shell
   make -j
   ```

The compiled simulator executable is called `gpu-simulator.x`, and is just in the current directory.

## Clone the Benchmark Suit 

The benchmark application suit we used for validation consisted of 43 applications with a total 
of 1784 kernels. We evaluate them on a real NVIDIA QUADRO GV100. These applications are from the 
cuBLAS library, the heterogeneous computing benchmark suite PolyBench and Rodinia, the fluid 
dynamics benchmark LULESH, the deep learning basic operator benchmark suite DeepBench, the DNN 
benchmark suite Tango, the unstructured mesh mini-app benchmark PENNANT. For all selected 
applications, we select their first 100 kernels for evaluation. All workloads are compiled using 
CUDA 11.8 with the compute capability `sm70` for the Volta architecture. You can clone this 
benchmark suit we used from [here](https://github.com/RepositoryAnonymous/simulator-apps).

1. Change to the simulator project directory:
   ```shell
   cd gpu-simulator
   ```

2. Clone the benchmark suit repository:
   ```shell
   git clone https://github.com/RepositoryAnonymous/simulator-apps.git apps
   ```

For guidance on compiling these applications and getting their traces on real hardware, please 
refer to [here](https://github.com/RepositoryAnonymous/simulator-apps/blob/main/README.md). And this repo 
also provides the guide to obtain experimental results and reproduction images of our paper.

## Running the Simulator

To run the GPU simulator, use the following command:
```shell
mpirun -np [num of processes] ./gpu-simulator.x 
  --configs [/path/to/application/configs] 
  --kernel_id [kernel id you want to evaluate] 
  --config_file ./DEV-Def/QV100.config
```

The options:
- `-np <num>`: Number of processes.
- `--configs <path>`: Specify the configuration file of the application
- `--kernel_id <num>`: Specify the kernel ID for simulation

Example:
```shell
mpirun -np 10 ./gpu_simulator --configs ./apps/Rodinia/hotspot/configs --kernel_id 0 --config_file ./DEV-Def/QV100.config
```

Here, the `./apps/Rodinia/hotspot/configs` are generated by `tracing_tool`. For detailed instructions, please refer to 
[here](https://github.com/RepositoryAnonymous/gpu-simulator/blob/main/tracing-tool/README.md).
