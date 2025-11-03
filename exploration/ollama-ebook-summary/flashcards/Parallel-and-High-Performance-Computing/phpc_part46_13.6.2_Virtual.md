# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 46)

**Starting Chapter:** 13.6.2 Virtual machines using VirtualBox

---

#### Docker Containers Overview
Docker containers provide a lightweight and portable way to package software. They are particularly useful for developers who need an isolated environment for their applications without having to manage the underlying infrastructure fully.

:p What is the primary advantage of using Docker containers mentioned?
??x
The primary advantage of using Docker containers is that they offer a lightweight and portable environment, making it easier to develop, test, and deploy applications consistently across different environments. This isolation ensures that dependencies are encapsulated within the container, which can significantly reduce setup time and improve consistency.

Docker containers also allow for rapid deployment and scaling since they abstract away much of the underlying infrastructure management.
x??

---

#### NVIDIA Docker Container
NVIDIA provides a prebuilt Docker container that supports GPU acceleration. It is available at https://github.com/NVIDIA/nvidia-docker/ for up-to-date instructions on setting it up.

:p What site can you visit to get started with using NVIDIA's prebuilt Docker container?
??x
Visit the site at <https://github.com/NVIDIA/nvidia-docker/> to get started with using NVIDIA's prebuilt Docker container. This site provides detailed instructions and resources for integrating GPU acceleration into your Docker images.
x??

---

#### ROCm Docker Containers
For ROCm, there are comprehensive documentation on Docker containers available at https://github.com/RadeonOpenCompute/ROCm-docker/. These containers provide support for AMD GPUs.

:p Where can you find extensive instructions for using ROCm with Docker?
??x
You can find extensive instructions for using ROCm with Docker at <https://github.com/RadeonOpenCompute/ROCm-docker/>. This site offers detailed documentation and resources to help integrate ROCm into your Docker setup, enabling GPU acceleration on AMD hardware.
x??

---

#### Intel OneAPI Containers
Intel provides containers for setting up their oneAPI software. The relevant site is available at <https://github.com/intel/oneapi-containers/>. Some of these base containers are large and require a good internet connection.

:p Where can you find resources to set up Intel's oneAPI software in Docker containers?
??x
You can find resources to set up Intel's oneAPI software in Docker containers at <https://github.com/intel/oneapi-containers/>. This site provides instructions and documentation for integrating Intel's tools into Docker environments. Note that some of the base containers are quite large, so a good internet connection is recommended.
x??

---

#### PGI Compilers in Containers
The PGI compilers are essential for developing OpenACC code and other GPU-related applications. You can find the container site at <https://ngc.nvidia.com/catalog/containers/hpc:pgi-compilers>.

:p Where can you get a Docker container with the PGI compiler installed?
??x
You can get a Docker container with the PGI compiler installed from the NVIDIA NGC catalog at <https://ngc.nvidia.com/catalog/containers/hpc:pgi-compilers>. This site offers detailed instructions on how to use these containers for GPU development.
x??

---

#### Virtual Machines Using VirtualBox
Virtual machines (VMs) allow users to create a guest OS within their host system. They provide a more restrictive environment than Docker containers but are easier to set up GUI applications and may have limitations in accessing the GPU.

:p What is a key difference between VMs and Docker containers when it comes to setting up graphical user interfaces (GUIs)?
??x
A key difference between VMs and Docker containers is that VMs typically provide better support for setting up graphical user interfaces (GUIs) compared to Docker containers. This is because VMs run a full operating system, which can handle GUI applications more naturally.

However, VMs may have limitations in accessing the GPU for computation. While some GPU languages allow running on the host CPU processor, direct GPU access from a VM is often difficult or impossible.
x??

---

#### Setting Up Ubuntu Guest OS in VirtualBox
To set up an Ubuntu guest operating system in VirtualBox, you need to download and install VirtualBox, then create a new virtual machine (VM) with settings appropriate for your needs.

:p How do you start the process of setting up an Ubuntu guest OS in VirtualBox?
??x
To start the process of setting up an Ubuntu guest OS in VirtualBox:

1. Download and install VirtualBox from its official site.
2. Create a new virtual machine by clicking "New" in VirtualBox, naming it (e.g., chapter13), selecting Linux as the type, and choosing Ubuntu 64-bit as the version.
3. Allocate memory to the VM and create a fixed-size virtual hard disk with at least 50 GB.

After setting up the VM, you can proceed to install Ubuntu by clicking "Start" in VirtualBox, selecting the `ubuntu-20.04-desktop-amd64.iso` file, installing it, and following the on-screen instructions.
x??

---

#### Cloud Computing Overview
Cloud computing refers to servers provided by large data centers, which can be accessed via various cloud providers. These services are useful when specific hardware resources like GPUs are limited or unavailable on local machines. Some cloud providers cater specifically towards High Performance Computing (HPC) needs.

:p What is the primary purpose of using cloud computing in the context of this chapter?
??x
The primary purpose of using cloud computing is to access hardware resources, such as GPUs, that may not be available locally, allowing for exploration and experimentation with parallel computing applications.
x??

---

#### Google Cloud Platform (GCP) Setup
Google offers a Fluid Numerics Cloud cluster on the GCP which has Slurm batch scheduler and MPI capabilities. NVIDIA GPUs can also be scheduled through this setup. The process to get started involves navigating to specific URLs provided by the Fluid Numerics site.

:p How do you start using the Google Cloud Platform for HPC tasks?
??x
To start using the Google Cloud Platform for HPC tasks, follow these steps:
1. Visit the Fluid Numerics Cloud cluster URL: <https://mng.bz/Q2YG>
2. Follow the instructions provided on the site to set up your environment.
3. Note that the process can be complex and may require patience.
x??

---

#### Installing Basic Build Tools
To prepare an Ubuntu virtual machine for downloading and installing software, you need to install basic build tools using specific commands.

:p What are the commands needed to install essential build tools on an Ubuntu system within a VirtualBox?
??x
The commands needed to install essential build tools on an Ubuntu system within a VirtualBox are as follows:
```sh
sudo apt install build-essential dkms git -y
```
These commands will ensure that you have the necessary tools installed for compiling and managing dependencies.
x??

---

#### Configuring VirtualBox Settings
Configuring the VirtualBox settings properly ensures seamless file transfer and application of guest additions, which are essential for running applications smoothly.

:p What steps are required to configure a VirtualBox environment for better integration with the host system?
??x
To configure a VirtualBox environment for better integration with the host system, follow these steps:
1. Make the VirtualBox window active.
2. Select the Devices pull-down menu from the window’s menus at the top of the screen.
3. Set the Shared Clipboard option to Bidirectional.
4. Set the Drag and Drop option to Bidirectional.
5. Install the guest additions by selecting the menu option `virtualbox-guest-additions-iso`.
6. Remove the optical disk: from the desktop, right-click and eject the device or in the VirtualBox window, select Devices > Optical Disk and remove the disk from the virtual drive.
7. Reboot and test by copying and pasting (copy on the Mac is Command-C and paste in Ubuntu is Shift-Ctrl-v).
x??

---

#### Running Shallow Water Application
After setting up the environment, you can clone the repository for Chapter 13, navigate to the directory, and run the provided scripts to build and run the shallow water application.

:p How do you set up and run the shallow water application on a virtual machine?
??x
To set up and run the shallow water application on a virtual machine, follow these steps:
```sh
git clone --recursive https://github.com/essentialsofparallelcomputing/Chapter13.git
cd Chapter13
sh -v README.virtualbox
```
The `README.virtualbox` file contains commands to install software and build/run the shallow water application. Real-time graphics output should also work.
x??

---

#### Profiling with nvprof Utility
You can use the `nvprof` utility to profile the shallow water application, providing insights into performance bottlenecks.

:p How do you profile the shallow water application using the `nvprof` utility?
??x
To profile the shallow water application using the `nvprof` utility, run the following command after building and running the application:
```sh
nvprof ./your_shallow_water_executable
```
This will provide detailed performance metrics that can help identify any bottlenecks in the code.
x??

---

#### OneAPI Initiative and Intel GPUs
Background context: Intel has set up a cloud service for testing out their GPUs as part of their oneAPI initiative. This initiative includes access to both software and hardware, specifically through their DPCPP compiler which provides SYCL implementation.

:p What is the oneAPI initiative by Intel?
??x
The oneAPI initiative by Intel aims to provide developers with a unified programming model for heterogeneous systems, including CPUs and GPUs. It offers tools and compilers like DPCPP (oneAPI Data Parallel C++), which supports SYCL (Standard for Parallel Algorithms). Developers can access this via the cloud service set up by Intel.

This initiative is designed to simplify the development process across different hardware architectures.
??x
---

#### Registering for the Cloud Service
Background context: To use the cloud service for testing Intel GPUs, developers need to register at https://software.intel.com/en-us/oneapi. This service allows access to both software and hardware necessary for oneAPI development.

:p How do developers get access to the cloud service for testing Intel GPUs?
??x
Developers can register on the official website provided by Intel: <https://software.intel.com/en-us/oneapi>. After registration, they will gain access to the cloud environment where they can test and develop their applications using Intel's GPU resources.
??x
---

#### Customization for Development Environments
Background context: The examples in this chapter may require customization based on specific hardware configurations. Setting up development systems for GPU computing is challenging due to the variety of possible hardware setups.

:p Why might developers need to customize the examples from this chapter?
??x
Developers might need to customize the examples because they are likely tailored to a specific hardware configuration, and different systems may have varying requirements. Customization ensures that the code runs optimally on their particular setup.
??x
---

#### Docker Containers for Development
Background context: Using pre-built Docker containers can be easier than manually configuring and installing software on individual systems, especially given the complexity of setting up development environments for GPU computing.

:p How can developers simplify the setup process using Docker?
??x
Developers can use pre-built Docker containers to simplify the setup process. These containers encapsulate all necessary dependencies and configurations, making it easy to replicate a consistent development environment without manually installing software.
??x
---

#### NVIDIA Profiling Tools
Background context: The chapter mentions that tools and workflows in GPU programming are rapidly evolving. For NVIDIA GPUs, there are various profiling tools available, including NSight Compute.

:p What are some resources for NVIDIA's profiling tools?
??x
NVIDIA provides resources such as the NVIDIA NSight Guide at <https://docs.nvidia.com/nsight-compute/Nsight-Compute/index.html#nvvp-guide> and a comparison of their profiling tools at <https://devblogs.nvidia.com/migrating-nvidia-nsight-tools-nvvp-nvprof/>. These resources can help developers understand and use NVIDIA's profiling tools effectively.
??x
---

#### Other Tools for GPU Development
Background context: In addition to NVIDIA tools, there are other tools available such as CodeXL (now part of the GPUopen initiative) and various GPU cloud services.

:p What other tools are mentioned for GPU development?
??x
Other tools include:
- **CodeXL**: An open-source tool released under the GPUopen initiative.
- **PGI Compilers on NVIDIA GPU Cloud**: Resources available at <https://ngc.nvidia.com/catalog/containers/hpc:pgi-compilers>.
- **AMD Tools**: AMD has also updated their tools to be cross-platform, with information available on setting up virtualization environments and containers.

These tools can provide additional support for developers working on different hardware platforms.
??x
---

