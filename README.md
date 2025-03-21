# Polaris
Yet Another High Level AI Simulator

## Introduction
*polaris* is a high level simulator for performance analysis of AI architectures. It takes as input an *AI Workload* and an *Architecture Configuration*. It represents the input workload into an in-memory directed acyclic graph (DAG) data structure, where each node represents a *computation* or a *communication* operator, and each edge represents a *dataflow*. The graph data structure represents an *intermediate representation* (IR) for the Simulator virtual machine (VM). We can execute various graph transformations, and eventually schedule the DAG on a *backend* for performance analysis.

## Environment Setup

The recommended setup uses python with the miniforge installation manager. It is expected that the reader is familiar 
with conda environment, creating and switching between environments. One can familiarize oneself with these concepts at
https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html. 

Note that we use the **miniforge** installer, instead of anaconda or miniconda installers.
As described in https://www.anaconda.com/blog/is-conda-free, 

> conda, the software program for package and environment management, is free to use by anyone.  It’s open source, too.
>
> The conda-compatible packages in conda-forge, Bioconda, and almost all other publicly accessible channels are free to use by any one.
>
> The conda-compatible packages in the default channel and Anaconda Distribution are free to use if:
>
> - Your organization has less than 200 people, or
>
> - Your organization has 200 or more people, but qualifies as an exempt organization in Anaconda’s terms of service:
>
> Students and educational entities may use our free offerings in curriculum-based courses.

**Consequently, we should use only the conda-forge channel.** 
Miniforge installer pre-configures conda-forge channel as the default and only channel, and hence we use the miniforge
installer

1. Install miniforge as described in https://github.com/conda-forge/miniforge. 
   * **Instructions as of 17-Feb-2025**
   * Run `curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"`
   * Download and execute as a (ba)sh script
   * Update conda by running: `conda update -n base -c conda-forge conda`
2. Once miniforge is installed, run the command `conda env create --file environment.yml`. The conda environment will
be created with the name 'polaris'. If you wish to provide a different name to the environment, run the command `conda env create --file environment.yml --name <name-of-your-choice>` instead.
3. Run `conda activate polaris`.
