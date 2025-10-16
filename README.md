## ASU AI For Business Analytics Assistant

A sophisticated assistant leveraging the Qwen2.5-7B-Instruct model to deliver human-like negotiation capabilities for business analytics applications.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
  - [1. Server Access](#1-server-access)
  - [2. Resource Allocation](#2-resource-allocation)
  - [3. Environment Setup](#3-environment-setup)
  - [4. Project Installation](#4-project-installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Overview

This project develops an advanced AI-powered chatbot designed for business analytics assistance. Built on the Qwen2.5-7B-Instruct model, the system demonstrates sophisticated natural language processing capabilities with a focus on human-like negotiation behavior.

**Project Goals:**

- Provide intelligent business analytics assistance
- Implement natural, human-like conversation patterns
- Demonstrate advanced AI negotiation capabilities
- Serve as a foundation for enterprise-level AI assistants

## Setup

### 1. Server Access

**Connect to VPN and SSH into the server:**

```bash
ssh <your_asu_id>@sol.asu.edu
```

> **Need help?** Check out the [ASU Research Computing guide](https://asurc.atlassian.net/wiki/spaces/RC/pages/2319417345/A+Brief+Example#Step-3---Use-/-Test) for detailed setup instructions.

### 2. Resource Allocation

**Request an interactive session with GPU access:**

```bash
# For new sessions
interactive -p htc -t 2:00:00 --gres=gpu:a100:1

# If already in an interactive session
salloc -p htc -t 2:00:00 --gres=gpu:a100:1
```

> **Resource Limits**: HTC partition provides up to 240 minutes with a single A100 GPU

### 3. Environment Setup

**Load required modules:**

```bash
# Load package manager
module load mamba/latest

# Load CUDA drivers
module load cuda-12.6.1-gcc-12.1.0
```

**Create and activate virtual environment:**

```bash
# Create environment with Python 3.13
mamba create -n venv python=3.13

# Activate environment
source activate venv
```

**Install core dependencies:**

```bash
mamba install -c conda-forge accelerate transformers -c pytorch
```

> **Package Management**: Use `mamba` for package installation instead of `pip`. Find packages at [anaconda.org](https://anaconda.org/) using the format: `mamba install -c <channel> <package>`

### 4. Project Installation

**Clone and set up the project:**

```bash
# Clone repository
git clone https://github.com/dhruvb26/cse485-capstone.git

# Navigate to project directory
cd cse485-capstone
```

## Usage

**Start the chatbot:**

```bash
python main.py
```

This launches the interactive chat interface with the Qwen2.5-7B-Instruct model, enabling you to engage in conversations and test the negotiation capabilities.

## Troubleshooting

### Common Issues

**Connection Problems:**

- Ensure Cisco VPN is active before SSH attempts
- Verify your ASU credentials are correct

**Resource Allocation:**

- If GPU allocation fails, try requesting during off-peak hours
- Check current cluster usage with `squeue`

**Environment Issues:**

- Verify CUDA module is loaded: `module list`
- Confirm virtual environment activation: `which python`

**Package Installation:**

- Use `mamba` instead of `pip` for conda packages
- Check package availability on anaconda.org before installation

### Getting Help

- ASU Research Computing Support: [RC Documentation](https://asurc.atlassian.net/wiki/spaces/RC)
- Project Issues: Create an issue in the repository
- Technical Questions: Consult course materials or instructor

**Course**: CSE 485 - Capstone Project 1</br>
**Institution**: Arizona State University
