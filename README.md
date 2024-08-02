# Super Mario Bros Reinforcement Learning

Let's create an AI that's able to play Super Mario Bros! We'll be using Double Deep Q Network Reinforcement Learning algorithm to do this.

Watch the accompanying YouTube video [here](https://youtu.be/_gmQZToTMac)! Hope you enjoy it!

## Installation

### First, clone this repository

```bash
git clone https://github.com/Sourish07/Super-Mario-Bros-RL.git
```

### Next, create a virtual environment

The command below is for a conda environment, but use whatever you're comfortable with. I'm using Python 3.10.12.

```bash
conda create --name smbrl python=3.10.12
```

Make sure you activate the environment.

```bash
conda activate smbrl
```

### Finally, install dependencies

```bash
cd Super-Mario-Bros-RL
pip install -r requirements.txt
```

### Check if PyTorch is GPU accelerated

If the following command in your terminal with the virtual environment activated returns `True`, then PyTorch is GPU accelerated.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### If issues with GPU support arise, manually install PyTorch v2.1.1

The steps here will be a little different for everyone depending on if you're using a GPU or not.

If you are using a GPU, it also depends on what version of CUDA you're using (assuming you're using an NVIDIA card). I'm using CUDA 12.1, so I have to go to PyTorch's website and then install PyTorch v2.1.1 for CUDA version 12.1.

For more information, please go to [PyTorch's website](https://pytorch.org/get-started/locally/).

My command looked like:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
