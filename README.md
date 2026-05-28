## <a name="installation"></a> :wrench: Installation

This codebase was tested with the following environment configurations. It may work with other versions.

- Ubuntu 20.04
- CUDA 11.7
- Python 3.9
- PyTorch 2.0.1 + cu117

(**Note:** If you uses a newer cuda version, say 12.x, you may refer to the official github page of [causal_conv_1d](https://github.com/Dao-AILab/causal-conv1d/releases) and [mamba_ssm](https://github.com/state-spaces/mamba/releases/tag/v2.2.4) to find a matched version.)


The following give three possible solution to install the mamba-related libraries.

### Previous installation
To use the selective scan with efficient hard-ware design, the `mamba_ssm` library is needed to install with the folllowing command.

```
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```
One can also create a new anaconda environment, and then install necessary python libraries with this [requirement.txt](https://drive.google.com/file/d/1SXtjaYDRN53Mz4LsCkgcL3wV23cOa8_P/view?usp=sharing) and the following command: 
```
conda install --yes --file requirements.txt
```
### Train on SR
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 basicsr/train.py -opt SMoE_x2.yml --launcher pytorch
```
### Test on SR
```
python basicsr/test.py -opt test_SMoE_x2.yml
```
# hfdmmamba
