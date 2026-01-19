# T-DSM

Paper: 
T-DSM: Adaptive Distribution Shift Modeling for Temporal Knowledge Graph Reasoning

## Dependencies
* Python 3
* PyTorch 2.5.1
* PyYAML

## Usage
1. Unzip `data.zip`. 
2. Pretrain LSTM models `python3 LSTM_train.py` (this will generate a pretrained LSTM model file)
3. Pretrain: `python3 pretrain.py --pretrain_config=<model_name>` (this will generate a pretrained G or D model file)
4. T3DM train: `python3 train.py --g_config=<G_model_name> --d_config=<D_model_name>` (G and D models will be randomly initialized unless they are both pre-trained.)

Decrease `batch_size` in config files if you experience GPU memory exhaustion. (this would make the program runs slower, but would not affect the test result)

Feel free to explore and modify parameters in config files. Default parameters are those used in experiments reported in the paper.
