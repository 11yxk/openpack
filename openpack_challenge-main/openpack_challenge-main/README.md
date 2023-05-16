# openpack_challenge
Pytorch implementation of paper '' A Spatial-Temporal Graph Convolutional Networks-based Approach for the OpenPack Challenge 2022''. 
We won the 3rd place in the [OpenPack challenge 2022](https://open-pack.github.io/challenge2022).

# Get started
Please follow the instructions in [openpack-toolkit](https://github.com/open-pack/openpack-toolkit) and [openpack-torch](https://github.com/open-pack/openpack-torch)


# Data Preparation
Please download data from [OpenPack dataset](https://open-pack.github.io/release/v0-3-1).

# Training & Testing

### Training and Testing

For skeleton data:
Replace the path with your dataset path in 'config/ctr-gcn/configs/ctr-gcn.yaml'.
```
python main.py mode=train debug=false
python main.py mode=test debug=false
```

For sensors data:
Replace the path with your dataset path in 'config/TCN/configs/TCN.yaml'.
```
python main_acc_boundary.py mode=train debug=false
python main_acc_boundary.py mode=test debug=false
```
(Please remove ['U0202', 'S0300'] in openpack_toolkit/configs/datasets/splits.py from test set if have mismatch error.)

### Making a prediction file

```
python main.py mode=submission debug=false
python main_acc_boundary.py mode=submission debug=false
```
This will generate a '.pkl' file in 'v0.3.1/log/openpack-2d-kpt/your_issue_name/modality/save_scores'


- To ensemble the results of different modalities, replace the path with your dataset path in align.py and run:
```
python align.py
```
Final file will generate in results folder.

### Pretrained Models

- Pretrained Models are avaliable at https://drive.google.com/file/d/1v2bpcJtRofwi1ovl82DHrQlrD08WohO-/view?usp=share_link.


## Acknowledgements

This repo is based on [openpack-toolkit](https://github.com/open-pack/openpack-toolkit) and [openpack-torch](https://github.com/open-pack/openpack-torch).

Thanks original authors for their work!
