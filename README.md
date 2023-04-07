<!--
 * @Author: Johannes Liu
 * @LastEditors: Johannes Liu
 * @email: iexkliu@gmail.com
 * @github: https://github.com/johannesliu
 * @Date: 2023-04-06 14:38:48
 * @LastEditTime: 2023-04-07 13:53:32
 * @motto: Still water run deep
 * @Description: Modify here please
 * @FilePath: \Layer-enhanced-Knowledge-Aggregation-Networks\README.md
-->
# LeKAN 

## Introduction 
Source codes for Johannes's Paper "LeKAN: Extracting Long-tail Relations via Layer-Enhanced Knowledge-Aggregation Networks".

## Project Structrue
* /baseline results of baseline models
* /data: Processed training data, the wordvec files and so on.
* /lekankit: Lekan Repository。
* /scripts/inital.py: Generate the *.npy files from /raw_data.
* /scripts/transE.py: obtain the TransE embeddings.
* /scripts/main.py: run the training program.
* /raw_data: Store the original data.
* /embeddings/GetEmbeddings: obtain the embedding vectors.
* /embeddings/Visualization.ipynb: obtain the visual graph.

## Execution Process
1. run /scripts/inital.py to obtain the processed data.
2. run /scripts/transE.py to obtian TranseE Embeddings
3. run /embeddings/GetEmbeddings to obtain the final embeddings.
4. run /scripts/main.py to train and eval. The models are stored in /outputs/ckpt，and the results are stored in /outputs/logits

## Dataset
Google Drive: https://drive.google.com/file/d/14MFhEdTd46V0rbNrUh6Ygx9NUM4nGIaX/view?usp=sharing

## Requirements
* Pytorch 1.8.2
* dgl
* scikit-learn

## Cite
```bib
@inproceedings{liu2022lekan,
  title={Lekan: Extracting long-tail relations via layer-enhanced knowledge-aggregation networks},
  author={Liu, Xiaokai and Zhao, Feng and Gui, Xiangyu and Jin, Hai},
  booktitle={Database Systems for Advanced Applications: 27th International Conference, DASFAA 2022, Virtual Event, April 11--14, 2022, Proceedings, Part I},
  pages={122--136},
  year={2022},
  organization={Springer}
```