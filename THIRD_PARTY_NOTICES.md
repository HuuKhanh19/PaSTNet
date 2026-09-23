# Third-party code

## SchNet-GP

The scaffold splitter is copied byte-for-byte from
[`src/data/splitter.py`](https://github.com/HuuKhanh19/SchNet-GP/blob/9cef51083d02b4c0c7c53fd6e99d818ef269b13c/src/data/splitter.py)
at commit `9cef51083d02b4c0c7c53fd6e99d818ef269b13c` of
[HuuKhanh19/SchNet-GP](https://github.com/HuuKhanh19/SchNet-GP).
It is stored in `pastnet/_vendor/schnet_gp_splitter.py`.

`pastnet/_vendor/schnet_gp_preprocessing.py` extracts `validate_smiles`,
`target_column_names`, and `preprocess_dataframe` unchanged from that commit's
`src/data/data_loader.py`. The upstream splitter credits MolHFCNet.

`pastnet/model/schnet.py` derives from the same commit's `src/models/schnet.py`.
It retains the continuous-filter interactions and initialization order, exposes
only atom embeddings, and uses cached complete directed radius graphs with a
10 Å cutoff and no neighbor cap. The original model source has SHA-256
`04ec32ba471247c7885d027b1cd5733130400ad88c97ee6d9c347ad4bcd3cd58`.
The pinned SchNet-GP tree does not contain a separate license file.

## Path Complex Neural Network

The atom-order convention and bond descriptors originate from
[LongLee220/Path-Complex-Neural-Network](https://github.com/LongLee220/Path-Complex-Neural-Network),
copyright (c) 2025 LongLee220, under the MIT license retained in [LICENSE](LICENSE).
Only the encodings needed by PaSTNet are included; the PCNN model is not bundled.
The 92-dimensional CGCNN atom descriptors are provided by the `jarvis-tools`
dependency. PyTorch Geometric supplies the SchNet message-passing primitive.

Dataset files and pretrained weights are not distributed by this repository.
Their original terms continue to apply.
