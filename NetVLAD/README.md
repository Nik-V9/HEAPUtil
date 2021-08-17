# pytorch-NetVlad

Adapted Pytorch Implementation of [NetVlad](https://arxiv.org/abs/1511.07247) from Nanne for Inference on multiple datasets.

## Acknowledgments

Original Pytorch Implementation of [NetVlad](https://arxiv.org/abs/1511.07247): https://github.com/Nanne/pytorch-NetVlad

## Dependencies

1. [PyTorch](https://pytorch.org/get-started/locally/) (at least v0.4.0)
2. [Faiss](https://github.com/facebookresearch/faiss)
3. [scipy](https://www.scipy.org/)
    - [numpy](http://www.numpy.org/)
    - [sklearn](https://scikit-learn.org/stable/)
    - [h5py](https://www.h5py.org/)

Simply run the following command: `pip3 install faiss-gpu scipy torch`

## NetVLAD BibTeX Citation

```txt
@inproceedings{arandjelovic2016netvlad,
  title={NetVLAD: CNN architecture for weakly supervised place recognition},
  author={Arandjelovic, Relja and Gronat, Petr and Torii, Akihiko and Pajdla, Tomas and Sivic, Josef},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5297--5307},
  year={2016}
}
```