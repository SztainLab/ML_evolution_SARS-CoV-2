#! /usr/bin/env python

import argparse, os, re
import numpy as np, pandas as pd, h5py, torch

from typing import Union
from bio_embeddings.utilities import read_fasta
from bio_embeddings.embed import EmbedderInterface, SeqVecEmbedder
from bio_embeddings.embed import ProtTransBertBFDEmbedder, ESM1bEmbedder
from tape import TAPETokenizer, UniRepModel
from tqdm import tqdm


class OneHotEmbedder(EmbedderInterface):

    name = "onehot"
    number_of_layers = 1

    def __init__(self, extended=False):
        super().__init__()
        if not extended:
            self._AMINO_ACIDS = np.asarray(list("ACDEFGHIKLMNPQRSTVWY"))
        else:
            self._AMINO_ACIDS = np.asarray(list("ACDEFGHIKLMNPQRSTVWYX"))
        self.embedding_dimension = len(self._AMINO_ACIDS)

    def embed(self, sequence: str) -> np.ndarray:
        
        if sum([s in self._AMINO_ACIDS for s in sequence]) == len(sequence):
            return np.asarray([self._AMINO_ACIDS == s for s in sequence]).astype(np.float32)
        else:
            raise ValueError("Sequence contains unsupported characters.")

    @staticmethod
    def reduce_per_protein(embedding: np.ndarray) -> np.ndarray:
        return embedding.mean(axis=0)


class UniRepEmbedder(EmbedderInterface):

    name = "unirep"
    embedding_dimension = 1900
    number_of_layers = 1

    def __init__(self, device: Union[None, str, torch.device] = None, **kwargs):
        super().__init__(device, **kwargs)
        self._tokenizer = TAPETokenizer(vocab='unirep')
        self._model = UniRepModel.from_pretrained('babbler-1900').eval().to(self._device)

    def embed(self, sequence: str) -> np.ndarray:
        token_ids = torch.tensor([self._tokenizer.encode(sequence)]).to(self._device)
        with torch.no_grad():
            return self._model(token_ids)[0].squeeze()[1:-1].cpu().numpy()

    @staticmethod
    def reduce_per_protein(embedding: np.ndarray) -> np.ndarray:
        return embedding.mean(axis=0)


# function to obtain embedding
def get_embedding(emb_name, model, sequence):
    if emb_name in ['onehot', 'unirep', 'protbert', 'esm1b']:
        emb = model.embed(sequence)
    elif emb_name == 'seqvec':
        emb = model.embed(sequence)
        emb = emb.sum(0) # sums over 3 layers
    return emb

# function to pad embedding according to gaps in alignment
def pad_embedding(emb, align):
    """ emb: array of per-residue embeddings
        align: sequence alignment
        return: array of per-residue embeddings padded in positions of gaps in alignment
    """
    non_gap_ids = np.where(np.asarray(list(align)) != '-')
    padded_emb = np.zeros((len(align), emb.shape[1]))
    padded_emb[non_gap_ids] = emb
    return padded_emb

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('emb_name',
            choices=['onehot', 'unirep', 'seqvec', 'protbert', 'esm1b'])
    args = parser.parse_args()

    if args.emb_name == 'onehot':
        model = OneHotEmbedder()
    elif args.emb_name == 'unirep':
        model = UniRepEmbedder()
    elif args.emb_name == 'seqvec':
        model = SeqVecEmbedder()
    elif args.emb_name == 'protbert':
        model = ProtTransBertBFDEmbedder()
    elif args.emb_name == 'esm1b':
        model = ESM1bEmbedder()

    # read alignment
    filename_aln = './fastaseq/{}.fasta'
    alignment = {dataset: read_fasta(filename_aln.format(dataset))\
            for dataset in ['generated_sequence']} #, 'train',  'valid', 'test']}

    # make base directory
    base_dir = './data/sequence/{}'.format(args.emb_name)
    os.makedirs(base_dir, exist_ok=True)
    
    # for each dataset generate embeddings and write to .h5 file
    for dataset in alignment.keys():
        with h5py.File('{}/{}.h5'.format(base_dir, dataset), 'w') as f:
            data = f.create_dataset("{}_{}".format(args.emb_name, dataset),
                    shape=(len(alignment[dataset]), len(alignment[dataset][0]),
                        model.embedding_dimension), dtype=np.float32)
            for i in tqdm(range(len(alignment[dataset])), desc='{:>10}'.format(dataset)):
                emb = get_embedding(args.emb_name, model,
                        str(alignment[dataset][i].seq).replace('-', ''))
                emb = pad_embedding(emb, str(alignment[dataset][i].seq))
                data[i, :, :] = emb

