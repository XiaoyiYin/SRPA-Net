"""Reuse the verified 224-band split, normalization and augmentation protocol."""
import importlib.util
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("srpa_benchmark_shared",ROOT/"vendor/benchmark/run_benchmark.py")
shared=importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


class OilSpillTiles(torch.utils.data.Dataset):
    def __init__(self,cache,split,band_indices=None):
        self.base=shared.CachedDataset(cache,split,augment=False)
        self.band_indices=list(range(224)) if band_indices is None else band_indices
        assert len(set(self.band_indices))==len(self.band_indices)
        assert all(0<=i<224 for i in self.band_indices)

    def __len__(self):
        return len(self.base)

    def __getitem__(self,index):
        image,label=self.base[index]
        if self.band_indices!=list(range(224)):
            image=image[self.band_indices]
        return image,label


def subset_metadata(metadata,band_indices):
    metadata=dict(metadata)
    metadata["wavelength_nm"]=[metadata["wavelength_nm"][i] for i in band_indices]
    metadata["fwhm_nm"]=[metadata["fwhm_nm"][i] for i in band_indices]
    metadata["kept_original_band_indices_0based"]=list(band_indices)
    return metadata
