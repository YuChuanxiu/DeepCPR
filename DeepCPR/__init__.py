"""DeepCPR public API with lazy imports.

Keeping heavy TensorFlow/SciPy modules lazy allows the standalone adaptive
resolver to be tested and used without importing the full legacy pipeline.
"""


def __getattr__(name):
    if name in {"data_resolution", "DeepCPRAdaptive"}:
        from .DeepCPR import DeepCPRAdaptive, data_resolution
        return {"data_resolution": data_resolution, "DeepCPRAdaptive": DeepCPRAdaptive}[name]
    if name == "peaktable":
        from .csv_merge import peaktable
        return peaktable
    if name == "netcdf_reader":
        from .NetCDF import netcdf_reader
        return netcdf_reader
    if name == "Chromseg":
        from .DeepCS import Chromseg
        return Chromseg
    raise AttributeError(name)


__all__ = [
    "data_resolution",
    "DeepCPRAdaptive",
    "peaktable",
    "netcdf_reader",
    "Chromseg",
]
