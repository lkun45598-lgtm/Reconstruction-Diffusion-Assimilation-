"""Headless-compatible placeholder for optional upstream debugging plots.
Sampler paths shipped in this project always use debug=False.
"""
def see(*args, **kwargs):
    raise RuntimeError('Interactive upstream plotting is not bundled; use saved .npz outputs.')
