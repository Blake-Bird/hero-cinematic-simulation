"""Radius–radial peculiar-velocity portrait of a saved halo particle sample."""
from pathlib import Path

import numpy as np


def phase_space_histogram(displacement, velocity, weights, r200, *, bins=96):
    """Histogram weighted particles relative to the inner-halo bulk velocity.

    Displacements and r200 must share units; velocities are peculiar km/s.
    The bulk is the weighted mean inside R200. No Hubble flow is added.
    """
    d = np.asarray(displacement, dtype=float)
    v = np.asarray(velocity, dtype=float)
    w = np.asarray(weights, dtype=float)
    if d.ndim != 2 or d.shape[1] != 3 or v.shape != d.shape or w.shape != (len(d),):
        raise ValueError('Particle displacements, velocities and weights must be aligned')
    if not np.isfinite(r200) or r200 <= 0 or type(bins) is not int or bins < 2:
        raise ValueError('Positive halo radius and at least two bins are required')
    radius = np.linalg.norm(d, axis=1)
    valid = np.isfinite(d).all(axis=1) & np.isfinite(v).all(axis=1) & np.isfinite(w) & (w > 0) & (radius > 0) & (radius < 2*r200)
    inner = valid & (radius < r200)
    if np.count_nonzero(inner) < 10 or np.count_nonzero(valid) < 20:
        return None
    bulk = np.average(v[inner], axis=0, weights=w[inner])
    radial = np.sum((v[valid]-bulk)*d[valid], axis=1)/radius[valid]
    rnorm = radius[valid]/r200
    limit = float(np.max(np.abs(radial)))
    if not np.isfinite(limit) or limit <= 0:
        return None
    hist, redges, vedges = np.histogram2d(rnorm,radial,bins=(bins,bins),
                                           range=((0,2),(-limit,limit)),weights=w[valid])
    return {'weighted_count':hist, 'radius_edges_r200':redges, 'radial_velocity_edges_km_s':vedges,
            'bulk_velocity_km_s':bulk, 'sampled_particles':int(np.count_nonzero(valid)),
            'inner_bulk_sampled_particles':int(np.count_nonzero(inner))}


def export_phase_portrait(portrait, path, *, snapshot, halo_row, redshift):
    if portrait is None:
        return None
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from figures import export_figure
    count = np.ma.masked_less_equal(portrait['weighted_count'].T,0)
    positive = count.compressed()
    fig, ax = plt.subplots(figsize=(9,6), layout='constrained')
    image = ax.pcolormesh(portrait['radius_edges_r200'], portrait['radial_velocity_edges_km_s'], count,
                          shading='auto', cmap='magma', norm=LogNorm(vmin=float(positive.min()),
                          vmax=max(float(positive.max()),float(positive.min())*1.001)))
    ax.axhline(0,color='white',lw=.7,alpha=.55)
    ax.axvline(1,color='white',lw=.7,alpha=.55,ls='--')
    ax.set(xlabel=(r'Radius $r/R_{200c}$' + '\nBulk: weighted mean within R200c · no Hubble flow · %s sampled particles' % format(portrait['sampled_particles'],',')),
           ylabel='Radial peculiar velocity relative to halo [km/s]',
           title='Dark-matter phase space · snapshot %d · halo row %d · z=%.4g' % (snapshot,halo_row,redshift))
    fig.colorbar(image,ax=ax,label='Estimated particle count per bin')
    try:
        export_figure(fig, Path(path), metadata={'snapshot':int(snapshot),'halo_catalogue_row':int(halo_row),
            'redshift':float(redshift),'product':'dark-matter radial peculiar phase space',
            'definition':'inverse-inclusion-weighted count; peculiar radial velocity relative to weighted mean velocity inside R200c; no Hubble flow',
            'sampled_particles':portrait['sampled_particles']})
    finally:
        plt.close(fig)
    np.savez_compressed(Path(path).with_suffix('.phase_data.npz'), **portrait)
    return path
