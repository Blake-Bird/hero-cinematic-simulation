"""Shared scientific figure export, plotted-array contracts and graph recipes.

Callers own measurements, physical units and support masks. This module owns
Matplotlib styling/layout, PNG/PDF/SVG publication, plotted geometry sidecars,
and selected measured story/diagnostic recipes. Styling never establishes a
physical unit conversion or scientific acceptance.
"""
from __future__ import annotations

# ========================================================================
# plot_style: Shared publication-oriented plotting style and reproducible figure export.

# Workflow and units
# Centralizes palettes, fonts and Matplotlib export options for analysis figures.
# Figure dimensions are inches, resolution is dots/inch and plotted physical units remain the caller's responsibility.
# Scientific figure sidecars preserve axes, legends and display choices; export styling cannot alter scientific values or uncertainties.
# Bundled source patches: core/patches. Runtime choices are saved with each run.

import datetime
import json
import os
import tempfile
import uuid
from pathlib import Path
import numpy as np

import matplotlib
from matplotlib import font_manager
from matplotlib import pyplot as plot_style_plt
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler


_CURRENT_OPTIONS = {
    "publication_style": "journal",
    "publication_font": "STIX Two Text",
    "figure_layout": "double_column",
    "figure_formats": ["png", "pdf", "svg"],
    "figure_dpi": 450,
}

_OKABE_ITO = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#9A6C00", "#007A8A", "#6A3D9A", "#222222",
]


def draw_measured_representation(axis, kind, values, options=None):
    """Draw measured arrays and return the exact derived display arrays.

    Masks must be Boolean. Bins, normalization and orientation are explicit;
    this function never fills missing measurements or fits a physical model.
    Returned arrays belong in the caller's numeric export alongside its recipe.
    """
    options = dict(options or {})
    arrays = {key: np.asarray(value) for key, value in values.items()}

    def supported(shape, *keys):
        mask = np.ones(shape, dtype=bool)
        if 'support' in arrays:
            source = arrays['support']
            if source.dtype.kind != 'b' or source.shape != shape:
                raise ValueError('Representation requires an aligned Boolean support mask')
            mask &= source
        for key in keys:
            if arrays[key].shape != shape:
                raise ValueError('Representation arrays have different shapes: ' + key)
            mask &= np.isfinite(arrays[key])
        return mask

    def edges(name):
        value = np.asarray(options[name], dtype=float)
        if value.ndim != 1 or len(value) < 2 or not np.isfinite(value).all() or np.any(np.diff(value) <= 0):
            raise ValueError('Explicit finite increasing bin edges required: ' + name)
        return value

    if kind in {'histogram', 'histogram2d', 'quantile_band'}:
        x = arrays['x']
        if x.ndim != 1:
            raise ValueError('Binned samples must be one dimensional')
        keys = ['x'] + (['y'] if kind != 'histogram' else [])
        if 'weight' in arrays:
            keys.append('weight')
        mask = supported(x.shape, *keys)
        weight = arrays.get('weight', np.ones(x.shape, dtype=float))
        if np.any(weight[mask] < 0):
            raise ValueError('Binned statistical weights must be nonnegative')
        bins = edges('x_edges')
        if kind == 'histogram':
            counts, _ = np.histogram(x[mask], bins=bins)
            totals, _ = np.histogram(x[mask], bins=bins, weights=weight[mask])
            normalization = options.get('normalization', 'sum')
            if normalization == 'probability_density':
                denominator = totals.sum() * np.diff(bins)
                plotted = np.divide(totals, denominator, out=np.full_like(totals, np.nan, dtype=float), where=denominator > 0)
            elif normalization == 'sum':
                plotted = totals
            else:
                raise ValueError('Histogram normalization must be sum or probability_density')
            axis.stairs(plotted, bins, label=options.get('label'))
            return {'edges': bins, 'counts': counts, 'weighted_sum': totals,
                    'values': plotted, 'sample_support': mask}
        y = arrays['y']
        if kind == 'histogram2d':
            ybins = edges('y_edges')
            grid, _, _ = np.histogram2d(x[mask], y[mask], bins=(bins, ybins), weights=weight[mask])
            count, _, _ = np.histogram2d(x[mask], y[mask], bins=(bins, ybins))
            # Histogram weights are nonnegative. A zero-support grid must not
            # acquire a negative mass/count color scale through autoscaling.
            upper = float(np.max(grid)) if grid.size else 0.
            artist = axis.pcolormesh(bins, ybins, grid.T, shading='flat', cmap=options.get('cmap', 'viridis'),
                                     vmin=0., vmax=upper if upper > 0 else 1.)
            axis.figure.colorbar(artist, ax=axis, label=options.get('color_label', ''))
            return {'x_edges': bins, 'y_edges': ybins, 'weighted_sum_xy': grid,
                    'counts_xy': count.astype(np.int64), 'sample_support': mask}
        probabilities = np.asarray(options.get('quantiles', [.16, .5, .84]), dtype=float)
        if probabilities.shape != (3,) or not np.isfinite(probabilities).all() or np.any(np.diff(probabilities) <= 0) or probabilities[0] < 0 or probabilities[-1] > 1:
            raise ValueError('Three ordered quantile probabilities in [0,1] required')
        minimum = options.get('minimum_count', 3)
        if type(minimum) is not int or minimum < 1:
            raise ValueError('Positive minimum bin count required')
        result = np.full((len(bins)-1, 3), np.nan)
        counts = np.zeros(len(bins)-1, dtype=np.int64)
        effective = np.zeros(len(bins)-1)
        for index, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
            selected = mask & (weight > 0) & (x >= left) & ((x <= right) if index == len(counts)-1 else (x < right))
            counts[index] = selected.sum()
            w = weight[selected]
            if len(w):
                effective[index] = w.sum()**2 / np.square(w).sum()
            if counts[index] < minimum or effective[index] < minimum:
                continue
            order = np.argsort(y[selected], kind='stable')
            ordered_y, ordered_w = y[selected][order], w[order]
            # Weighted empirical inverse CDF; no invented values between samples.
            cumulative = np.cumsum(ordered_w) / ordered_w.sum()
            result[index] = ordered_y[np.minimum(np.searchsorted(cumulative, probabilities), len(ordered_y)-1)]
        center = (bins[:-1]+bins[1:])/2
        axis.fill_between(center, result[:, 0], result[:, 2], alpha=.25)
        axis.plot(center, result[:, 1], label=options.get('label'))
        return {'edges': bins, 'centers': center, 'quantiles': result,
                'probabilities': probabilities, 'counts': counts,
                'effective_count': effective, 'sample_support': mask}

    if kind in {'heatmap', 'contour'}:
        x, y, grid = arrays['x'], arrays['y'], arrays['value']
        if x.ndim != 1 or y.ndim != 1 or grid.shape != (len(y), len(x)):
            raise ValueError('Grid values must have declared (y,x) orientation')
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
            raise ValueError('Grid coordinates must be finite and increasing')
        mask = supported(grid.shape, 'value')
        shown = np.ma.array(grid, mask=~mask)
        if kind == 'heatmap':
            artist = axis.pcolormesh(x, y, shown, shading='nearest', cmap=options.get('cmap', 'viridis'))
        else:
            levels = edges('levels')
            artist = axis.contour(x, y, shown, levels=levels, cmap=options.get('cmap', 'viridis'))
        axis.figure.colorbar(artist, ax=axis, label=options.get('color_label', ''))
        return {'x': x, 'y': y, 'values_yx': grid, 'support': mask}

    if kind == 'tensor':
        x, y, tensor = arrays['x'], arrays['y'], arrays['tensor']
        if x.ndim != 1 or y.shape != x.shape or tensor.shape != (len(x), 2, 2):
            raise ValueError('Tensor glyphs require one symmetric 2x2 tensor per 2D position')
        mask = supported(x.shape, 'x', 'y') & np.isfinite(tensor).all(axis=(1, 2))
        if not np.allclose(tensor[mask], np.swapaxes(tensor[mask], 1, 2), rtol=1e-10, atol=0):
            raise ValueError('Tensor eigenaxis glyphs require symmetric measured tensors')
        scale = float(options['scale'])
        tolerance = float(options['minimum_relative_eigengap'])
        if not np.isfinite([scale, tolerance]).all() or scale <= 0 or tolerance < 0:
            raise ValueError('Explicit positive glyph scale and nonnegative eigenaxis support threshold required')
        eigenvalues = np.full((len(x), 2), np.nan)
        eigenvectors = np.full((len(x), 2, 2), np.nan)
        eigenvalues[mask], eigenvectors[mask] = np.linalg.eigh(tensor[mask])
        gap = np.abs(eigenvalues[:, 1]-eigenvalues[:, 0])
        magnitude = np.max(np.abs(eigenvalues), axis=1)
        orientations = mask & (magnitude > 0) & (gap > tolerance*magnitude)
        from matplotlib.collections import LineCollection
        segments = []
        for component in range(2):
            center = np.column_stack((x[orientations], y[orientations]))
            displacement = eigenvectors[orientations, :, component] * np.abs(eigenvalues[orientations, component, None])/scale
            segments.extend(np.stack((center-displacement, center+displacement), axis=1))
        axis.add_collection(LineCollection(segments, colors=options.get('color', 'tab:blue'), alpha=.7))
        axis.scatter(x[mask], y[mask], s=options.get('marker_size', 2))
        axis.autoscale_view()
        return {'x': x, 'y': y, 'tensor': tensor, 'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors, 'support': mask, 'orientation_support': orientations}

    if kind == 'vector':
        x = arrays['x']
        mask = supported(x.shape, 'x', 'y', 'u', 'v')
        if x.ndim != 1:
            raise ValueError('Vector glyph coordinates must be one dimensional')
        scale = float(options['scale'])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Explicit positive vector display scale required')
        axis.quiver(x[mask], arrays['y'][mask], arrays['u'][mask], arrays['v'][mask],
                    angles='xy', scale_units='xy', scale=scale, pivot='mid')
        return {**arrays, 'support': mask, 'display_scale': np.asarray(scale)}

    if kind == 'sky':
        longitude, latitude = arrays['x'], arrays['y']
        mask = supported(longitude.shape, 'x', 'y', 'color')
        if longitude.ndim != 1 or np.any(np.abs(latitude[mask]) > np.pi/2):
            raise ValueError('Sky coordinates must be one dimensional radians with latitude in [-pi/2,pi/2]')
        if axis.name not in {'mollweide', 'aitoff', 'hammer'}:
            raise ValueError('Sky representation requires a spherical map projection')
        wrapped = (longitude + np.pi) % (2*np.pi) - np.pi
        artist = axis.scatter(wrapped[mask], latitude[mask], c=arrays['color'][mask],
                              s=options.get('marker_size', 4), cmap=options.get('cmap', 'viridis'))
        axis.figure.colorbar(artist, ax=axis, label=options.get('color_label', ''))
        axis.grid(True, alpha=.3)
        return {**arrays, 'display_longitude': wrapped, 'support': mask}

    if kind in {'correlation', 'pca'}:
        features = arrays['features']
        if features.ndim != 2 or features.shape[1] < 2:
            raise ValueError('Feature analysis requires an object-by-feature matrix')
        mask = np.isfinite(features).all(axis=1)
        if 'support' in arrays:
            if arrays['support'].dtype.kind != 'b' or arrays['support'].shape != mask.shape:
                raise ValueError('Feature support must be one Boolean per object')
            mask &= arrays['support']
        labels = options['feature_labels']
        if len(labels) != features.shape[1] or any(not isinstance(label, str) for label in labels):
            raise ValueError('Every frozen feature requires a label')
        if mask.sum() < 3:
            raise ValueError('At least three complete supported objects required')
        selected = features[mask]
        mean = selected.mean(axis=0)
        scale = selected.std(axis=0, ddof=1)
        if np.any(scale <= 0) or not np.isfinite(scale).all():
            raise ValueError('Constant or unrepresentable features must be explicitly removed')
        standardized = (selected-mean)/scale
        if kind == 'correlation':
            matrix = standardized.T @ standardized / (len(selected)-1)
            artist = axis.imshow(matrix, vmin=-1, vmax=1, cmap='coolwarm', origin='lower')
            axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha='right')
            axis.set_yticks(np.arange(len(labels)), labels)
            axis.figure.colorbar(artist, ax=axis, label='Pearson correlation')
            return {'correlation': matrix, 'feature_mean': mean, 'feature_scale': scale,
                    'sample_support': mask, 'sample_count': np.asarray(len(selected), dtype=np.int64)}
        _, singular, axes = np.linalg.svd(standardized, full_matrices=False)
        # Fix SVD sign ambiguity deterministically for reproducible saved scores.
        dominant = np.argmax(np.abs(axes), axis=1)
        signs = np.where(axes[np.arange(len(axes)), dominant] < 0, -1., 1.)
        axes *= signs[:, None]
        scores = standardized @ axes.T
        variance = singular**2/(len(selected)-1)
        fractions = variance/variance.sum()
        components = options.get('components', [0, 1])
        if len(components) != 2 or any(type(index) is not int or not 0 <= index < scores.shape[1] for index in components):
            raise ValueError('Two available PCA component indices required')
        a, b = components
        axis.scatter(scores[:, a], scores[:, b], s=options.get('marker_size', 4))
        axis.set_xlabel('PC%d (%.1f%%)' % (a+1, 100*fractions[a]))
        axis.set_ylabel('PC%d (%.1f%%)' % (b+1, 100*fractions[b]))
        complete_scores = np.full((len(features), scores.shape[1]), np.nan)
        complete_scores[mask] = scores
        return {'scores': complete_scores, 'components': axes, 'variance': variance,
                'variance_fraction': fractions, 'feature_mean': mean,
                'feature_scale': scale, 'sample_support': mask}

    if kind == 'graph':
        positions, endpoints = arrays['positions'], arrays['edges']
        if positions.ndim != 2 or positions.shape[1] not in {2, 3} or not np.isfinite(positions).all():
            raise ValueError('Graph requires finite measured or explicitly supplied layout coordinates')
        if endpoints.ndim != 2 or endpoints.shape[1] != 2 or endpoints.dtype.kind not in 'iu' or np.any(endpoints < 0) or np.any(endpoints >= len(positions)):
            raise ValueError('Graph edges must be valid integer node indices')
        if positions.shape[1] == 3:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            axis.add_collection3d(Line3DCollection(positions[endpoints], alpha=.5))
            axis.scatter(*positions.T, s=options.get('marker_size', 8))
        else:
            from matplotlib.collections import LineCollection
            axis.add_collection(LineCollection(positions[endpoints], alpha=.5))
            axis.scatter(*positions.T, s=options.get('marker_size', 8))
        axis.autoscale_view()
        return arrays
    raise ValueError('Unknown measured representation: ' + kind)

_PRESETS = {
    "journal": {
        "single_column_width": 3.54,
        "double_column_width": 7.20,
        "presentation_width": 12.80,
        "base_font_size": 8.5,
        "line_width": 1.25,
        "marker_size": 3.2,
    },
    "presentation": {
        "single_column_width": 6.0,
        "double_column_width": 10.0,
        "presentation_width": 12.80,
        "base_font_size": 12.0,
        "line_width": 1.8,
        "marker_size": 5.0,
    },
}


def _resolved_font(preferred):
    entries = font_manager.fontManager.ttflist
    available = {entry.name for entry in entries}
    bold_capable = {
        entry.name for entry in entries
        if isinstance(entry.weight, (int, float)) and float(entry.weight) >= 600
    }
    if preferred in available and preferred in bold_capable:
        return preferred
    for fallback in ("STIX Two Text", "STIXGeneral", "DejaVu Serif", "DejaVu Sans"):
        if fallback in available and fallback in bold_capable:
            return fallback
    return "DejaVu Sans"


def apply_plot_style(visualization=None):
    """Apply a consistent Matplotlib style and remember export settings."""
    options = dict(_CURRENT_OPTIONS)
    if isinstance(visualization, dict):
        options.update({key: visualization[key] for key in _CURRENT_OPTIONS if key in visualization})
    style_name = str(options["publication_style"]).lower()
    if style_name not in _PRESETS:
        raise ValueError("visualization.publication_style must be 'journal' or 'presentation'")
    font_name = _resolved_font(str(options["publication_font"]))
    if options["figure_layout"] not in {"single_column", "double_column", "presentation"}:
        raise ValueError("visualization.figure_layout must be single_column, double_column, or presentation")
    dpi = int(options["figure_dpi"])
    formats = [str(fmt).lower().lstrip(".") for fmt in options["figure_formats"]]
    if not formats or any(fmt not in {"png", "pdf", "svg"} for fmt in formats):
        raise ValueError("visualization.figure_formats must contain png, pdf, and/or svg")
    if "png" not in formats:
        formats.append("png")
    options["publication_font"] = font_name
    options["figure_dpi"] = dpi
    options["figure_formats"] = list(dict.fromkeys(formats))
    _CURRENT_OPTIONS.update(options)

    preset = _PRESETS[style_name]
    base = float(preset["base_font_size"])
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": [font_name, "STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "font.size": base,
        "mathtext.fontset": "stix",
        "axes.titlesize": base + 1.0,
        "axes.titleweight": "bold",
        "axes.titlepad": 5.0,
        "axes.labelsize": base,
        "axes.labelweight": "regular",
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=_OKABE_ITO),
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#D9E0E8",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.65,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": base - 0.5,
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.55,
        "lines.linewidth": float(preset["line_width"]),
        "lines.markersize": float(preset["marker_size"]),
        "errorbar.capsize": 2.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def figure_size(layout="single_column", height_ratio=0.72, panels=1):
    """Return a journal-aware figure size in inches."""
    style = _PRESETS[_CURRENT_OPTIONS["publication_style"]]
    if layout not in {"single_column", "double_column", "presentation"}:
        raise ValueError("layout must be single_column, double_column, or presentation")
    width_key = {
        "single_column": "single_column_width",
        "double_column": "double_column_width",
        "presentation": "presentation_width",
    }[layout]
    width = float(style[width_key])
    if isinstance(panels, bool) or int(panels) < 1:
        raise ValueError("panels must be a positive integer")
    width *= max(1.0, min(float(panels) / 2.0, 2.0))
    return width, width * float(height_ratio)


def semantic_cmap(kind):
    maps = {
        "density": "cosmic_web",
        "cosmic_web": "cosmic_web",
        "temperature": "inferno",
        "sequential": "cividis",
        "difference": "coolwarm",
        "categorical": "tab10",
    }
    try:
        return maps[kind]
    except KeyError as exc:
        raise ValueError("unknown scientific colormap semantic: " + str(exc)) from exc


def _register_cosmic_web_colormap():
    name = "cosmic_web"
    if name in matplotlib.colormaps:
        return
    cmap = LinearSegmentedColormap.from_list(
        name,
        [
            (0.00, "#050713"),
            (0.18, "#17123f"),
            (0.38, "#44306f"),
            (0.58, "#9b426f"),
            (0.78, "#ed8249"),
            (1.00, "#fff2c5"),
        ],
        N=1024,
    )
    matplotlib.colormaps.register(cmap, name=name)


_register_cosmic_web_colormap()

def _axes_description(fig):
    descriptions = []
    for axis in fig.axes:
        shared_x_label = next((
            other.get_xlabel() for other in fig.axes
            if other is not axis and other.get_xlabel()
            and axis.get_shared_x_axes().joined(axis, other)
        ), "")
        shared_y_label = next((
            other.get_ylabel() for other in fig.axes
            if other is not axis and other.get_ylabel()
            and axis.get_shared_y_axes().joined(axis, other)
        ), "")
        descriptions.append({
            "title": axis.get_title(),
            "x_label": axis.get_xlabel() or shared_x_label,
            "y_label": axis.get_ylabel() or shared_y_label,
            "role": "colorbar" if getattr(axis, "_colorbar", None) is not None else "plot",
            "has_data": bool(axis.has_data()),
        })
    return descriptions


def _layout_diagnostics(fig):
    """Record visible text outside the crop that ``bbox_inches='tight'`` exports."""
    from matplotlib.text import Text

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = (float(value) for value in fig.get_size_inches())
    padding_inches = 0.035
    tolerance_inches = 0.015
    try:
        export_bbox = fig.get_tightbbox(renderer).padded(padding_inches)
        export_bounds = (float(export_bbox.x0), float(export_bbox.y0),
                         float(export_bbox.x1), float(export_bbox.y1))
    except (AttributeError, RuntimeError, ValueError, TypeError):
        export_bounds = (0.0, 0.0, width, height)
    outside = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        # Some locators leave detached, obsolete tick-label Text instances in
        # the figure artist tree after a constrained-layout redraw. They are
        # not part of the exported image and have no owning axes.
        if artist.axes is None and artist not in fig.texts:
            continue
        try:
            bounds = artist.get_window_extent(renderer=renderer).transformed(fig.dpi_scale_trans.inverted())
        except (AttributeError, RuntimeError, ValueError, TypeError):
            continue
        if (bounds.x0 < export_bounds[0] - tolerance_inches
                or bounds.y0 < export_bounds[1] - tolerance_inches
                or bounds.x1 > export_bounds[2] + tolerance_inches
                or bounds.y1 > export_bounds[3] + tolerance_inches):
            outside.append({
                "text": artist.get_text().replace("\n", " ")[:120],
                "bounds_inches": [float(bounds.x0), float(bounds.y0), float(bounds.x1), float(bounds.y1)],
            })
    return {
        "canvas_inches": [width, height],
        "export_bbox_inches": list(export_bounds),
        "export_padding_inches": padding_inches,
        "tolerance_inches": tolerance_inches,
        "visible_text_outside_canvas_count": len(outside),
        "visible_text_outside_canvas": outside[:20],
    }


def _plotted_arrays(fig):
    """Preserve the arrays actually drawn, including masks, without inventing units."""
    arrays={}
    def capture(key,values):
        array=np.asanyarray(values)
        if array.dtype.kind in 'biufc':
            arrays[key]=np.ma.getdata(array)
            if np.ma.isMaskedArray(array):
                arrays[key+'_mask']=np.ma.getmaskarray(array)
    for index,axis in enumerate(fig.axes):
        for j,line in enumerate(axis.lines):
            if hasattr(line,'get_data_3d'):
                xx,yy,zz=line.get_data_3d();capture(f'axis_{index}_line_{j}_z',zz)
            capture(f'axis_{index}_line_{j}_x',line.get_xdata())
            capture(f'axis_{index}_line_{j}_y',line.get_ydata())
        for j,collection in enumerate(axis.collections):
            if hasattr(collection,'_offsets3d'):
                for coordinate,values in zip(('x','y','z'),collection._offsets3d):capture(f'axis_{index}_collection_{j}_offsets3d_{coordinate}',values)
            capture(f'axis_{index}_collection_{j}_offsets',collection.get_offsets())
            if collection.get_array() is not None:
                capture(f'axis_{index}_collection_{j}_values',collection.get_array())
            if hasattr(collection,'get_segments'):
                for k,segment in enumerate(collection.get_segments()):
                    capture(f'axis_{index}_collection_{j}_segment_{k}',segment)
            for k,path in enumerate(collection.get_paths()):
                capture(f'axis_{index}_collection_{j}_path_{k}',path.vertices)
        for j,image in enumerate(axis.images):
            capture(f'axis_{index}_image_{j}_values',image.get_array())
            capture(f'axis_{index}_image_{j}_extent',image.get_extent())
        for j,patch in enumerate(axis.patches):
            if hasattr(patch,'get_xy'):
                capture(f'axis_{index}_patch_{j}_xy',patch.get_xy())
            if hasattr(patch,'get_width') and hasattr(patch,'get_height'):
                capture(f'axis_{index}_patch_{j}_width_height',[patch.get_width(),patch.get_height()])
    return arrays


def verify_figure_export(sidecar_path):
    from core.foundation import file_sha256 as digest
    path=Path(sidecar_path).resolve()
    receipt=path.with_name(path.name+'.sha256')
    if not receipt.is_file() or receipt.read_text().strip()!=digest(path):
        raise ValueError('Figure manifest checksum mismatch')
    record=json.loads(path.read_text())
    for entry in record['artifacts']:
        asset=(path.parent/entry['path']).resolve()
        if not asset.is_relative_to(path.parent) or not asset.is_file() or digest(asset)!=entry['sha256']:
            raise ValueError('Figure asset missing or changed: '+entry['path'])
        if asset.suffix=='.npz':
            with np.load(asset,allow_pickle=False) as arrays:
                for name in arrays.files:
                    arrays[name]
    if record.get('plot_data_contract'):
        from core.data import verify_numpy_archive_contract
        verify_numpy_archive_contract(path.parent/record['plot_data'],record['plot_data_contract'])
    return record


def export_figure(fig, path, metadata=None, formats=None, dpi=None, close=True, layout=None):
    """Export one figure as PNG plus requested editable/vector formats.

    A JSON companion records the axes text, export settings, and caller-provided
    scientific metadata.  ``path`` supplies the output stem and may end in any
    extension; output names are chosen from ``formats``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    selected_formats = list(formats or _CURRENT_OPTIONS["figure_formats"])
    selected_formats = [str(fmt).lower().lstrip(".") for fmt in selected_formats]
    if "png" not in selected_formats:
        selected_formats.append("png")
    invalid = sorted(set(selected_formats) - {"png", "pdf", "svg"})
    if invalid:
        raise ValueError("unsupported figure format(s): " + ", ".join(invalid))
    output_dpi = int(dpi or _CURRENT_OPTIONS["figure_dpi"])
    selected_layout = str(layout or _CURRENT_OPTIONS["figure_layout"])
    target_width = figure_size(selected_layout)[0]
    original_width, original_height = (float(value) for value in fig.get_size_inches())
    if original_width > 0 and original_height > 0:
        fig.set_size_inches(target_width, target_width * original_height / original_width, forward=True)
    layout_diagnostics = _layout_diagnostics(fig)
    stem = target.with_suffix("") if target.suffix else target
    axes = _axes_description(fig)
    title = next((entry["title"] for entry in axes if entry["title"]), stem.stem)
    scientific_metadata = json.loads(json.dumps(dict(metadata or {}),default=str,allow_nan=False))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    stage=Path(tempfile.mkdtemp(prefix=".figure-export-",dir=target.parent))
    for fmt in dict.fromkeys(selected_formats):
        output = stage/stem.with_suffix("." + fmt).name
        figure_facecolor = fig.get_facecolor()
        save_options = {
            "format": fmt,
            "bbox_inches": "tight",
            "pad_inches": 0.035,
            "facecolor": figure_facecolor,
            "edgecolor": figure_facecolor,
        }
        if fmt == "png":
            save_options["dpi"] = output_dpi
            save_options["metadata"] = {
                "Software": "DarkMatterSim",
                "Title": title,
                "Description": json.dumps(scientific_metadata, sort_keys=True, default=str),
            }
        elif fmt == "pdf":
            save_options["metadata"] = {
                "Creator": "DarkMatterSim",
                "Title": title,
                "Subject": json.dumps(scientific_metadata, sort_keys=True, default=str),
            }
        elif fmt == "svg":
            save_options["metadata"] = {
                "Creator": "DarkMatterSim",
                "Title": title,
                "Description": json.dumps(scientific_metadata, sort_keys=True, default=str),
            }
        fig.savefig(output, **save_options)
    from core.foundation import atomic_json, file_sha256 as digest, fsync_directory, publish_directory
    array_path=stage/(stem.name+".plotdata.npz")
    drawn_arrays=_plotted_arrays(fig)
    np.savez_compressed(array_path,**drawn_arrays)
    from core.data import numpy_archive_contract
    def artist_unit(name):
        if name.endswith('_mask'):return 'dimensionless'
        index=int(name.split('_')[1]);declarations=getattr(fig.axes[index],'_physical_plot_units',{})
        if not declarations:return 'UNDECLARED'
        if name.endswith(('_x','_y','_z')):return declarations.get(name[-1],'UNDECLARED')
        if '_image_' in name and name.endswith('_values'):return declarations.get('color','UNDECLARED')
        if '_collection_' in name and name.endswith('_values'):return declarations.get('color','UNDECLARED')
        if '_path_' in name:return 'display marker geometry; no physical units'
        return 'mixed axis geometry; '+str(declarations)
    array_contract=numpy_archive_contract(array_path,{name:{
        'units':artist_unit(name),
        'quantity':'artist availability mask' if name.endswith('_mask') else 'drawn artist '+name,
        'missing_policy':'Masked or nonfinite artist values are preserved without interpolation',
        'annotations':{'scope':'Display geometry; source measurement definitions remain in scientific_metadata'}}
        for name in drawn_arrays},epoch=scientific_metadata.get('epoch'),
        provenance={'algorithm_sha256':digest(__file__),'scientific_metadata':scientific_metadata})
    sidecar = {
        "schema": "darkmattersim-figure-v1",
        "created_utc": now,
        "title": title,
        "axes": axes,
        "font_family": _CURRENT_OPTIONS["publication_font"],
        "style": _CURRENT_OPTIONS["publication_style"],
        "layout": selected_layout,
        "formats": list(dict.fromkeys(selected_formats)),
        "png_dpi": output_dpi,
        "figure_size_inches": [float(x) for x in fig.get_size_inches()],
        "figure_facecolor_rgba": [float(x) for x in fig.get_facecolor()],
        "layout_diagnostics": layout_diagnostics,
        "scientific_metadata": scientific_metadata,
        "plot_data": array_path.name,
        "plot_data_contract": array_contract,
        "plot_data_scope": "Arrays actually drawn, including artist geometry and masks; physical measurement definitions remain in the source product",
        "units_status": ("SAVED_DATASET_DECLARATIONS" if scientific_metadata.get('unit_source')=='saved HDF5 producer declarations'
                         else "Caller metadata only; axis labels are preserved, never interpreted as a verified unit contract"),
        "science_acceptance": "NOT_ACCEPTED",
        "artifacts": [{'path':p.name,'sha256':digest(p),'size_bytes':p.stat().st_size} for p in sorted(stage.iterdir()) if p.is_file()],
    }
    sidecar_path = stem.with_suffix(".figure.json")
    staged_manifest=stage/sidecar_path.name
    atomic_json(staged_manifest,json.loads(json.dumps(sidecar,default=str,allow_nan=False)))
    staged_receipt=staged_manifest.with_name(staged_manifest.name+'.sha256')
    staged_receipt.write_text(digest(staged_manifest)+'\n')
    verify_figure_export(staged_manifest)
    # Preserve a complete previous figure record before replacing its names.
    prior=[p for p in target.parent.iterdir() if p.is_file() and p.name in {v.name for v in stage.iterdir()}]
    if prior:
        import shutil
        history=target.parent/'.figure_history';history.mkdir(exist_ok=True)
        previous=Path(tempfile.mkdtemp(prefix='.previous-',dir=history))
        for old in prior:
            shutil.copy2(old,previous/old.name)
        publish_directory(previous,history/(stem.name+'_'+uuid.uuid4().hex))
    files=[p for p in sorted(stage.iterdir()) if p not in (staged_manifest,staged_receipt)]
    files.extend([staged_manifest,staged_receipt])
    for source in files:
        with source.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(source,target.parent/source.name)
    fsync_directory(target.parent)
    stage.rmdir()
    if close:
        plot_style_plt.close(fig)
    return [stem.with_suffix("." + fmt) for fmt in dict.fromkeys(selected_formats)]


apply_plot_style()

# ========================================================================
# cosmic_web_figures: Scientific cosmic-web figure plates for analysis outputs.

# Workflow and units
# Renders 3-D density/morphology plates and a 2-D particle-to-ridge sequence.
# Density ratios are dimensionless; map extents are comoving Mpc/h. Gaussian smoothing is expressed in mesh cells and recorded in sidecars.
# Colour, opacity and percentile contrast are presentation choices. Delaunay edges and Hessian ridge cuts do not supply a gravitational force, persistence significance or confirmed physical filament.
# Bundled source patches: core/patches. Runtime choices are saved with each run.

import math

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgb, to_rgba


_DARK = "#171c20"
_INK = "#15233b"
_RIDGE = "#b65a44"
_PALETTE = {
    "void": "#8ba69a",
    "wall": "#c5aa72",
    "filament": "#b98268",
    "node": "#f5eee0",
}


def _contrast_map():
    return LinearSegmentedColormap.from_list(
        "cosmic_web_scientific",
        ["#171c20", "#34454e", "#617a7d", "#9bada0", "#c6bf98", "#e4d1a0", "#fff8ef"],
        N=1024,
    )


def _rotation_basis():
    """Return an orthonormal oblique viewing basis for rotating the numerical density volume."""
    view = np.asarray([1.0, -1.12, 0.78], dtype=float)
    view /= np.linalg.norm(view)
    right = np.cross(view, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    up /= np.linalg.norm(up)
    return np.column_stack((right, up, view))


def _rotate_volume(volume, basis, side, order):
    """Resample the density/class mesh in the supplied viewing basis; interpolation order is explicitly chosen."""
    from scipy.ndimage import affine_transform

    n = int(volume.shape[0])
    center_in = np.full(3, (n - 1.0) / 2.0)
    center_out = np.full(3, (side - 1.0) / 2.0)
    offset = center_in - basis @ center_out
    return affine_transform(
        np.asarray(volume), basis, offset=offset, output_shape=(side, side, side),
        order=order, mode="constant", cval=0.0, prefilter=(order > 1),
    )


def _ray_cast(rotated_density, rotated_mask, density_limits, base_color=None):
    """Front-to-back emission/absorption projection of a rotated volume."""
    low, high = density_limits
    contrast = np.log1p(np.maximum(rotated_density, 0.0)).astype(np.float32)
    strength = np.clip((contrast - low) / max(high - low, 1e-6), 0.0, 1.0)
    side = strength.shape[0]
    accumulated = np.zeros((side, side, 3), dtype=np.float32)
    transmittance = np.ones((side, side), dtype=np.float32)
    cmap = _contrast_map()
    opacity_scale = 0.23 if base_color is None else 0.14
    for depth in range(side):
        visible = rotated_mask[:, :, depth]
        if not np.any(visible):
            continue
        s = np.clip(strength[:, :, depth] * 1.25, 0.0, 1.0) ** 0.8
        if base_color is None:
            color = cmap(s)[..., :3].astype(np.float32)
            opacity = opacity_scale * np.power(s, 1.15)
        else:
            color = np.asarray(to_rgb(base_color), dtype=np.float32)[None, None, :]
            color = color * (0.28 + 0.9 * s[..., None]) + 0.06 * s[..., None]
            opacity = opacity_scale * (0.35 + 0.65 * s)
        opacity = np.where(visible, opacity, 0.0).astype(np.float32)
        accumulated += transmittance[..., None] * opacity[..., None] * color
        transmittance *= 1.0 - opacity
    return np.clip(accumulated, 0.0, 1.0)


def _cube_edges(axis, n, side, basis):
    center = (side - 1.0) / 2.0
    corners = np.asarray([
        [x, y, z]
        for x in (-0.5, 0.5)
        for y in (-0.5, 0.5)
        for z in (-0.5, 0.5)
    ]) * float(n)
    screen = corners @ basis
    screen[:, :2] += center
    for first in range(8):
        for second in range(first + 1, 8):
            if sum(corners[first, k] != corners[second, k] for k in range(3)) == 1:
                axis.plot(
                    screen[[first, second], 0], screen[[first, second], 1],
                    color="#c7ccc8", alpha=0.60, linewidth=0.72, zorder=4,
                )
    axis.set_xlim(-2, side + 1)
    axis.set_ylim(-2, side + 1)
    axis.set_aspect("equal")
    axis.set_facecolor(_DARK)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _skeletonize(binary):
    """Small periodic Zhang-Suen thinning routine; no optional package needed."""
    image = np.asarray(binary, dtype=bool).copy()
    if image.ndim != 2 or not np.any(image):
        return image
    for _ in range(max(image.shape)):
        changed = False
        for phase in (0, 1):
            p2 = np.roll(image, -1, axis=0)
            p3 = np.roll(p2, -1, axis=1)
            p4 = np.roll(image, -1, axis=1)
            p5 = np.roll(np.roll(image, 1, axis=0), -1, axis=1)
            p6 = np.roll(image, 1, axis=0)
            p7 = np.roll(np.roll(image, 1, axis=0), 1, axis=1)
            p8 = np.roll(image, 1, axis=1)
            p9 = np.roll(np.roll(image, -1, axis=0), 1, axis=1)
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(p.astype(np.uint8) for p in neighbors)
            transitions = sum(
                ((neighbors[index] == 0) & (neighbors[(index + 1) % 8] == 1)).astype(np.uint8)
                for index in range(8)
            )
            if phase == 0:
                directional = ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                directional = ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            remove = image & (count >= 2) & (count <= 6) & (transitions == 1) & directional
            if np.any(remove):
                image[remove] = False
                changed = True
        if not changed:
            break
    return image


def _wrapped_window(points, center, box, width):
    """Select projected positions by minimum-image separation from the requested comoving centre."""
    delta = np.asarray(points[:, :2], dtype=float) - np.asarray(center[:2])[None, :]
    delta -= float(box) * np.round(delta / float(box))
    keep = np.all(np.abs(delta) <= float(width) / 2.0, axis=1)
    return delta[keep], keep


def _periodic_crop(field, center, box, width):
    """Sample the requested physical window at its displayed pixel centres.

    Input pixels lie at (i+1/2)*box/n. Sampling that geometry, rather than
    rounding a start index and stretching the extracted block, keeps particle
    coordinates and extrema markers aligned. Scalar fields use bilinear display
    interpolation; categorical ridge masks use nearest neighbours.
    """
    from scipy.ndimage import map_coordinates
    values = np.asarray(field)
    center = np.asarray(center, dtype=float)
    n = int(values.shape[0])
    if values.shape != (n, n) or center.shape != (2,) or np.any(~np.isfinite(center)):
        raise ValueError("Periodic crops need a square field and finite 2-D centre")
    if not math.isfinite(box) or box <= 0 or not math.isfinite(width) or not 0 < width <= box:
        raise ValueError("Crop width must be positive and no larger than the periodic box")
    count = max(8, min(n, int(math.ceil(width*n/box))))
    offsets = ((np.arange(count)+.5)/count-.5)*width
    x, y = np.meshgrid(center[0]+offsets, center[1]+offsets, indexing="ij")
    sample_coordinates = np.stack((x*n/box-.5, y*n/box-.5))
    return map_coordinates(values, sample_coordinates, order=0 if values.dtype.kind in "biu" else 1,
                           mode="grid-wrap", prefilter=False)


def _projected_hessian_ridges(surface):
    """Build nested, thin 2-D ridge candidates from density Hessian response."""
    from scipy.ndimage import gaussian_filter, maximum_filter

    scalar = np.log1p(np.maximum(np.asarray(surface, dtype=np.float32), 0.0))
    sigma = 2.2
    hxx = gaussian_filter(scalar, sigma=sigma, order=(2, 0), mode="wrap")
    hyy = gaussian_filter(scalar, sigma=sigma, order=(0, 2), mode="wrap")
    hxy = gaussian_filter(scalar, sigma=sigma, order=(1, 1), mode="wrap")
    gx = gaussian_filter(scalar, sigma=sigma, order=(1, 0), mode="wrap")
    gy = gaussian_filter(scalar, sigma=sigma, order=(0, 1), mode="wrap")
    smallest = 0.5 * (hxx + hyy - np.sqrt((hxx - hyy) ** 2 + 4.0 * hxy ** 2))
    response = np.maximum(-smallest, 0.0)
    # Eigenvector of the most-negative curvature direction; density gradient
    # should vanish along that normal for a ridge center.
    angle = 0.5 * np.arctan2(2.0 * hxy, hxx - hyy)
    normal_gradient = np.abs(-np.sin(angle) * gx + np.cos(angle) * gy)
    local_response = response == maximum_filter(response, size=3, mode="wrap")
    positive = response[response > 0]
    if not positive.size:
        return [np.zeros(response.shape, dtype=bool) for _ in range(3)], [0.0, 0.0, 0.0]
    response_cuts = np.percentile(positive, [58.0, 75.0, 89.0])
    density_cuts = np.percentile(surface, [25.0, 45.0, 66.0])
    gradient_limit = np.percentile(normal_gradient, 58.0) if normal_gradient.size else np.inf
    masks = []
    for response_cut, density_cut in zip(response_cuts, density_cuts):
        selected = (
            (response >= response_cut) & (surface >= density_cut)
            & (normal_gradient <= max(float(gradient_limit), 1e-7))
        )
        if np.count_nonzero(selected) < 10:
            selected = local_response & (response >= response_cut) & (surface >= density_cut)
        masks.append(_skeletonize(selected))
    return masks, [float(x) for x in response_cuts]


def _delaunay_overlay(axis, points, density_image, box, center=None, width=None,
                      norm=None, cmap=None, alpha=0.48, max_points=1800):
    """Draw a triangulation of sampled projected positions; edges are illustrative mesh geometry."""
    from matplotlib.tri import Triangulation
    from scipy.spatial import Delaunay

    if len(points) < 4:
        return False
    if center is not None and width is not None:
        selected, keep = _wrapped_window(points, center, box, width)
        if len(selected) > max_points:
            stride = int(math.ceil(len(selected) / max_points))
            selected = selected[::stride]
        xy = selected + float(width) / 2.0
        axis.set_xlim(0, width)
        axis.set_ylim(0, width)
    else:
        selected = np.asarray(points[:, :2], dtype=float)
        if len(selected) > max_points:
            selected = selected[np.linspace(0, len(selected) - 1, max_points, dtype=int)]
        xy = selected
    if len(xy) < 4:
        return False
    try:
        triangulation = Delaunay(xy)
        triangles = triangulation.simplices
        tri = Triangulation(xy[:, 0], xy[:, 1], triangles)
        if center is None:
            tri.set_mask(np.any((xy[triangles] < 0.0) | (xy[triangles] > box), axis=(1, 2)))
            axis.triplot(tri, color="#91aaa4", linewidth=0.27, alpha=alpha, zorder=4)
        else:
            axis.triplot(tri, color="#adc0b6", linewidth=0.34, alpha=min(0.85, alpha + 0.2), zorder=4)
            axis.scatter(xy[:, 0], xy[:, 1], s=1.2, c="#f7fbff", alpha=0.52, linewidths=0, zorder=5)
        return True
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return False


def render_cosmic_web_story(density, environment, particles, box, redshift, snap,
                            run_name, outdir, visualization, deposition_metadata,
                            smoothing_cells, eigenvalue_threshold, *, scientific_source=None):
    """Render a 2x2 volumetric web plate and a 3x3 ridge-extraction plate."""
    from scipy.ndimage import gaussian_filter, maximum_filter

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    apply_plot_style(visualization)
    density = np.asarray(density, dtype=np.float32)
    raw_environment = np.asarray(environment)
    if raw_environment.dtype.kind not in "iu" or np.any(raw_environment < 0) or np.any(raw_environment > 3):
        raise ValueError("Morphology classes must be integer codes 0 through 3")
    environment = raw_environment.astype(np.uint8, copy=False)
    particles = np.asarray(particles, dtype=np.float32).reshape((-1, 3))
    n = int(density.shape[0])
    if n < 8 or density.shape != (n, n, n) or environment.shape != density.shape:
        raise ValueError("Density and environment fields must be aligned cubic arrays with at least eight cells per axis")
    if np.any(~np.isfinite(density)) or np.any(density < 0) or not np.any(density > 0):
        raise ValueError("Density must be finite, non-negative and contain positive mass")
    if not math.isfinite(box) or box <= 0 or np.any(~np.isfinite(particles)):
        raise ValueError("Figure coordinates need finite particles and a positive comoving box")

    display_density = gaussian_filter(density, sigma=0.75, mode="wrap")
    log_density = np.log1p(np.maximum(display_density, 0.0))
    density_limits = tuple(float(x) for x in np.percentile(log_density, [32.0, 99.5]))
    if density_limits[1] <= density_limits[0]:
        density_limits = (density_limits[0], density_limits[0] + 1.0)
    basis = _rotation_basis()
    side = int(math.ceil(math.sqrt(3.0) * n)) + 2
    rotated_density = _rotate_volume(display_density, basis, side, order=1).astype(np.float32, copy=False)
    rotated_inside = _rotate_volume(np.ones_like(density, dtype=np.uint8), basis, side, order=0) > 0
    rotated_environment = _rotate_volume(environment, basis, side, order=0).astype(np.uint8, copy=False)

    overview = outdir / "cosmic_web_3d_morphology"
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.0), layout=None)
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.07, top=0.91, wspace=0.025, hspace=0.12)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "The cosmic web in three dimensions  ·  %s  ·  z = %.3f" % (run_name, redshift),
        fontsize=15, weight="bold", color=_INK,
    )
    panel_specs = [
        (0, "Dark-matter density field", None, None),
        (1, "3-D void-like basins", 0, _PALETTE["void"]),
        (2, "2-D wall-like sheets", 1, _PALETTE["wall"]),
        (3, "1-D filament-like ridges", 2, _PALETTE["filament"]),
    ]
    short_tags = ["A", "B", "C", "D"]
    for index, (axis, spec, letter) in enumerate(zip(axes.flat, panel_specs, short_tags)):
        _class_id, title, class_value, color = spec
        if class_value is None:
            image = _ray_cast(rotated_density, rotated_inside, density_limits)
        else:
            selected = rotated_inside & (rotated_environment == class_value)
            class_values = log_density[environment == class_value]
            class_limits = tuple(float(x) for x in np.percentile(class_values, [5.0, 99.5])) if class_values.size else density_limits
            if class_limits[1] <= class_limits[0]:
                class_limits = (class_limits[0], class_limits[0] + 1.0)
            image = _ray_cast(rotated_density, selected, class_limits, base_color=color)
            if class_value == 2:
                nodes = rotated_inside & (rotated_environment == 3)
                if np.any(nodes):
                    node_values = log_density[environment == 3]
                    node_limits = tuple(float(x) for x in np.percentile(node_values, [5.0, 99.5])) if node_values.size else density_limits
                    if node_limits[1] <= node_limits[0]:
                        node_limits = (node_limits[0], node_limits[0] + 1.0)
                    node_image = _ray_cast(rotated_density, nodes, node_limits, base_color=_PALETTE["node"])
                    image = np.clip(image + 0.85 * node_image, 0.0, 1.0)
        axis.imshow(np.transpose(image, (1, 0, 2)), origin="lower", interpolation="lanczos")
        _cube_edges(axis, n, side, basis)
        axis.text(
            0.025, 0.965, letter, transform=axis.transAxes, va="top", ha="left",
            fontsize=10, fontweight="bold", color="#f0e8d6",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#252d30", "edgecolor": "#7b8785", "alpha": 0.94},
            zorder=8,
        )
        axis.text(
            0.14, 0.965, title, transform=axis.transAxes, ha="left", va="top",
            color="#f4f8ff", fontsize=9.1, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#252d30", "edgecolor": "#7b8785", "alpha": 0.90},
        )
        if class_value is None and len(particles):
            xyz = np.mod(particles, float(box))
            # Project a deterministic subset of particles at local density peaks
            # to restore the fine, star-like texture visible in the reference.
            cell = np.floor(xyz * (n / float(box))).astype(int) % n
            local_density = density[cell[:, 0], cell[:, 1], cell[:, 2]]
            cutoff = np.percentile(local_density, 88.0)
            bright = np.flatnonzero(local_density >= cutoff)
            if len(bright) > 5000:
                bright = bright[np.linspace(0, len(bright) - 1, 5000, dtype=int)]
            centered = xyz[bright] * (n / float(box)) - (n - 1.0) / 2.0
            screen = centered @ basis
            screen[:, :2] += (side - 1.0) / 2.0
            axis.scatter(screen[:, 0], screen[:, 1], s=0.8, c="#f4f8ff", alpha=0.45,
                         linewidths=0, zorder=5, rasterized=True)

    fig.text(
        0.5, 0.012,
        "Box: %g comoving Mpc/h. Brightness/opacity are density display choices. Panels B–D use density-Hessian morphology classes.\n"
        "These are morphology proxies, not persistence-defined Morse manifolds." % float(box),
        ha="center", color="#43516a", fontsize=8.0,
    )
    metadata = {
        "product": "3-D cosmic-web density and Hessian morphology plate",
        "snapshot": int(snap), "redshift": float(redshift), "run": str(run_name),
        "box_mpc_h": float(box), "grid_size": n,
        "density_assignment": deposition_metadata.get("method"),
        "density_particle_count": int(deposition_metadata.get("particle_count", len(particles))),
        "smoothing_sigma_cells": float(smoothing_cells),
        "volume_display_smoothing_sigma_cells": 0.75,
        "density_display_log1p_limits": list(density_limits),
        "volume_brightness": "density/opacity transfer function; not luminosity or a quantitative projected surface density",
        "hessian_collapse_threshold": float(eigenvalue_threshold),
        "method": "periodic CIC density; Gaussian-smoothed log-density Hessian; eigenvalue-sign morphology classes",
        "class_mapping": {"0": "void-like 3-D regions", "1": "wall-like 2-D regions", "2": "filament-like 1-D ridges", "3": "node-like peaks"},
        "interpretation": "Hessian morphology proxies, not persistent Morse manifolds or a DisPerSE skeleton.",
        "density_palette": "luminous navy-blue-cyan-mint-gold",
        "morphology_colors": _PALETTE,
    }
    if scientific_source is not None: metadata.update(scientific_source)
    export_figure(fig, overview, metadata=metadata)

    # 2-D sequence: particles and their density field; actual Delaunay
    # tessellations; progressively selective Hessian ridge masks; a node zoom.
    # Project the ALL-particle, mass-weighted 3-D CIC mesh. A reservoir of
    # plotted positions is for scatter/tessellation only and cannot define a
    # scientific density field (especially for unequal particle masses).
    map_size = n
    surface = np.mean(density, axis=2, dtype=np.float64)
    # CIC values live at periodic mesh nodes i*dx. The figure pixels live at
    # (i+1/2)*dx: bilinear interpolation to cell centres preserves total mass
    # and aligns the density/ridge/zoom overlays with particle coordinates.
    surface = 0.25 * (surface + np.roll(surface, -1, axis=0)
                      + np.roll(surface, -1, axis=1)
                      + np.roll(np.roll(surface, -1, axis=0), -1, axis=1))
    mean_surface = float(np.mean(surface))
    if mean_surface <= 0 or not np.all(np.isfinite(surface)):
        raise ValueError("Projected mass-density field must be finite with positive mean")
    surface = np.asarray(surface / mean_surface, dtype=np.float32)
    smooth_surface = gaussian_filter(surface, sigma=0.55, mode="wrap")
    log_surface = np.log10(np.maximum(smooth_surface, 1e-5))
    finite = log_surface[np.isfinite(log_surface)]
    lo, hi = np.percentile(finite, [0.5, 99.7]) if finite.size else (-1.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    norm = Normalize(float(lo), float(hi))
    cmap = _contrast_map()
    mesh_centers = (np.arange(map_size, dtype=float) + 0.5) * float(box) / map_size

    # Select progressively stronger projected density-Hessian ridges. These
    # are curvature/density cuts, not persistence significance thresholds.
    ridge_masks, ridge_response_cuts = _projected_hessian_ridges(surface)

    # Select the densest smoothed projected cell for a reproducible detail view.
    peak_index = np.unravel_index(int(np.argmax(smooth_surface)), smooth_surface.shape)
    center = (float(mesh_centers[peak_index[0]]), float(mesh_centers[peak_index[1]]))
    zoom_width = max(float(box) * 0.28, 24.0 * float(box) / map_size)

    sequence = outdir / "filament_detection_sequence"
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 12.0), layout=None)
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.145, top=0.91,
                        wspace=0.085, hspace=0.28)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "From particles to the filament network  ·  %s  ·  z = %.3f" % (run_name, redshift),
        fontsize=15, weight="bold", color=_INK,
    )
    axes[0, 0].set_facecolor(_DARK)
    _map_frame(axes[0, 0], "1  ·  Discrete dark-matter particles",
               extent=(0.0, float(box), 0.0, float(box)))
    if len(particles):
        axes[0, 0].scatter(
            particles[:, 0], particles[:, 1], s=0.62, c="#eaf7ff", alpha=0.60,
            linewidths=0, rasterized=True, zorder=3,
        )
    axes[0, 1].imshow(
        log_surface.T, origin="lower", extent=(0, box, 0, box), cmap=cmap,
        norm=norm, interpolation="nearest", aspect="equal", rasterized=True,
    )
    axes[0, 1].set_title("2  ·  Projected surface-density field", color=_INK, fontsize=9,
                         pad=5, loc="left", weight="bold")
    axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])
    for spine in axes[0, 1].spines.values():
        spine.set_visible(True); spine.set_color("#141b2a"); spine.set_linewidth(1.05)
    _delaunay_overlay(axes[0, 1], particles, log_surface, box, alpha=0.23, max_points=1100)

    ax = axes[0, 2]
    _map_frame(ax, "3  ·  Local Delaunay tessellation",
               extent=(0.0, zoom_width, 0.0, zoom_width))
    detail_density = _periodic_crop(log_surface, center, box, zoom_width)
    ax.imshow(detail_density.T, origin="lower", extent=(0.0, zoom_width, 0.0, zoom_width),
              cmap=cmap, norm=norm, interpolation="nearest", aspect="equal", rasterized=True)
    _map_frame(ax, "3  ·  Local Delaunay tessellation",
               extent=(0.0, zoom_width, 0.0, zoom_width))
    _delaunay_overlay(ax, particles, log_surface, box, center=center, width=zoom_width,
                      norm=norm, cmap=cmap, alpha=0.58, max_points=1900)

    row_titles = [
        "4  ·  Inclusive ridge candidates",
        "5  ·  Reference ridge network",
        "6  ·  High-density ridge cores",
    ]
    colors = [_RIDGE, _RIDGE, _RIDGE]
    for column in range(3):
        ax = axes[1, column]
        ax.imshow(log_surface.T, origin="lower", extent=(0, box, 0, box), cmap=cmap,
                  norm=norm, interpolation="nearest", aspect="equal", rasterized=True)
        mask = ridge_masks[column]
        rgba = np.zeros(mask.shape + (4,), dtype=float)
        color = to_rgba(colors[column])
        rgba[mask] = color
        rgba[mask, 3] = 0.97
        ax.imshow(rgba.transpose(1, 0, 2), origin="lower", extent=(0, box, 0, box),
                  interpolation="nearest", aspect="equal", rasterized=True, zorder=3)
        _map_frame(ax, row_titles[column], extent=(0.0, float(box), 0.0, float(box)))

    # Three matching density-peak crops show the selected web more clearly at each
    # stage; the final panel adds projected density peaks and troughs.
    detail_titles = [
        "7  ·  Projected density-peak zoom",
        "8  ·  Refined filament trace",
        "9  ·  Projected density extrema",
    ]
    detail_widths = [zoom_width, zoom_width * 0.72, zoom_width * 0.52]
    for column in range(3):
        ax = axes[2, column]
        detail_width = detail_widths[column]
        detail_density = _periodic_crop(log_surface, center, box, detail_width)
        detail_mask = _periodic_crop(ridge_masks[min(column, 2)], center, box, detail_width)
        ax.imshow(
            detail_density.T, origin="lower", extent=(0.0, detail_width, 0.0, detail_width),
            cmap=cmap, norm=norm, interpolation="nearest", aspect="equal", rasterized=True,
        )
        rgba = np.zeros(detail_mask.shape + (4,), dtype=float)
        rgba[detail_mask] = (*to_rgba(_RIDGE)[:3], 0.98)
        ax.imshow(rgba.transpose(1, 0, 2), origin="lower", extent=(0.0, detail_width, 0.0, detail_width),
                  interpolation="nearest", aspect="equal", rasterized=True, zorder=3)
        if column == 2:
            local_max = smooth_surface == maximum_filter(smooth_surface, size=5, mode="wrap")
            local_min = smooth_surface == -maximum_filter(-smooth_surface, size=5, mode="wrap")
            max_cut = np.percentile(smooth_surface, 98.5)
            min_cut = np.percentile(smooth_surface, 7.5)
            for condition, marker, color, size in (
                (local_max & (smooth_surface >= max_cut), "^", "#c5aa72", 29),
                (local_min & (smooth_surface <= min_cut), "x", "#b98268", 28),
            ):
                ix, iy = np.nonzero(condition)
                if len(ix) > 24:
                    order = np.argsort(smooth_surface[ix, iy])
                    if marker == "^":
                        order = order[::-1]
                    order = order[:24]
                    ix, iy = ix[order], iy[order]
                dx = mesh_centers[ix] - center[0]
                dy = mesh_centers[iy] - center[1]
                dx -= float(box) * np.round(dx / float(box))
                dy -= float(box) * np.round(dy / float(box))
                in_detail = (np.abs(dx) <= detail_width / 2) & (np.abs(dy) <= detail_width / 2)
                marker_options = {"edgecolors": "#09101f", "linewidths": 0.5} if marker == "^" else {"linewidths": 0.8}
                ax.scatter(dx[in_detail] + detail_width / 2, dy[in_detail] + detail_width / 2,
                           marker=marker, s=size, c=color, zorder=6, **marker_options)
        _map_frame(ax, detail_titles[column], extent=(0.0, detail_width, 0.0, detail_width))

    _map_frame(axes[0, 0], "1  ·  Discrete dark-matter particles", extent=(0.0, float(box), 0.0, float(box)))
    _map_frame(axes[0, 1], "2  ·  Projected density + Delaunay mesh", extent=(0.0, float(box), 0.0, float(box)))
    # One key serves all density panels because they share the same limits.
    # Scatter/tessellation/ridge glyph colours are separate display encodings.
    for axis in axes.flat:
        width = float(axis.get_xlim()[1]-axis.get_xlim()[0])
        target = width/4.
        decade = 10.**math.floor(math.log10(target))
        bar = max(value*decade for value in (1., 2., 5.) if value*decade <= target)
        axis.plot([width*.05, width*.05+bar], [width*.06]*2, color="white", linewidth=2.2, zorder=10)
        axis.text(width*.05, width*.095, "%g comoving Mpc/h" % bar, color="white", fontsize=6.5,
                  ha="left", va="bottom", zorder=10,
                  bbox={"facecolor": "#09101f", "edgecolor": "none", "alpha": .7, "pad": 1.4})
    key_axis = fig.add_axes((.37, .106, .26, .010))
    key = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=key_axis,
                       orientation="horizontal", extend="both", ticks=np.linspace(lo, hi, 3))
    key.ax.tick_params(labelsize=7, length=2, pad=1)
    key.set_label(r"Projected density: $\log_{10}(\Sigma/\langle\Sigma\rangle)$", fontsize=8, labelpad=2)
    fig.text(
        0.5, 0.007,
        "Warm traces are projected 2-D density-Hessian ridges selected at progressively stricter curvature and density cuts.\n"
        "Markers are 2-D projected extrema, not 3-D Morse critical points or DisPerSE persistence pairs.",
        ha="center", color="#43516a", fontsize=7.8,
    )
    sequence_metadata = {
        **metadata,
        "product": "nine-panel particle-to-filament morphology sequence",
        "views": ["sampled particle distribution", "all-particle mass-weighted projected CIC density", "local Delaunay triangulation",
                  "inclusive, reference, and high-density Hessian ridge masks", "projected density-peak zoom", "projected extrema"],
        "particle_plot_count": int(len(particles)),
        "ridge_response_cut_values": ridge_response_cuts,
        "ridge_candidate_pixel_counts": [int(np.count_nonzero(mask)) for mask in ridge_masks],
        "projected_grid": "mass-preserving periodic bilinear interpolation from CIC nodes to pixel centres",
        "projected_density_grid_size": int(map_size),
        "projected_density_source": "line-of-sight mean of the all-particle mass-weighted CIC mesh; normalized by its mean",
        "surface_display_smoothing_cells": 0.55,
        "ridge_method": "skeletonized 2-D ridges of the periodic particle-projected log-density Hessian",
        "ridge_significance": "curvature and density thresholds, not persistence sigma",
        "zoom_center_mpc_h": [float(x) for x in center],
        "zoom_width_mpc_h": float(zoom_width),
        "detail_widths_mpc_h": list(detail_widths),
        "zoom_resampling": "periodic pixel-centre bilinear scalar/nearest-mask display interpolation",
        "projected_density_log10_colour_limits": [float(lo), float(hi)],
        "projected_density_display_percentiles": [0.5, 99.7],
        "interpretation": "Illustrative Hessian ridge extraction; not persistence significance, Morse extrema, or a DisPerSE skeleton.",
    }
    export_figure(fig, sequence, metadata=sequence_metadata)

    # Retire the previous generic 2-D figures only after both replacements
    # have been written successfully, so a failed render preserves old output.
    for obsolete in (
        "cosmic_web_environment_overview", "cosmic_web_orthogonal_projections",
        "cosmic_web_3d_manifolds",
    ):
        for suffix in ("png", "pdf", "svg", "figure.json"):
            (outdir / (obsolete + "." + suffix)).unlink(missing_ok=True)

    formats = visualization.get("figure_formats", ["png", "pdf", "svg"])
    if not isinstance(formats, (list, tuple)) or not formats:
        formats = ["png", "pdf", "svg"]
    formats = list(dict.fromkeys([str(x).lower().lstrip(".") for x in formats if str(x).lower().lstrip(".") in {"png", "pdf", "svg"}] + ["png"]))
    return [
        {"status": "COMPLETE", "stem": overview.name,
         "files": [overview.with_suffix("." + fmt).name for fmt in formats],
         "snapshot": int(snap), "redshift": float(redshift), "grid_size": n},
        {"status": "COMPLETE", "stem": sequence.name,
         "files": [sequence.with_suffix("." + fmt).name for fmt in formats],
         "snapshot": int(snap), "redshift": float(redshift), "grid_size": n},
    ]


def _map_frame(axis, title, extent):
    axis.set_title(title, color=_INK, fontsize=9, pad=5, loc="left", weight="bold")
    axis.set_xlim(extent[:2])
    axis.set_ylim(extent[2:])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor(_DARK)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#141b2a")
        spine.set_linewidth(1.05)

# ========================================================================
# inference_figures: Reusable measured-sample, posterior and prediction diagnostics.
from matplotlib import pyplot as inference_figures_plt


def normalized_weights(weights, count):
    if weights is None:
        return np.full(count, 1 / count)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (count,) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.max() <= 0:
        raise ValueError('Weights must align, be finite/nonnegative and include positive mass')
    # Scaling first avoids overflow with valid large finite weights.
    weights = weights / weights.max()
    return weights / weights.sum()


def weighted_quantiles(values, quantiles, weights):
    """Inverse empirical weighted CDF; zero-weight samples have no influence."""
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Quantiles require a finite one-dimensional sample')
    if not np.isfinite(quantiles).all() or np.any((quantiles < 0) | (quantiles > 1)):
        raise ValueError('Quantiles must be probabilities in [0,1]')
    w = normalized_weights(weights, len(values))
    keep = w > 0
    order = np.argsort(values[keep], kind='stable')
    x, w = values[keep][order], w[keep][order]
    cdf = np.cumsum(w)
    cdf[-1] = 1.
    return x[np.minimum(np.searchsorted(cdf, quantiles, side='left'), len(x) - 1)]


def _context(provenance):
    if not isinstance(provenance, dict) or not provenance.get('source') or not provenance.get('selection'):
        raise ValueError('Figure provenance must declare source and selection')
    return dict(provenance)


def corner_figure(samples, labels, units, destination, *, sample_kind, provenance,
                  weights=None, truth=None, bins=32):
    """Lower-triangle pair densities and marginals for any measured parameters.

    sample_kind='posterior' is reserved for samples from a declared fit.
    Catalogue correlations use sample_kind='population'; bootstrap samples use
    'bootstrap'. Weighted 68/95% regions are binned highest-density regions.
    Constant columns are rendered explicitly without ill-defined contours.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 3 or not 1 <= samples.shape[1] <= 16 or not np.isfinite(samples).all():
        raise ValueError('Samples must be finite N x D, N>=3 and 1<=D<=16')
    n, d = samples.shape
    if len(labels) != d or len(units) != d or any(not isinstance(x, str) or not x for x in list(labels) + list(units)):
        raise ValueError('Every parameter requires a name and units')
    if sample_kind not in {'posterior', 'population', 'bootstrap'}:
        raise ValueError('Declare posterior, population or bootstrap sample kind')
    provenance = _context(provenance)
    if sample_kind == 'posterior' and (not provenance.get('model') or not provenance.get('sampler')):
        raise ValueError('Posterior samples require model and sampler provenance')
    if type(bins) is not int or not 4 <= bins <= 128:
        raise ValueError('Histogram bins must be an integer in [4,128]')
    w = normalized_weights(weights, n)
    truth = None if truth is None else np.asarray(truth, dtype=float)
    if truth is not None and (truth.shape != (d,) or not np.isfinite(truth).all()):
        raise ValueError('Truth must be finite and align to parameters')
    fig, axes = inference_figures_plt.subplots(d, d, figsize=(max(4, 1.8*d), max(4, 1.8*d)), squeeze=False)
    quantiles = []
    for j in range(d):
        quantiles.append(weighted_quantiles(samples[:, j], [.16, .5, .84], w).tolist())
        for i in range(d):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                ax.hist(samples[:, j], weights=w, bins=bins, density=True, histtype='step', color='#0072B2')
                ax.set_title(f'{quantiles[j][1]:.3g} [{quantiles[j][0]:.3g}, {quantiles[j][2]:.3g}]', fontsize=7)
                if truth is not None:
                    ax.axvline(truth[j], color='#D55E00', lw=.9)
            else:
                hist, xedges, yedges = np.histogram2d(samples[:, j], samples[:, i], bins=bins, weights=w)
                ax.pcolormesh(xedges, yedges, hist.T, cmap='Blues', shading='flat', rasterized=True)
                if np.ptp(samples[w > 0, j]) > 0 and np.ptp(samples[w > 0, i]) > 0:
                    ranked = np.sort(hist.ravel())[::-1]
                    cumulative = np.cumsum(ranked) / ranked.sum()
                    levels = sorted(set(float(ranked[min(np.searchsorted(cumulative, p), len(ranked)-1)]) for p in (.68, .95)))
                    levels = [level for level in levels if hist.min() < level < hist.max()]
                    if levels:
                        ax.contour((xedges[:-1]+xedges[1:])/2, (yedges[:-1]+yedges[1:])/2,
                                   hist.T, levels=levels, colors='#0072B2', linewidths=.7)
                if truth is not None:
                    ax.axvline(truth[j], color='#D55E00', lw=.7)
                    ax.axhline(truth[i], color='#D55E00', lw=.7)
            if i == d-1:
                ax.set_xlabel(f'{labels[j]} [{units[j]}]', fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i > 0:
                ax.set_ylabel(f'{labels[i]} [{units[i]}]', fontsize=8)
            else:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=6)
    fig.suptitle(f'{sample_kind.capitalize()} parameter distributions', fontsize=11)
    fig.tight_layout()
    meta = {'kind': sample_kind, 'provenance': provenance, 'parameters': list(labels), 'units': list(units),
            'quantiles_16_50_84': quantiles, 'sample_count': n,
            'weight_effective_count': float(1/np.sum(w*w)),
            'effective_count_definition': 'weight concentration only; does not account for MCMC autocorrelation',
            'truth': None if truth is None else truth.tolist(),
            'regions': '68/95 percent binned highest-density probability regions; no Gaussian assumption'}
    return export_figure(fig, destination, metadata=meta, layout='presentation')


def predictive_figure(coordinate, observed, uncertainty, predictions, destination, *,
                      coordinate_label, observable_label, provenance, weights=None):
    """Prediction envelopes and normalized residuals for any 1-D observable.

    Samples are conditional model means at the measured coordinates. Bands
    describe parameter uncertainty, not noise-added predictive draws. Residual
    sigma must include the declared observation/model error policy.
    """
    x, y, sigma = [np.asarray(v, dtype=float) for v in (coordinate, observed, uncertainty)]
    predictions = np.asarray(predictions, dtype=float)
    if (x.ndim != 1 or len(x) < 2 or y.shape != x.shape or sigma.shape != x.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(sigma).all()
            or np.any(sigma <= 0) or np.any(np.diff(x) <= 0)):
        raise ValueError('Measured coordinates must increase; observations and positive sigma must align')
    if predictions.ndim != 2 or predictions.shape[1] != len(x) or len(predictions) < 3 or not np.isfinite(predictions).all():
        raise ValueError('Predictions must be finite sample x measured-coordinate arrays')
    provenance = _context(provenance)
    if not provenance.get('model') or not provenance.get('uncertainty_definition'):
        raise ValueError('Prediction provenance requires model and uncertainty_definition')
    w = normalized_weights(weights, len(predictions))
    quantiles = np.array([weighted_quantiles(predictions[:, i], [.025, .16, .5, .84, .975], w) for i in range(len(x))]).T
    residuals = (predictions - y) / sigma
    rq = np.array([weighted_quantiles(residuals[:, i], [.025, .16, .5, .84, .975], w) for i in range(len(x))]).T
    fig, axes = inference_figures_plt.subplots(1, 3, figsize=(12.8, 3.5))
    axes[0].errorbar(x, y, yerr=sigma, fmt='.', color='#D55E00', label='Measured')
    for ax, q in zip(axes[:2], (quantiles, rq)):
        ax.fill_between(x, q[0], q[4], color='#0072B2', alpha=.13, label='95%')
        ax.fill_between(x, q[1], q[3], color='#0072B2', alpha=.3, label='68%')
        ax.plot(x, q[2], color='#0072B2', label='Median')
        ax.set_xlabel(coordinate_label)
    axes[0].set_ylabel(observable_label)
    axes[0].set_title('Conditional model predictions')
    axes[0].legend(fontsize=7)
    axes[1].axhline(0, color='#222222', lw=.8)
    axes[1].axhspan(-1, 1, color='gray', alpha=.12)
    axes[1].set_ylabel('(model − measured) / σ')
    axes[1].set_title('Normalized residuals')
    axes[2].hist(rq[2], bins=min(30, len(x)), histtype='step', color='#0072B2')
    axes[2].set_xlabel('Median normalized residual')
    axes[2].set_ylabel('Measurement bins')
    axes[2].set_title('Residual distribution')
    fig.tight_layout()
    return export_figure(fig, destination, metadata={
        'kind': 'conditional_model_predictions', 'provenance': provenance,
        'coordinate': x.tolist(), 'observed': y.tolist(), 'uncertainty': sigma.tolist(),
        'prediction_quantiles_025_16_50_84_975': quantiles.tolist(),
        'normalized_residual_quantiles_025_16_50_84_975': rq.tolist(),
        'sample_count': len(predictions), 'bands': 'conditional model means; excludes added observation noise'},
        layout='presentation')


def export_diagnostics(request):
    """Atomically publish supplied scientific arrays, portable inputs and figures."""
    import json
    import tempfile
    from core.foundation import file_sha256 as digest, atomic_json, atomic_copy, publish_directory
    from core.native import native_run_lock
    required = {'samples_npz', 'metadata_json', 'output_directory'}
    if not isinstance(request, dict) or set(request) != required:
        raise ValueError('Diagnostics request requires samples_npz, metadata_json, output_directory')
    source = Path(request['samples_npz']).expanduser().resolve()
    meta_path = Path(request['metadata_json']).expanduser().resolve()
    out = Path(request['output_directory']).expanduser().resolve()
    with native_run_lock(out,operation='immutable inference diagnostics'):
        if out.exists():
            raise FileExistsError('Refusing to overwrite inference diagnostics: ' + str(out))
        before = {'samples':digest(source),'metadata':digest(meta_path)}
        metadata = json.loads(meta_path.read_text())
        provenance = dict(metadata['provenance'], source='inputs/samples.npz',
                          source_sha256=before['samples'], metadata_sha256=before['metadata'])
        out.parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'-stage-',dir=out.parent))
        try:
            inputs=stage/'inputs';inputs.mkdir()
            if atomic_copy(source,inputs/'samples.npz')!=before['samples']:
                raise ValueError('Inference sample source changed before copying')
            if atomic_copy(meta_path,inputs/'metadata.json')!=before['metadata']:
                raise ValueError('Inference metadata source changed before copying')
            with np.load(inputs/'samples.npz', allow_pickle=False) as saved:
                samples = saved['samples']
                weights = saved['weights'] if 'weights' in saved else None
                truth = saved['truth'] if 'truth' in saved else None
                keys = {'coordinate', 'observed', 'uncertainty', 'predictions'}
                present = keys & set(saved.files)
                if present and present != keys:
                    raise ValueError('Prediction diagnostics require coordinate, observed, uncertainty and predictions together')
                records = {'corner': corner_figure(samples, metadata['labels'], metadata['units'], stage/'corner',
                    sample_kind=metadata['sample_kind'], provenance=provenance, weights=weights, truth=truth)}
                if present:
                    records['predictions'] = predictive_figure(saved['coordinate'], saved['observed'], saved['uncertainty'],
                        saved['predictions'], stage/'predictions', coordinate_label=metadata['coordinate_label'],
                        observable_label=metadata['observable_label'], provenance=provenance, weights=weights)
                specifications={}
                for name in saved.files:
                    array=saved[name]
                    if array.dtype.kind not in 'biufc' or array.dtype.hasobject:
                        raise ValueError('Inference arrays must be numeric without object deserialization')
                    unit=(metadata['units'] if name in ('samples','truth') else
                          'dimensionless' if name=='weights' else metadata.get(name+'_units'))
                    specifications[name]={'units':'heterogeneous parameter columns' if isinstance(unit,list) else unit or 'UNDECLARED',
                        'quantity':metadata['sample_kind']+' diagnostic '+name,
                        'missing_policy':'No identity or ancestry implied by sample row',
                        'annotations':{'sample_kind':metadata['sample_kind'],'column_units':unit if isinstance(unit,list) else None,
                            'columns':metadata['labels'] if name in ('samples','truth') else None,
                            'identity':'sample index is not a particle or object identity'}}
                from core.data import numpy_archive_contract
                array_contract=numpy_archive_contract(inputs/'samples.npz',specifications,
                    epoch=metadata.get('epoch'),coordinates=metadata.get('coordinates'),provenance=provenance)
            if digest(source)!=before['samples'] or digest(meta_path)!=before['metadata']:
                raise ValueError('Inference source changed during diagnostics')
            relative_records={name:[str(Path(path).relative_to(stage)) for path in paths] for name,paths in records.items()}
            artifacts=[{'path':str(path.relative_to(stage)),'sha256':digest(path)}
                       for path in sorted(stage.rglob('*')) if path.is_file()]
            record={'schema':'inference-diagnostics-v2','status':'COMPLETE','science_status':'NOT_ACCEPTED',
                    'provenance':provenance,'figures':relative_records,'output_artifacts':artifacts,
                    'data_contract':array_contract,
                    'algorithm_sha256':digest(__file__)}
            atomic_json(stage/'diagnostics.json',record)
            (stage/'diagnostics.sha256').write_text(digest(stage/'diagnostics.json')+'\n')
            publish_directory(stage,out)
        except BaseException as exc:
            if stage.exists():
                atomic_json(stage/'failure.json',{'status':'FAILED','type':type(exc).__name__,'message':str(exc)})
            raise
        # Existing callers receive usable paths; the saved receipt stays portable.
        return dict(record,figures={name:[str(out/path) for path in paths] for name,paths in relative_records.items()})


def verify_diagnostics(directory):
    """Verify saved inference inputs and artifacts before later reuse."""
    import json
    from core.foundation import file_sha256 as digest
    from core.data import verify_numpy_archive_contract
    root=Path(directory).resolve()
    seal=root/'diagnostics.sha256'
    if seal.exists() and digest(root/'diagnostics.json')!=seal.read_text().strip():
        raise ValueError('Inference diagnostics checksum mismatch')
    record=json.loads((root/'diagnostics.json').read_text())
    if record.get('schema')!='inference-diagnostics-v2':raise ValueError('Unsupported inference diagnostics')
    for item in record['output_artifacts']:
        relative=Path(item['path']);path=root/relative
        if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(root):
            raise ValueError('Inference artifact path escapes bundle')
        if digest(path)!=item['sha256']:raise ValueError('Inference artifact checksum mismatch')
    contract=record['data_contract']
    if contract.get('container')=='pickle-free NPZ':verify_numpy_archive_contract(root/'inputs/samples.npz',contract)
    return record
from core.native import directory_owner
from core.foundation import atomic_json, file_sha256 as digest

import h5py


_WEB_LABELS = ["Void-like", "Wall-like", "Filament-like", "Node-like"]


def _figure_formats(visualization):
    """Resolve the declared publication export formats while retaining a PNG preview."""
    raw = visualization.get("figure_formats", ["png", "pdf", "svg"])
    if not isinstance(raw, (list, tuple)) or not raw:
        raw = ["png", "pdf", "svg"]
    formats = [str(value).lower().lstrip(".") for value in raw]
    formats = [value for value in formats if value in {"png", "pdf", "svg"}]
    if "png" not in formats:
        formats.append("png")
    return list(dict.fromkeys(formats))


def _sample_dark_matter_positions(files, box, maximum=30_000, seed=1847, *, return_ids=False):
    """Return a deterministic uniform sample of PartType1 positions in Mpc/h."""
    from core.data import convert_pinned_native, exact_uint64_ids
    if isinstance(maximum, bool) or int(maximum) != maximum or maximum < 1:
        raise ValueError("Figure sample budget must be a positive integer")
    if not np.isfinite(box) or box <= 0:
        raise ValueError("Figure sample box must be finite and positive")
    rng = np.random.default_rng(int(seed))
    retained_positions = np.empty((0, 3), dtype=np.float64)
    retained_ids = np.empty(0, dtype=np.uint64)
    retained_keys = np.empty(0, dtype=np.float64)
    seen = 0
    for path in files:
        with h5py.File(path, "r") as handle:
            particles = handle.get("PartType1")
            if particles is None or "Coordinates" not in particles:
                continue
            coordinates = particles["Coordinates"]
            ids = particles.get("ParticleIDs")
            if ids is None or ids.dtype != np.dtype("uint64") or ids.shape != (len(coordinates),):
                raise ValueError("Figure sample requires aligned exact uint64 ParticleIDs")
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise ValueError("PartType1/Coordinates must have shape (particles, 3)")
            for start in range(0, len(coordinates), 250_000):
                stop = min(start + 250_000, len(coordinates))
                chunk = convert_pinned_native(coordinates[start:stop], "position_comoving_mpc_h")
                chunk_ids = exact_uint64_ids(ids[start:stop])
                if np.any(~np.isfinite(chunk)):
                    raise ValueError("Non-finite PartType1 coordinates in scientific figure input: " + str(path))
                chunk = np.mod(chunk, float(box))
                if not len(chunk):
                    continue
                keys = rng.random(len(chunk))
                merged_keys = np.concatenate((retained_keys, keys))
                merged_positions = np.concatenate((retained_positions, chunk), axis=0)
                merged_ids = np.concatenate((retained_ids, chunk_ids))
                count = min(int(maximum), len(merged_keys))
                if len(merged_keys) > count:
                    chosen = np.argpartition(merged_keys, count - 1)[:count]
                    retained_keys = merged_keys[chosen]
                    retained_positions = merged_positions[chosen]
                    retained_ids = merged_ids[chosen]
                else:
                    retained_keys, retained_positions, retained_ids = merged_keys, merged_positions, merged_ids
                seen += len(chunk)
    exact_uint64_ids(retained_ids)
    if return_ids: return retained_positions, retained_ids
    return retained_positions.astype(np.float32)


def plot_saved_analysis_curves(analysis_h5, output_dir, *, final_source_name=None):
    """Read values AND producer unit declarations from saved HMF/P(k) datasets."""
    from figures import export_figure
    from core.foundation import file_sha256
    source=Path(analysis_h5).resolve()
    from core.data import verify_hdf5_dataset_payloads
    verify_hdf5_dataset_payloads(source)
    import os
    source_reference=final_source_name or os.path.relpath(source,Path(output_dir).resolve())
    before=file_sha256(source)
    figures=[]
    with h5py.File(source) as handle:
        for snapshot in sorted(handle):
            if not snapshot.startswith('snap_'):
                continue
            for product,xkey,ykey in [('hmf','mass_msun_h','dn_dlogM'),('pk','k_h_Mpc','P_Mpc_h3')]:
                path=snapshot+'/'+product+'/data'
                error_key='P_gaussian_1sigma_Mpc_h3'
                if path not in handle:
                    path=snapshot
                    if product=='hmf':
                        xkey,ykey='hmf_mass','hmf_dn_dlogM'
                    else:
                        xkey,ykey='pk_k','pk_P'
                        error_key='pk_P_gaussian_1sigma'
                if xkey not in handle[path] or ykey not in handle[path]:
                    continue
                group=handle[path]
                for key in [xkey,ykey]:
                    if group[key].attrs.get('units_status') != 'PRODUCER_DECLARED' or not group[key].attrs.get('units'):
                        raise ValueError('Saved figure requires producer-declared units: '+group[key].name)
                x,y=np.asarray(group[xkey][:]),np.asarray(group[ykey][:])
                if x.ndim!=1 or x.shape!=y.shape or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(x<=0):
                    raise ValueError('Saved HMF/P(k) curve requires aligned finite bins and positive abscissae')
                figure,axis=plt.subplots(figsize=(8,5))
                axis.set_xscale('log')
                if np.all(y>0):
                    axis.set_yscale('log')
                axis.plot(x,y,'o-',markersize=3,label='saved '+product+' measurement')
                if product=='pk' and error_key in group:
                    error=group[error_key]
                    if error.attrs.get('units')!=group[ykey].attrs['units']:
                        raise ValueError('Saved power uncertainty has incompatible units')
                    sigma=np.asarray(error[:])
                    if sigma.shape!=y.shape or np.any(~np.isfinite(sigma)) or np.any(sigma<0):
                        raise ValueError('Saved power uncertainty is invalid')
                    axis.fill_between(x,y-sigma,y+sigma,alpha=.2,label='Gaussian mode-counting uncertainty')
                axis.set_xlabel(xkey+' ['+group[xkey].attrs['units']+']')
                axis.set_ylabel(ykey+' ['+group[ykey].attrs['units']+']')
                epoch=json.loads(handle[snapshot].attrs['epoch_contract_json'])
                axis.set_title(product.upper()+f" | a={epoch['scale_factor']:.6g} | z={epoch['redshift']:.6g}")
                axis.legend()
                output=Path(output_dir)/(product+'_saved_'+snapshot+'.png')
                export_figure(figure,output,metadata={'source_hdf5':source_reference,'source_hdf5_sha256':before,
                    'datasets':[group[xkey].name,group[ykey].name],
                    'axis_units':{'x':group[xkey].attrs['units'],'y':group[ykey].attrs['units']},
                    'unit_source':'saved HDF5 producer declarations','epoch_contract':epoch,
                    'nonpositive_ordinate_count':int(np.count_nonzero(y<=0)),
                    'science_acceptance':'NOT_ACCEPTED','limits':'No numerical convergence or representative-pilot acceptance'})
                figures.append(output.name)
    if file_sha256(source)!=before:
        raise ValueError('Saved analysis changed while rendering figures')
    return figures


def _generate_single_run_story_figures(analysis_h5, output_dir, visualization=None, source_run=None):
    """Create high-quality, method-labelled cosmic-web figures for an analyzed run."""
    from core.data import verify_hdf5_dataset_payloads
    verify_hdf5_dataset_payloads(analysis_h5)
    visualization = visualization if isinstance(visualization, dict) else {}
    output_dir = Path(output_dir)
    result = {
        "status": "UNAVAILABLE", "figures": [],
        "method": "periodic CIC density with 3-D morphology views and a Hessian ridge extraction sequence",
    }
    try:
        source = Path(source_run).resolve() if source_run is not None else output_dir.resolve().parent.parent
        from analysis import find_hdf5_files, read_snapshot_header, snapshot_ids
        from visualization import _particle_mass_density_grid
        from scipy.ndimage import gaussian_filter, uniform_filter

        run_label = source.name
        try:
            with h5py.File(analysis_h5, "r") as analysis:
                recorded_source = analysis.attrs.get("source_run")
                if recorded_source:
                    run_label = Path(str(recorded_source)).name
        except (OSError, TypeError, ValueError):
            pass

        snapshots = snapshot_ids(source)
        if not snapshots:
            result["reason"] = "no numbered GADGET snapshots were found"
            return result
        candidates = []
        for snap in snapshots:
            header, _ = read_snapshot_header(source, snap)
            scale = float(np.asarray(header.get("Time", 1.0)).reshape(-1)[0])
            if np.isfinite(scale) and scale > 0:
                candidates.append((1.0 / scale - 1.0, int(snap)))
        if not candidates:
            result["reason"] = "snapshot headers do not contain a valid scale factor"
            return result
        redshift, snap = min(candidates, key=lambda row: row[0])
        cfg_path = source / "config.yaml"
        with cfg_path.open("r", encoding="utf-8") as stream:
            import yaml
            cfg = yaml.safe_load(stream) or {}
        box = float(cfg.get("simulation", {}).get("box_mpc_h", 0.0))
        if not np.isfinite(box) or box <= 0:
            result["reason"] = "config.yaml has no finite positive simulation.box_mpc_h"
            return result
        files = find_hdf5_files(source / "data", "snapshot_%03d" % snap)
        if not files:
            result["reason"] = "the selected snapshot has no readable HDF5 shards"
            return result
        from analysis import validate_gadget_snapshot_set
        from halos import _unit_conversion
        from core.data import coordinate_contract, run_epoch_contract, hdf5_scalar_attr
        from core.foundation import product_provenance, verify_product_provenance
        from core.foundation import physical_constants_contract
        from core.data import export_numeric_audit
        validate_gadget_snapshot_set(files, snap)
        _unit_conversion(files)
        coordinates = coordinate_contract(position_units='comoving Mpc/h', origin='global simulation Cartesian box origin',
            periodic_wrapping='native fundamental box; periodic CIC mesh', velocity_convention='not applicable to density morphology')
        provenance = product_provenance(source, algorithm=__file__, snapshot=snap,
            helper_sources=[Path(__file__).with_name('figures.py')],
            source_files=[Path(analysis_h5).resolve(), *files], coordinates=coordinates,
            selection={'visualization': visualization, 'product': 'density-Hessian morphology proxy'})
        epoch = run_epoch_contract(source, snap, hdf5_scalar_attr(read_snapshot_header(source, snap)[0]['Time'], 'Time'))
        # Use roughly two mesh cells per particle-spacing scale, bounded by the
        # configured ceiling. CIC smoothing keeps the field finite when the
        # mesh is finer than the mean particle spacing.
        header, _ = read_snapshot_header(source, snap)
        from core.data import snapshot_total_counts
        counts = snapshot_total_counts(header)
        dm_count = int(counts[1]) if counts.size > 1 else 0
        configured_grid = visualization.get("cosmic_web_grid_size", 128)
        try:
            configured_grid = 128 if isinstance(configured_grid, bool) else int(configured_grid)
        except (TypeError, ValueError, OverflowError):
            configured_grid = 128
        target_grid = min(128, max(24, int(configured_grid)))
        particle_detail_grid = int(math.ceil(max(dm_count, 1) ** (1.0 / 3.0) * 3.0 / 8.0) * 8)
        grid_size = min(target_grid, max(min(32, target_grid), particle_detail_grid))
        density, deposition = _particle_mass_density_grid(
            files, 1, box, grid_size, return_density_ratio=True,
        )
        density = np.asarray(density, dtype=np.float32)
        if density.ndim != 3 or len(set(density.shape)) != 1 or not np.all(np.isfinite(density)):
            raise ValueError("the mass-density mesh is not a finite cubic array")
        if not np.any(density > 0):
            raise ValueError("the selected snapshot has no positive dark-matter density cells")
        smoothing = visualization.get("cosmic_web_smoothing_cells", 1.25)
        try:
            smoothing = float(smoothing)
        except (TypeError, ValueError, OverflowError):
            smoothing = 1.25
        if not np.isfinite(smoothing):
            smoothing = 1.25
        smoothing = min(4.0, max(0.5, smoothing))
        scalar = np.log1p(np.maximum(density, 0.0))
        hessian = np.empty(density.shape + (3, 3), dtype=np.float32)
        for first in range(3):
            order = [0, 0, 0]
            order[first] = 2
            hessian[..., first, first] = gaussian_filter(scalar, sigma=smoothing, order=order, mode="wrap")
            for second in range(first + 1, 3):
                order = [0, 0, 0]
                order[first] = order[second] = 1
                derivative = gaussian_filter(scalar, sigma=smoothing, order=order, mode="wrap")
                hessian[..., first, second] = derivative
                hessian[..., second, first] = derivative
        eigenvalues = np.linalg.eigvalsh(hessian)
        if not np.all(np.isfinite(eigenvalues)):
            raise ValueError("the smoothed density Hessian contains non-finite eigenvalues")
        finite_magnitudes = np.abs(eigenvalues[np.isfinite(eigenvalues)])
        eigenvalue_threshold = float(np.percentile(finite_magnitudes, 30.0))
        environment_raw = np.count_nonzero(eigenvalues < -eigenvalue_threshold, axis=-1).astype(np.uint8)
        # A periodic majority filter suppresses isolated voxel flips and makes
        # the environment regions read as coherent structures at print size.
        class_votes = np.stack([
            uniform_filter((environment_raw == label).astype(np.float32), size=3, mode="wrap")
            for label in range(4)
        ])
        environment = np.argmax(class_votes, axis=0).astype(np.uint8)
        try:
            maximum_plot_particles = int(visualization.get("cosmic_web_max_particles", 30_000))
        except (TypeError, ValueError, OverflowError):
            maximum_plot_particles = 30_000
        maximum_plot_particles = min(100_000, max(256, maximum_plot_particles))
        particles, sample_ids = _sample_dark_matter_positions(
            files, box, maximum=maximum_plot_particles, seed=1847 + int(snap), return_ids=True,
        )
        namespace = provenance['identity_namespaces']['dark_matter']
        arrays = {'density_ratio': density, 'log_density': scalar, 'density_hessian_cell': hessian,
            'hessian_eigenvalues_cell': eigenvalues, 'environment_raw': environment_raw,
            'environment_votes': class_votes, 'environment_display': environment,
            'sample_positions_comoving_mpc_h': particles, 'sample_particle_ids': sample_ids}
        specifications = {name: {'units': 'dimensionless', 'quantity': quantity} for name, quantity in {
            'density_ratio': 'periodic CIC mass density divided by native mean density',
            'log_density': 'log1p of CIC density ratio before Gaussian derivatives',
            'density_hessian_cell': 'Gaussian derivative Hessian of log1p density ratio with respect to mesh indices; not gravitational tidal field',
            'hessian_eigenvalues_cell': 'ascending eigenvalues of the density morphology Hessian per mesh-index squared',
            'environment_raw': 'number of negative Hessian eigenvalues below the declared morphology threshold (0..3)',
            'environment_votes': 'periodic 3-cell filtered class indicator votes; display smoothing',
            'environment_display': 'majority class for morphology display (0..3)'}.items()}
        specifications['sample_positions_comoving_mpc_h'] = {'units': 'comoving Mpc/h', 'quantity': 'saved sampled DM positions in native fundamental box'}
        specifications['sample_particle_ids'] = {'units': 'identity', 'quantity': 'exact saved identities of corresponding scatter rows',
            'identity_namespace': namespace, 'ancestry': {'relationship': 'native snapshot rows; deterministic uniform random-key display sample'},
            'missing_policy': 'no sentinel; every sampled coordinate has an exact saved identity'}
        numeric_path = output_dir / 'cosmic_web_measurements.json'
        numeric = export_numeric_audit(numeric_path, {'schema': 'cosmic-web-morphology-measurements-v1',
            'epoch': epoch, 'snapshot': snap, 'box_mpc_h': box, 'grid_size': grid_size,
            'smoothing_sigma_cells': smoothing, 'eigenvalue_threshold': eigenvalue_threshold,
            'threshold_definition': '30th percentile of absolute Hessian eigenvalues; morphology proxy display threshold',
            'deposition': deposition, 'method': 'CIC density / Gaussian log-density derivatives; not potential/tidal Hessian',
            'sample_seed': 1847 + int(snap), 'sample_budget': maximum_plot_particles}, arrays, specifications,
            algorithm=__file__, context={'epoch': epoch, 'coordinates': coordinates, 'data_contract': provenance,
                'physical_constants': physical_constants_contract()}, source_hashes={item['path']: item['sha256'] for item in provenance['input_hashes']})
        result['numeric_measurements'] = {'path': numeric_path.name, 'sha256': digest(numeric_path),
            'numeric_data': numeric['numeric_data']}
        verify_product_provenance(source, provenance)
        from figures import render_cosmic_web_story
        result["figures"] = render_cosmic_web_story(
            density, environment, particles, box, redshift, snap, run_label, output_dir,
            visualization, deposition, smoothing, eigenvalue_threshold,
            scientific_source={"epoch": epoch, "identity_namespace": namespace,
                "data_contract": provenance, "numeric_measurements": result["numeric_measurements"]},
        )
        verify_product_provenance(source, provenance)
        result.update({
            "status": "COMPLETE", "snapshot": int(snap), "redshift": float(redshift),
            "grid_size": int(grid_size), "density_particle_count": int(deposition.get("particle_count", dm_count)),
            "environment_cell_counts": {
                _WEB_LABELS[index]: int(np.count_nonzero(environment == index)) for index in range(4)
            },
            "interpretation": (
                "The 3-D regions and 2-D ridge network are reproducible density-Hessian morphology proxies. "
                "They are not persistence-filtered DisPerSE manifolds or Morse critical structures."
            ),
        })
    except Exception as exc:
        result["reason"] = "%s: %s" % (type(exc).__name__, str(exc))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "paper_figure_manifest.json"
    atomic_json(manifest, result)
    return result


@directory_owner('output_dir')
def generate_single_run_story_figures(analysis_h5, output_dir, visualization=None, source_run=None):
    """Publish an immutable owned figure/measurement bundle; retain failed stages."""
    import tempfile
    from core.foundation import atomic_text, publish_directory
    destination = Path(output_dir).resolve()
    if destination.exists(): raise FileExistsError('Keep prior story figures; choose a fresh destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.story-figures-', dir=destination.parent))
    source = Path(source_run).resolve() if source_run is not None else destination.parent.parent
    try:
        result = _generate_single_run_story_figures(analysis_h5, stage, visualization, source)
        result.update(science_acceptance='NOT_ACCEPTED', validation_state='VALIDATION_PENDING',
            publication_contract='scientific-figure-bundle-v1', schema='scientific-figure-bundle-v1')
        result['artifacts'] = [{'path': path.relative_to(stage).as_posix(), 'sha256': digest(path)}
            for path in sorted(stage.rglob('*')) if path.is_file() and path.name != 'paper_figure_manifest.json'
            and not any(part.startswith('.') for part in path.relative_to(stage).parts)]
        manifest = stage / 'paper_figure_manifest.json'
        atomic_json(manifest, result)
        atomic_text(manifest.with_name(manifest.name + '.sha256'), digest(manifest) + '\n')
        from blender import verify_export_manifest
        verify_export_manifest(manifest, schema=result["schema"])
        publish_directory(stage, destination)
        return result
    except BaseException as exc:
        if stage.exists(): atomic_json(stage / 'failure.json', {'status': 'FAIL', 'science_acceptance': 'NOT_ACCEPTED',
            'failure': {'type': type(exc).__name__, 'message': str(exc)}})
        raise


@directory_owner('output_dir')
def generate_paired_story_figures(comparison_h5, output_dir, visualization=None):
    """Summarize the measured paired HMF and power-spectrum ratios when present."""
    from core.data import verify_hdf5_dataset_payloads
    before=digest(comparison_h5)
    verify_hdf5_dataset_payloads(comparison_h5)
    visualization = visualization if isinstance(visualization, dict) else {}
    apply_plot_style(visualization)
    output_dir = Path(output_dir)
    result = {"status": "UNAVAILABLE", "figures": [], "reason": "no valid paired ratio data",
        "science_acceptance":"NOT_ACCEPTED","comparison_sha256":before}
    try:
        curves_hmf, curves_pk = [], []
        with h5py.File(comparison_h5, "r") as handle:
            pairs = handle.get("pairs")
            if pairs is not None:
                for name in sorted(pairs.keys()):
                    group = pairs[name]
                    z = float(group.attrs.get("redshift_a", group.attrs.get("redshift", np.nan)))
                    if "hmf_mass_bin_centers_msun_h" in group and "hmf_ratio_b_over_a" in group:
                        x = np.asarray(group["hmf_mass_bin_centers_msun_h"], dtype=float)
                        y = np.asarray(group["hmf_ratio_b_over_a"], dtype=float)
                        keep = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
                        if np.count_nonzero(keep) >= 2:
                            curves_hmf.append((z, x[keep], y[keep]))
                    if "pk_k_h_mpc" in group and "pk_ratio_b_over_a" in group:
                        x = np.asarray(group["pk_k_h_mpc"], dtype=float)
                        y = np.asarray(group["pk_ratio_b_over_a"], dtype=float)
                        keep = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
                        if np.count_nonzero(keep) >= 2:
                            curves_pk.append((z, x[keep], y[keep]))
        if not curves_hmf and not curves_pk:
            if digest(comparison_h5)!=before:raise ValueError('Paired numeric source changed while reading')
            return result
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")
        for z, x, y in curves_hmf:
            axes[0].semilogx(x, y, marker="o", markersize=2.8,
                             label=("z = %.3g" % z) if np.isfinite(z) else "paired snapshot")
        axes[0].axhline(1.0, color="#334155", linewidth=0.9, linestyle="--")
        axes[0].set_xlabel(r"$M\ [M_\odot/h]$")
        axes[0].set_ylabel("Run B / Run A")
        axes[0].set_title("Halo mass function")
        axes[0].grid(True, which="both", alpha=0.25)
        if curves_hmf:
            axes[0].legend(fontsize=7)
        for z, x, y in curves_pk:
            axes[1].semilogx(x, y, linewidth=1.35,
                             label=("z = %.3g" % z) if np.isfinite(z) else "paired snapshot")
        axes[1].axhline(1.0, color="#334155", linewidth=0.9, linestyle="--")
        axes[1].set_xlabel(r"$k\ [h/\mathrm{Mpc}]$")
        axes[1].set_ylabel("Run B / Run A")
        axes[1].set_title("Matter power spectrum")
        axes[1].grid(True, which="both", alpha=0.25)
        if curves_pk:
            axes[1].legend(fontsize=7)
        fig.suptitle("Paired analysis ratios", weight="bold")
        stem = output_dir / "paired_ratio_overview"
        export_figure(fig, stem, metadata={
            "product": "paired measured HMF and matter-power ratios",
            "comparison_file": os.path.relpath(comparison_h5,output_dir),
            "comparison_sha256":before,
            "science_acceptance":"NOT_ACCEPTED",
            "hmf_curve_count": len(curves_hmf), "power_curve_count": len(curves_pk),
        })
        result = {
            "status": "COMPLETE", "figures": [{"status": "COMPLETE", "stem": stem.name,
                "files": [stem.with_suffix("." + fmt).name for fmt in _figure_formats(visualization)]}],
            "hmf_curve_count": len(curves_hmf), "power_curve_count": len(curves_pk),
        }
    except Exception as exc:
        result.update({"status": "PARTIAL", "reason": "%s: %s" % (type(exc).__name__, str(exc))})
    output_dir.mkdir(parents=True, exist_ok=True)
    if digest(comparison_h5)!=before:
        atomic_json(output_dir/'paper_figure_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'reason':'Paired numeric source changed while plotting'})
        raise ValueError('Paired numeric source changed while plotting')
    result.update(science_acceptance='NOT_ACCEPTED',comparison_sha256=before)
    atomic_json(output_dir / 'paper_figure_manifest.json',result)
    return result
from core.foundation import atomic_copy, publish_directory

import matplotlib.image as mpimg


@directory_owner('folder')
def review_figure_folder(folder, create_gallery=False):
    """Check PNG readability and dimensions; optionally assemble a review PDF.

    The review is deliberately non-destructive: an unreadable figure is listed
    with its reason and never prevents an otherwise valid analysis from being
    saved.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        (path for path in folder.rglob("*.png") if path.is_file() and not any(
            part.startswith('.') or part=='figure_reviews' for part in path.relative_to(folder).parts)),
        key=lambda path: path.as_posix().lower(),
    )
    records = []
    for path in image_paths:
        record = {"file": path.relative_to(folder).as_posix(), "status": "COMPLETE"}
        try:
            record['sha256']=digest(path)
            sidecar=path.with_suffix('.figure.json')
            record['contract_status']='LEGACY_UNSEALED'
            if sidecar.exists():
                from figures import verify_figure_export
                verify_figure_export(sidecar)
                record['figure_manifest_sha256']=digest(sidecar)
                record['contract_status']='SEALED_ARTIFACTS_VERIFIED'
            image = np.asarray(mpimg.imread(path))
            if digest(path)!=record['sha256']:raise ValueError('Figure changed during review')
            if image.ndim not in (2, 3) or image.shape[0] < 2 or image.shape[1] < 2:
                raise ValueError("image is not at least 2 by 2 pixels")
            record["dimensions_px"] = [int(image.shape[1]), int(image.shape[0])]
            if image.ndim == 2:
                finite = image[np.isfinite(image)]
            else:
                finite = image[..., :3][np.isfinite(image[..., :3])]
            if finite.size:
                record["finite_pixel_fraction"] = float(finite.size / max(1, image[..., :3].size if image.ndim == 3 else image.size))
                record["intensity_range"] = [float(np.min(finite)), float(np.max(finite))]
            else:
                raise ValueError("image contains no finite pixel values")
        except Exception as exc:
            record.update({"status": "UNREADABLE", "reason": "%s: %s" % (type(exc).__name__, str(exc))})
        records.append(record)

    stage=Path(tempfile.mkdtemp(prefix='.figure-review-',dir=folder))
    final=folder/'figure_reviews'/uuid.uuid4().hex
    gallery_name = None
    if create_gallery and image_paths:
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            gallery_path=stage/'figures.pdf'
            with PdfPages(gallery_path) as pdf:
                for row in records:
                    if row['status']!='COMPLETE':continue
                    original=folder/row['file'];copy=stage/'inputs'/row['file']
                    if atomic_copy(original,copy)!=row['sha256']:
                        raise ValueError('Figure changed before gallery construction')
                    image=mpimg.imread(copy)
                    fig,axis=plt.subplots(figsize=(11.0,8.5),layout='constrained')
                    try:
                        axis.imshow(image);axis.set_title(row['file'],fontsize=10,pad=8)
                        axis.axis('off');pdf.savefig(fig)
                    finally:plt.close(fig)
            gallery_name=(final.relative_to(folder)/gallery_path.name).as_posix()
        except Exception as exc:
            records.append({'file':'figures.pdf','status':'UNAVAILABLE',
                'reason':f'{type(exc).__name__}: {exc}'})
            atomic_json(stage/'failure.json',{'status':'FAILED','reason':str(exc)})

    unreadable = sum(row["status"] != "COMPLETE" for row in records)
    if not image_paths:
        status = "NO_FIGURES"
    elif unreadable:
        status = "PARTIAL"
    else:
        status = "COMPLETE"
    report = {
        "schema": "darkmattersim-figure-review-v1",
        "status": status,
        "figure_count": len(image_paths),
        "unreadable_count": sum(row["status"] == "UNREADABLE" for row in records),
        "gallery_pdf": gallery_name,
        "figures": records,
    }
    report['science_acceptance']='NOT_ACCEPTED'
    report['algorithm_sha256']=digest(__file__)
    report['review_bundle']=final.relative_to(folder).as_posix()
    report['artifacts']=[{'path':p.relative_to(stage).as_posix(),'sha256':digest(p)}
                         for p in sorted(stage.rglob('*')) if p.is_file()]
    atomic_json(stage/'figure_review.json',report)
    (stage/'figure_review.sha256').write_text(digest(stage/'figure_review.json')+'\n')
    final.parent.mkdir(parents=True,exist_ok=True)
    publish_directory(stage,final)
    # Compatibility view; the immutable generation retains every prior review.
    atomic_json(folder/'figure_review.json',report)
    return report


def verify_figure_review(directory):
    root=Path(directory).resolve();manifest=root/'figure_review.json'
    if digest(manifest)!=(root/'figure_review.sha256').read_text().strip():
        raise ValueError('Figure review checksum mismatch')
    record=json.loads(manifest.read_text())
    for artifact in record['artifacts']:
        path=(root/artifact['path']).resolve()
        if not path.is_relative_to(root) or digest(path)!=artifact['sha256']:
            raise ValueError('Figure review asset missing or changed')
    return record


# Native analysis products: preserve measured units and independent support masks.

_NATIVE_ANALYSIS_RECIPES = (('cosmology', 'evolved_power', 'pk/data/k_h_Mpc', 'pk/data/P_Mpc_h3', None), ('abundance', 'differential_hmf', 'hmf/data/mass_msun_h', 'hmf/data/dn_dlogM', 'hmf/data/counts'), ('abundance', 'cumulative_hmf', 'hmf/data/cumulative_mass_msun_h', 'hmf/data/cumulative_n_gt', None), ('structure', 'density_profile', 'profiles/data/density_profile_scatter/radius_over_r200c', 'profiles/data/density_profile_scatter/median', 'profiles/data/density_profile_scatter/valid_halo_count'), ('structure', 'velocity_dispersion', 'profiles/data/velocity_profile_scatter/radius_over_r200c', 'profiles/data/velocity_profile_scatter/median', 'profiles/data/velocity_profile_scatter/valid_halo_count'))

def _verify_native_analysis_dataset(dataset):
    """Check both the canonical storage contract and explicit producer metadata.

    The analysis-bundle seal covers attributes as well as numeric payloads.
    This bridge plots recorded units; it does not infer missing declarations.
    """
    from core.data import verify_dataset_semantics
    verify_dataset_semantics(dataset)
    if dataset.dtype.kind not in 'biuf':raise ValueError('Native figure datasets must be real numeric values')
    for key in ('units_status', 'quantity_definition_status', 'epoch_status', 'provenance_status'):
        if dataset.attrs.get(key) != 'PRODUCER_DECLARED':
            raise ValueError('Native dataset lacks producer-declared ' + key)
    for key in ('units', 'quantity_definition', 'missing_policy'):
        if not isinstance(dataset.attrs.get(key), str) or not dataset.attrs[key] or dataset.attrs[key] == 'UNDECLARED':
            raise ValueError('Native dataset has undeclared ' + key)
    for key in ('epoch_json', 'provenance_json'):
        if not isinstance(json.loads(dataset.attrs[key]), dict):
            raise ValueError('Native dataset lacks ' + key)

def _native_analysis_identity(dataset):
    """Read an exact scalar halo ID without passing through floating point."""
    _verify_native_analysis_dataset(dataset)
    if dataset.shape!=() or dataset.dtype!=np.dtype('uint64') or dataset.attrs['units']!='identity':
        raise ValueError('Native history bindings require scalar uint64 identities')
    return int(dataset[()])

def export_native_analysis_figures(bundle, destination, *, memory_gib=1.0, dpi=160):
    """Plot sealed native power, abundance, profiles and progenitor histories.

    Values retain their producer units and native selections. Bin support and
    uncertainty support are independent: missing jackknife errors never hide
    otherwise measured bins or become zero errors. No model is fitted here.
    """
    import h5py
    from core.analysis import verify_analysis_bundle
    from core.data import export_numeric_audit
    from core.foundation import file_sha256
    from core.research.display import export_scientific_suite
    bundle = Path(bundle).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Keep prior figure evidence; choose a fresh destination')
    if type(memory_gib) not in (int,float) or not np.isfinite(memory_gib) or memory_gib<=0 or type(dpi) is not int or not 150<=dpi<=1200:
        raise ValueError('Positive memory budget and DPI150..1200 required')
    verify_analysis_bundle(bundle, verify_sources=True)
    source = bundle / 'analysis_products.h5'
    before = file_sha256(source)
    implementation=Path(__file__).resolve();implementation_hash=file_sha256(implementation)
    source_hashes={str(source):before,str(implementation):implementation_hash}
    destination.mkdir(parents=True)
    sources = {}
    figures = []
    pending = []
    with h5py.File(source) as h:
        for epoch in sorted(h):
            if not epoch.startswith('snap_'):
                continue
            for family, variant, xpath, ypath, supportpath in _NATIVE_ANALYSIS_RECIPES:
                paths = [epoch + '/' + q for q in (xpath, ypath)] + ([epoch + '/' + supportpath] if supportpath else [])
                if any((q not in h for q in paths)):
                    pending.append({'epoch': epoch, 'family': family, 'variant': variant, 'status': 'PENDING_MEASURED_INPUT', 'missing_datasets': [q for q in paths if q not in h]})
                    continue
                dx, dy = (h[paths[0]], h[paths[1]])
                for d in (dx, dy):
                    _verify_native_analysis_dataset(d)
                if dx.attrs['epoch_json'] != dy.attrs['epoch_json']:
                    raise ValueError('Native figure coordinates and values have different actual epochs')
                if dx.ndim != 1 or dy.shape != dx.shape:
                    raise ValueError('Recipe requires matching native one-dimensional arrays')
                if dx.size * 128 > memory_gib * 1024 ** 3:
                    raise MemoryError('Native recipe arrays exceed workspace')
                x, y = (dx[:], dy[:])
                mask = np.isfinite(x) & np.isfinite(y)
                if family=='abundance':mask&=x>0  # Physical mass and logarithmic display support.
                if supportpath:
                    d = h[paths[2]]
                    _verify_native_analysis_dataset(d)
                    if d.attrs['epoch_json'] != dx.attrs['epoch_json']:
                        raise ValueError('Native support census has a different actual epoch')
                    if d.shape != x.shape:
                        raise ValueError('Native support census shape differs')
                    census=d[:]
                    mask &= np.isfinite(census)&(census>0)
                if not mask.any():
                    pending.append({'epoch': epoch, 'family': family, 'variant': variant, 'status': 'NO_SUPPORTED_MEASUREMENTS'})
                    continue
                name = epoch + '-' + variant
                selection = {
                    'native_dataset_paths': paths,
                    'support': 'finite native values; positive measured bin/group census where supplied',
                    'limitations': 'No halo completeness, Poisson adequacy, fitting support or physical convergence inferred',
                }
                units = {'x': str(dx.attrs['units']), 'y': str(dy.attrs['units']), 'supported': 'dimensionless'}
                arrays = {'x': x, 'y': y, 'supported': mask}
                series = [{'source': name, 'kind': 'line', 'label': 'Native measured bins', 'x': 'x', 'y': 'y', 'support': 'supported'}]
                uncertainty = {'status': 'NOT_SUPPLIED_BY_NATIVE_PRODUCT', 'scope': 'No independent uncertainty or covariance inferred'}
                error_path = epoch + '/hmf/data/spatial_jackknife/sigma_dn_dlog10M_per_mpc_h3'
                if variant == 'differential_hmf' and error_path in h:
                    error = h[error_path]
                    _verify_native_analysis_dataset(error)
                    if error.shape != y.shape or error.attrs['units'] != dy.attrs['units'] or error.attrs['epoch_json'] != dy.attrs['epoch_json']:
                        raise ValueError('Native HMF jackknife error grid, units or epoch differs')
                    arrays['error'] = error[:]
                    arrays['error_supported'] = mask & np.isfinite(arrays['error']) & (arrays['error'] >= 0)
                    units.update(error=units['y'], error_supported='dimensionless')
                    paths.append(error_path)
                    uncertainty = {
                        'status': 'MEASURED_WITHIN_BOX_SPATIAL_JACKKNIFE' if arrays['error_supported'].any() else 'NO_SUPPORTED_NATIVE_JACKKNIFE',
                        'scope': 'Original native delete-one spatial jackknife sigma; excludes larger-than-box modes and numerical convergence',
                    }
                    series.append({'source': name, 'kind': 'errorbar', 'label': 'Native spatial jackknife', 'x': 'x', 'y': 'y', 'error': 'error', 'support': 'error_supported'})
                definitions = {k: {'units': unit, 'quantity': k + ' for ' + family + '/' + variant, 'missing_policy': 'explicit support mask; original nonfinite values retained'} for k, unit in units.items()}
                product = destination / (name + '.json')
                epoch_contract = json.loads(dx.attrs['epoch_json'])
                export_numeric_audit(product, {
                    'schema': 'native-analysis-figure-bridge-v1',
                    'status': 'MEASURED',
                    'science_acceptance': 'NOT_ACCEPTED',
                    'selection': selection,
                    'epoch': epoch_contract,
                    'uncertainty': uncertainty,
                    'native_quantity_definitions': {'x': str(dx.attrs['quantity_definition']), 'y': str(dy.attrs['quantity_definition'])},
                }, arrays, definitions, algorithm=__file__, context={'epoch': epoch_contract}, source_hashes=source_hashes)
                sources[name] = str(product)
                figures.append({
                    'figure_id': name,
                    'family': family,
                    'variant': variant,
                    'title': variant.replace('_', ' ') + ' ' + epoch,
                    'science_status': 'NOT_ACCEPTED',
                    'epoch': epoch_contract,
                    'panels': [
                        {
                            'panel_id': 'native',
                            'title': 'Native saved measurements',
                            'quality_status': 'PENDING_RESOLUTION_AND_CONVERGENCE',
                            'xlabel': ('Comoving wavenumber' if variant == 'evolved_power' else 'Mass' if family == 'abundance' else 'Radius / R200c') + ' [' + units['x'] + ']',
                            'ylabel': variant.replace('_', ' ') + ' [' + units['y'] + ']',
                            'x_units': units['x'],
                            'y_units': units['y'],
                            'xscale': 'log' if family == 'abundance' else 'linear',
                            'selection': json.dumps(selection, sort_keys=True),
                            'series': series,
                        },
                    ],
                })
        history_path = 'main_progenitor_histories/data/histories'
        if history_path in h:
            for history_name, history in sorted(h[history_path].items()):
                if 'track' not in history:
                    continue
                track = history['track']
                # Include Python point/binding records, not just two numeric columns.
                # This is a workspace estimate, not a measured process RSS bound.
                if len(track)*2048>memory_gib*1024**3:raise MemoryError('Native history arrays exceed figure workspace')
                points = []
                bindings = []
                for point_name, point in sorted(track.items()):
                    if 'scale_factor' not in point or 'mass_msun_h' not in point:
                        continue
                    da, dm = (point['scale_factor'], point['mass_msun_h'])
                    for d in (da, dm):
                        _verify_native_analysis_dataset(d)
                    if da.attrs['units'] != 'dimensionless scale factor' or dm.attrs['units'] != 'Msun/h':
                        raise ValueError('History physical units differ from registered mass/a recipe')
                    if da.shape!=() or dm.shape!=():raise ValueError('Each native history point needs scalar scale factor and mass')
                    if json.loads(da.attrs['epoch_json'])!=json.loads(dm.attrs['epoch_json']):raise ValueError('History scale factor and mass refer to different epochs')
                    points.append((float(da[()]), float(dm[()])))
                    bindings.append({'a': da.name, 'mass': dm.name, 'halo_id':_native_analysis_identity(point['halo_id'])})
                if len(points) < 2:
                    pending.append({'history':history_name,'family':'assembly','variant':'mass_scale_factor','status':'INSUFFICIENT_HISTORY_SAMPLING'})
                    continue
                values = np.asarray(points)
                order = np.argsort(values[:, 0])
                values = values[order]
                bindings = [bindings[i] for i in order]
                time_supported=np.isfinite(values[:,0])&(values[:,0]>0)
                clock=values[time_supported,0]
                if np.any(np.diff(clock)<=0):raise ValueError('Native history has repeated physical epochs')
                name = 'history-' + history_name
                mask = time_supported & np.isfinite(values[:,1]) & (values[:,1] > 0)
                if mask.sum()<2:
                    pending.append({'history':history_name,'family':'assembly','variant':'mass_scale_factor','status':'NO_SUPPORTED_HISTORY'})
                    continue
                units = {'x': 'dimensionless scale factor', 'y': 'Msun/h', 'supported': 'dimensionless'}
                definitions = {k: {'units': unit, 'quantity': k + ' for native main-progenitor mass history', 'missing_policy': 'explicit mask'} for k, unit in units.items()}
                product = destination / (name + '.json')
                selection = {
                    'source_track': track.name,
                    'native_point_bindings': bindings,
                    'final_halo_id': _native_analysis_identity(history['final_halo_id']),
                    'scope': 'native linked main progenitor; complete tree support and resolution remain conditional',
                }
                epoch_contract = {'representation': 'scale-factor-indexed native history', 'scale_factor_range': [float(clock[0]),float(clock[-1])]}
                export_numeric_audit(product, {'schema': 'native-analysis-figure-bridge-v1', 'status': 'MEASURED', 'science_acceptance': 'NOT_ACCEPTED', 'selection': selection, 'epoch': epoch_contract}, {'x': values[:, 0], 'y': values[:, 1], 'supported': mask}, definitions, algorithm=__file__, context={'epoch': epoch_contract}, source_hashes=source_hashes)
                sources[name] = str(product)
                figures.append({
                    'figure_id': name,
                    'family': 'assembly',
                    'variant': 'mass_scale_factor',
                    'title': 'Main progenitor mass history ' + history_name,
                    'science_status': 'NOT_ACCEPTED',
                    'epoch': epoch_contract,
                    'panels': [
                        {
                            'panel_id': 'history',
                            'title': 'Actual native tree links',
                            'quality_status': 'PENDING_CONVERGENCE',
                            'xlabel': 'Scale factor a',
                            'ylabel': 'Mass [Msun/h]',
                            'x_units': units['x'],
                            'y_units': units['y'],
                            'selection': json.dumps(selection, sort_keys=True),
                            'series': [
                                {
                                    'source': name,
                                    'kind': 'line',
                                    'label': 'Native main progenitor',
                                    'x': 'x',
                                    'y': 'y',
                                    'support': 'supported',
                                },
                            ],
                        },
                    ],
                })
    if file_sha256(source)!=before or file_sha256(implementation)!=implementation_hash:
        raise ValueError('Native analysis or figure implementation changed during export')
    request = {'sources': sources, 'dpi': dpi, 'maximum_panels_per_figure': 1, 'figures': figures, 'organized_outputs': True}
    (destination / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    report = export_scientific_suite(request, destination / 'gallery', base=destination, memory_gib=memory_gib)
    (destination / 'native-connectivity.json').write_text(json.dumps({
        'schema': 'native-analysis-figure-connectivity-v1',
        'source_analysis_h5': str(source),
        'source_sha256': before,
        'rendered_figures': len(figures),
        'missing_measurements': pending,
        'registered_pending': report['pending_measurements'],
        'science_acceptance': 'NOT_ACCEPTED',
    }, indent=2) + '\n')
    return report


def experiment_figures(report, arrays, save):
    """Plot measured experiment products through the caller's save callback.

    Unsupported track bins stay missing. Posterior panels require actual
    sampler output and valid nonnegative weights; these plots do not establish
    calibration or promote a diagnostic intervention to native evidence.
    """
    from matplotlib import pyplot as plt
    schema = report['schema']
    if schema == 'stream-encounters-v1':
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        valid = arrays['geometry_supported']
        axes[0].scatter(arrays['encounter_time_myr'][valid], arrays['impact_parameter_kpc'][valid], s=10, label='within recorded geometry budget')
        axes[0].scatter(arrays['encounter_time_myr'][~valid], arrays['impact_parameter_kpc'][~valid], marker='x', label='refinement required')
        axes[0].set(xlabel='proper time [Myr]', ylabel='closest separation [kpc]')
        axes[0].legend()
        axes[1].scatter(np.linalg.norm(arrays['relative_velocity_kms'], axis=1), arrays['impact_parameter_kpc'], s=10)
        axes[1].set(xlabel='full relative speed [km/s]', ylabel='closest separation [kpc]')
        save(fig, 'encounter_geometry', ['Myr', 'proper kpc', 'km/s'])
    if schema in {'stream-counterfactuals-v1', 'compound-stream-encounter-v1'}:
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        for name in report['interventions']:
            density = arrays[name+'__linear_density_msun_kpc'][-1]
            supported = arrays[name+'__measurement_supported'][-1]
            s = arrays[name+'__bin_center_kpc']
            offset = arrays[name+'__centroid_kpc'][-1]
            velocity = arrays[name+'__velocity_mean_kms'][-1]
            axes[0].plot(s, np.where(supported, density, np.nan), label=name)
            axes[1].plot(s, np.where(supported, np.linalg.norm(offset, axis=1), np.nan), label=name)
            axes[2].plot(s, np.where(supported, velocity[:, 0], np.nan), label=name)
        axes[0].set_ylabel('linear mass density [Msun/kpc]')
        axes[0].legend()
        axes[1].set_ylabel('transverse centroid [kpc]')
        axes[2].set(xlabel='fixed-track arclength [kpc]', ylabel='along-track velocity [km/s]')
        fig.suptitle('Paired forward interventions; scientific acceptance unresolved')
        save(fig, 'forward_control_features', ['proper kpc', 'Msun/kpc', 'km/s'])
    if schema == 'stream-perturber-templates-v1':
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for row in report['templates']:
            key = row['template_id']+'__'
            obs = arrays[key+'observed_phase_space']
            axes[0].scatter(obs[:, 0], obs[:, 1], s=2, label=row['category'])
            axes[1].scatter(obs[:, 0], obs[:, 3], s=2, label=row['category'])
        axes[0].set(xlabel='longitude [rad]', ylabel='latitude [rad]')
        axes[1].set(xlabel='longitude [rad]', ylabel='radial velocity [km/s]')
        axes[0].legend()
        fig.suptitle('Matched physical templates; discrimination uncalibrated')
        save(fig, 'matched_template_observables', ['radian', 'km/s'])
    if schema in {'stream-posterior-v1', 'halo-shape-population-posterior-v1'}:
        if not report.get('chain_written'):
            raise ValueError('Posterior figures require an actual sampler product')
        samples = arrays['posterior_samples']
        weights = arrays['posterior_weight']
        if not np.isfinite(samples).all() or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('Posterior array weights/samples invalid')
        parameters = report['parameter_priors']
        ndim = len(parameters)
        # Panel count remains bounded for nuisance-rich models.
        for begin in range(0, ndim, 6):
            count = min(6, ndim-begin)
            fig, axes = plt.subplots(count, 1, figsize=(7, 2.2*count), squeeze=False)
            for j in range(count):
                row = parameters[begin+j]
                axes[j, 0].hist(samples[:, begin+j], weights=weights, bins=40, density=True, histtype='step')
                axes[j, 0].set(xlabel=row['name']+' ['+row['units']+']', ylabel='posterior density')
            fig.suptitle('Actual '+report['sampler']+' samples; calibration '+report['calibration_status'])
            save(fig, 'posterior_marginals_%02d'%begin, [p['units'] for p in parameters[begin:begin+count]])
        if 'chain' in arrays:
            chain = arrays['chain']
            fig, axes = plt.subplots(min(ndim, 6), 1, figsize=(8, 2*min(ndim, 6)), squeeze=False)
            for j in range(min(ndim, 6)):
                axes[j, 0].plot(chain[:, :, j], alpha=.2, linewidth=.4)
                axes[j, 0].set(xlabel='retained step', ylabel=parameters[j]['name'])
            save(fig, 'sampler_trace', [p['units'] for p in parameters[:6]])




def export_stream_figures(product_path,destination,*,memory_gib,formats=('png','pdf'),dpi=240):
    """Use the existing figure registry/sidecars with physical masks and units."""
    from core.stream_common import read_product, digest
    from core.foundation import atomic_json
    from matplotlib import pyplot as plt
    report,arrays,sources=read_product(product_path,memory_gib=memory_gib)
    output=Path(destination)
    output.mkdir(parents=True,exist_ok=True)
    written=[]
    metadata={'science_acceptance':report['science_acceptance'],'stream_product_sha256':digest(product_path),
              'namespace':report['namespace'],'epoch':report.get('epoch'),'schema':report['schema']}
    def save(figure,name,axes):
        path=output/name
        export_figure(figure,path,formats=formats,dpi=dpi,metadata={**metadata,'axes':axes})
        written.append(str(path))
    experiment_figures(report,arrays,save)
    if report['schema']=='tidal-stream-catalogue-v1':
        x=arrays['position_kpc']
        labels=arrays['candidate_id']
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for cid in np.unique(labels):
            rows=labels==cid
            for ax,(i,j) in zip(axes,[(0,1),(0,2)]):
                ax.scatter(x[rows,i],x[rows,j],s=2,label='background' if cid<0 else 'candidate '+str(cid))
        axes[0].set(xlabel='x [proper kpc]',ylabel='y [proper kpc]')
        axes[1].set(xlabel='x [proper kpc]',ylabel='z [proper kpc]')
        save(fig,'candidate_memberships',['proper kpc','proper kpc'])
    if report['schema']=='stream-coordinates-v1':
        for wrap in report['wraps']:
            if wrap['status']!='MEASURED':
                continue
            prefix='wrap_%04d_'%wrap['wrap_id']
            s=arrays[prefix+'bin_center_kpc']
            valid=arrays[prefix+'supported']
            fig,axes=plt.subplots(3,1,figsize=(8,9),sharex=True)
            axes[0].errorbar(s,np.where(valid,arrays[prefix+'linear_density_msun_kpc'],np.nan),yerr=np.where(valid,arrays[prefix+'density_sampling_error_msun_kpc'],np.nan),fmt='.',label='measured')
            axes[0].plot(s,arrays[prefix+'density_baseline_msun_kpc'],label='declared smoothing baseline')
            axes[0].set_ylabel('linear mass density [Msun/kpc]')
            widths=np.full((len(s),2),np.nan)
            widths[valid]=np.sqrt(np.maximum(np.linalg.eigvalsh(arrays[prefix+'width_covariance_kpc2'][valid]),0))
            axes[1].plot(s,widths)
            axes[1].set_ylabel('local transverse axes [kpc]')
            axes[2].plot(s,np.where(valid[:,None],arrays[prefix+'velocity_mean_kms'],np.nan))
            axes[2].set(xlabel='arclength [proper kpc]',ylabel='local mean velocity [km/s]')
            for ax in axes:
                for index in np.flatnonzero(~valid):
                    ax.axvspan(arrays[prefix+'bin_edges_kpc'][index],arrays[prefix+'bin_edges_kpc'][index+1],color='gray',alpha=.2)
            save(fig,prefix+'local_observables',['proper kpc','Msun/kpc','proper kpc','km/s'])
    if report['schema']=='stream-geometry-v1':
        for view in report['views']:
            prefix='view_%03d_'%view['view_id']
            xy=arrays[prefix+'projected_kpc']
            valid=arrays[prefix+'convex_probability_supported']
            convex=arrays[prefix+'convex_probability']>=report['controls']['minimum_convex_probability']
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            axes[0].plot(xy[:,0],xy[:,1],color='gray')
            axes[0].scatter(xy[valid&convex,0],xy[valid&convex,1],s=8,c='red',label='supported convex probability')
            axes[0].set(xlabel='projected x [kpc]',ylabel='projected y [kpc]',aspect='equal')
            arc=arrays[prefix+'projected_arclength_kpc']
            axes[1].plot(arc,arrays[prefix+'signed_curvature_kpc_inverse'])
            axes[1].set(xlabel='projected arclength [kpc]',ylabel='signed curvature [1/kpc]')
            save(fig,prefix+'geometry',['proper kpc','1 / proper kpc'])
    if report['schema']=='live-stream-resimulation-v1':
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        x=arrays['position_kpc'][-1]
        bound=arrays['bound'][-1]
        axes[0].scatter(x[~bound,0],x[~bound,1],s=2,label='unbound')
        axes[0].scatter(x[bound,0],x[bound,1],s=2,label='self-bound')
        axes[0].set(xlabel='inertial x [kpc]',ylabel='inertial y [kpc]')
        axes[1].plot(arrays['times_myr'],arrays['bound_mass_msun'])
        axes[1].set(xlabel='proper time [Myr]',ylabel='self-bound mass [Msun]')
        save(fig,'live_stripping',['proper kpc','Myr','Msun'])
    if 'actions_kpc_kms' in arrays and arrays['actions_kpc_kms'].ndim==2:
        valid=arrays['action_supported']
        a=arrays['actions_kpc_kms']
        fig,ax=plt.subplots(figsize=(6,4))
        ax.scatter(a[valid,2],a[valid,0],s=3)
        ax.set(xlabel='Jphi [kpc km/s]',ylabel='Jr [kpc km/s]')
        save(fig,'action_distribution',['kpc km/s'])
    atomic_json(output/'figure_registry.json',{'schema':'stream-figure-registry-v1','product':str(Path(product_path).resolve()),'source_hashes':sources,'figures':written,'science_acceptance':'NOT_ACCEPTED'})
    return written
