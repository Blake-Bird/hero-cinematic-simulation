"""Animate the measured projected peculiar velocity field at a frozen saved epoch."""
from pathlib import Path
import json

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates, gaussian_filter

from core.foundation import atomic_json, file_sha256


def sample_flow(positions, velocity, support):
    coordinates = (positions * support.shape[0] - .5).T
    weight = map_coordinates(support.astype(float), coordinates, order=1, mode="grid-wrap")
    values = np.column_stack([
        map_coordinates(np.nan_to_num(component), coordinates, order=1, mode="grid-wrap")
        for component in velocity
    ])
    return values, weight >= .999


def make_paired_flow_movies(atlas, snapshot, redshift, *, frames=240, fps=24):
    root = Path(atlas).resolve()
    token = f"{redshift:07.3f}".replace(".", "p")
    models = ("EDE", "LCDM")
    state = {}
    sources = {}
    for model, section, h in (("EDE", "02_EDE", .7483), ("LCDM", "03_LCDM", .6781)):
        source = root / section / "Spatial_explorers" / f"{model}_z_{token}" / "Peculiar_velocity_XY"
        contract = json.loads((source / "flow_contract.json").read_text())
        path = source / "flow_data.npz"
        if file_sha256(path) != contract["artifacts"][path.name]:
            raise ValueError(f"Changed velocity field {path}")
        with np.load(path, allow_pickle=False) as data:
            support = data["supported"].copy()
            velocity = (data["velocity_x_km_s"].copy(), data["velocity_y_km_s"].copy())
            centres = data["cell_centers_mpc_h"]
            box = float(centres[0] + centres[-1])
            sigma = data["mass_msun_h"].T * h / (box / len(centres)) ** 2
            counts = data["particle_count"].copy()
        rng = np.random.default_rng(71021)
        particles = rng.random((1200, 2))
        state[model] = dict(support=support, velocity=velocity, sigma=sigma, box=box/h,
                            positions=particles, trail=[], rng=rng, counts=counts)
        sources[model] = {"path": str(path.relative_to(root)), "sha256": file_sha256(path),
                          "contract": str((source / "flow_contract.json").relative_to(root))}
    reference = max(float(np.percentile(np.hypot(*s["velocity"])[s["support"]], 90)) for s in state.values())
    if reference <= 0:
        raise ValueError("A nonzero measured velocity field is required")
    font_root = Path(matplotlib.get_data_path()) / "fonts/ttf"
    title = ImageFont.truetype(str(font_root / "DejaVuSerif.ttf"), 31)
    label = ImageFont.truetype(str(font_root / "DejaVuSans.ttf"), 19)
    small = ImageFont.truetype(str(font_root / "DejaVuSans.ttf"), 16)
    values = np.concatenate([np.log10(np.maximum(s["sigma"], 1))[s["sigma"] > 0] for s in state.values()])
    low, high = np.percentile(values, [1, 99.7])
    for s in state.values():
        smoothed = gaussian_filter(s["sigma"], 1., mode="wrap")
        rgb = matplotlib.colormaps["magma"](np.clip((np.log10(np.maximum(smoothed, 1))-low)/(high-low), 0, 1), bytes=True)[..., :3]
        s["background"] = Image.fromarray(np.flipud(rgb)).resize((704, 704), Image.Resampling.LANCZOS).convert("RGBA")
    outputs = {}
    writers = {}
    for group in (*models, "Comparisons"):
        prefix = group if group != "Comparisons" else "EDE_vs_LCDM"
        folder = root / "04_Movies" / group / f"{prefix}_z_{token}_Animated_peculiar_velocity_flow"
        folder.mkdir(parents=True, exist_ok=True)
        movie = folder / (folder.name + ".mp4")
        if movie.exists():
            raise FileExistsError(f"Preserve the published film {movie}")
        outputs[group] = (folder, movie)
        writers[group] = imageio.get_writer(movie, fps=fps, codec="libx264", quality=9,
                                             pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"])
    try:
        for frame_index in range(frames):
            panels = {}
            for model, s in state.items():
                previous = s["positions"].copy()
                velocity, valid = sample_flow(previous, s["velocity"], s["support"])
                step = .0025 / reference
                midpoint = np.mod(previous + velocity * step/2, 1)
                middle_velocity, middle_valid = sample_flow(midpoint, s["velocity"], s["support"])
                positions = np.mod(previous + middle_velocity * step, 1)
                respawn = ~valid | ~middle_valid | ((np.arange(len(previous)) + frame_index) % 180 == 0)
                positions[respawn] = s["rng"].random((int(respawn.sum()), 2))
                continuous = ~respawn & (np.max(abs(positions-previous), axis=1) < .2)
                s["trail"].append((previous, positions.copy(), continuous))
                s["trail"] = s["trail"][-10:]
                s["positions"] = positions
                overlay = Image.new("RGBA", (704, 704))
                draw = ImageDraw.Draw(overlay)
                for age, (start, end, supported) in enumerate(s["trail"]):
                    opacity = int(22 + 190*(age+1)/len(s["trail"]))
                    for p, q in zip(start[supported], end[supported]):
                        draw.line((p[0]*704, (1-p[1])*704, q[0]*704, (1-q[1])*704),
                                  fill=(145, 248, 244, opacity), width=1)
                for p in positions[continuous]:
                    x, y = p[0]*704, (1-p[1])*704
                    draw.ellipse((x-1.1, y-1.1, x+1.1, y+1.1), fill=(222, 255, 249, 235))
                panels[model] = Image.alpha_composite(s["background"], overlay).convert("RGB")
            for group, writer in writers.items():
                pair = group == "Comparisons"
                frame = Image.new("RGB", (1600, 944) if pair else (1024, 832), "#0b1923")
                draw = ImageDraw.Draw(frame)
                draw.text((36, 22), f"Peculiar velocity flow · z = {redshift:g}", font=title, fill="#f4e7ce")
                draw.text((36, 67), f"Saved snapshot {snapshot:03d} · XY mass-weighted mean velocity · full comoving box",
                          font=label, fill="#b0cad6")
                selected = models if pair else (group,)
                for i, model in enumerate(selected):
                    x = 40 + i*792 if pair else 32
                    y = 132 if pair else 100
                    frame.paste(panels[model], (x, y))
                    if pair:
                        draw.text((x, 103), model if model == "EDE" else "ΛCDM", font=label, fill="#f4e7ce")
                if not pair:
                    draw.text((760, 135), group, font=title, fill="#9ae5df")
                    draw.text((760, 195), "Cyan trails", font=label, fill="#d1f6ed")
                    draw.text((760, 230), "show flow direction", font=small, fill="#b0cad6")
                    draw.text((760, 283), "Shared speed scale", font=small, fill="#b0cad6")
                    draw.text((760, 314), f"{reference:.1f} km/s", font=label, fill="#e8c48d")
                footer_y = 864 if pair else 807
                draw.text((36, footer_y), "Animated tracers follow a frozen projected field; the display clock is illustrative.",
                          font=small, fill="#b0cad6")
                if pair:
                    draw.text((36, 896), f"Shared density palette · shared animation speed · field reference {reference:.1f} km/s",
                              font=small, fill="#b0cad6")
                writer.append_data(np.asarray(frame))
                if frame_index == frames//2:
                    frame.save(outputs[group][0] / "Poster.png")
    finally:
        for writer in writers.values():
            writer.close()
    for group, (folder, movie) in outputs.items():
        selected = models if group == "Comparisons" else (group,)
        atomic_json(folder / "film_source.json", {
            "title": "Animated peculiar velocity flow", "model": group, "redshifts": f"Frozen saved epoch z = {redshift:g}",
            "snapshot": snapshot, "source_fields": {m:sources[m] for m in selected},
            "fps": fps, "frames": frames, "speed_reference_km_s": reference,
            "display_step_box_fraction_per_frame_at_reference_speed": .0025,
            "definition": "Mass-weighted full-box XY peculiar velocity. Bilinear sampling requires four supported cells. Periodic RK2 display tracers respawn at unsupported cells and fixed display ages.",
            "timing": "Illustrative animation clock. Tracers are not native particle identities or reconstructed particle trajectories.",
            "algorithm_sha256": file_sha256(__file__), "published_sha256": file_sha256(movie)})
    return {group:str(path) for group, (_,path) in outputs.items()}
