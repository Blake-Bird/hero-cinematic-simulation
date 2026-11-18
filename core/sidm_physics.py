"""Elastic SIDM angular models, native tables and explicit engine parameters.

Relative speeds are km/s. Differential cross sections are d(sigma/m)/dOmega
in cm^2/g/sr, with mu=cos(theta) on [-1,1] and azimuthal measure 2*pi.
Total, ordinary-transfer, identical-particle modified-transfer and viscosity
moments remain distinct. Folded identical-particle angles use |mu| on [0,1].
Nonlinear scattering settings do not replace the linear cosmological transfer
model. Table preparation computes model inputs; it does not evolve particles
or establish numerical convergence.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
import hashlib
import json
import math
import numpy as np
from .foundation import atomic_text, canonical_digest, file_sha256


# Differential models, angular moments and portable native tables

_CONVENTIONS = {"total", "ordinary_transfer", "modified_transfer_identical", "viscosity"}


_POLICIES = {"error", "constant", "zero"}


_FOUR_PI = 4.0 * math.pi


def _finite_nonnegative(value, label):
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"Finite nonnegative {label} required")
    return array


@dataclass(frozen=True)
class AngularMoments:
    """Angular cross-section moments and normalized small-angle diagnostics.

    Cross-section moments have cm^2/g units. small_probability, small_transfer
    and small_second_transfer are divided by the total cross section; those
    three quantities and neglected_moment_ratio are dimensionless.
    """
    total: float
    ordinary_transfer: float
    modified_transfer_identical: float
    viscosity: float
    second_transfer: float
    small_probability: float
    small_transfer: float
    small_second_transfer: float
    neglected_moment_ratio: float


class ElasticSIDMModel:
    """One elastic species with declared normalization and angular support.

    `specification` is a serializable sidm-elastic-model-v2 object. Functions
    may instead be supplied through `velocity_function`, provided their source
    identity is retained; native evolution receives a bounded generated table,
    never Python code evaluated inside the integrator.
    """

    def __init__(self, specification: Mapping, *, velocity_function: Callable | None = None):
        self.specification = json.loads(json.dumps(dict(specification), allow_nan=False))
        spec = self.specification
        if spec.get("schema") != "sidm-elastic-model-v2":
            raise ValueError("Versioned sidm-elastic-model-v2 physics specification required")
        self.kind = spec.get("kind")
        self.supplied_cross_section = spec.get("supplied_cross_section")
        if self.supplied_cross_section not in _CONVENTIONS:
            raise ValueError("Declare total, ordinary_transfer, modified_transfer_identical or viscosity")
        self.particle_convention = spec.get("particle_convention", "directed")
        if self.particle_convention not in {"directed", "folded_identical"}:
            raise ValueError("Declare directed or folded_identical particle convention")
        if spec.get("differential_units", "cm2_per_g_per_sr") != "cm2_per_g_per_sr":
            raise ValueError("Differential table units must be cm2_per_g_per_sr")
        self._callable = velocity_function
        self.model_hash = canonical_digest(spec)
        self.interpolation = dict(spec.get("interpolation", {}))
        self.absolute_tolerance = float(self.interpolation.get("absolute_tolerance_cm2_per_g", 1.e-10))
        self.relative_tolerance = float(self.interpolation.get("relative_tolerance", 1.e-4))
        if not math.isfinite(self.absolute_tolerance) or self.absolute_tolerance <= 0 or not 0 < self.relative_tolerance < 1:
            raise ValueError("Explicit finite positive absolute and relative interpolation tolerances required")
        self.low_velocity_policy = spec.get("low_velocity_policy", "error")
        self.high_velocity_policy = spec.get("high_velocity_policy", "error")
        if self.low_velocity_policy not in _POLICIES or self.high_velocity_policy not in _POLICIES:
            raise ValueError("Explicit error, constant or zero low/high velocity policy required")
        if self.kind == "differential_table":
            if "low_velocity_policy" not in spec or "high_velocity_policy" not in spec or not self.interpolation:
                raise ValueError("Arbitrary tables require explicit velocity policies and interpolation error controls")
            self.velocities = _finite_nonnegative(spec["speed_kms"], "velocity grid")
            self.cosines = np.asarray(spec["cos_theta"], dtype=np.float64)
            self.differential = _finite_nonnegative(spec["differential_cm2_per_g_per_sr"], "differential table")
            if self.velocities.ndim != 1 or len(self.velocities) < 2 or np.any(self.velocities <= 0) or np.any(np.diff(self.velocities) <= 0):
                raise ValueError("At least two strictly increasing positive tabulated speeds required")
            if self.cosines.ndim != 1 or len(self.cosines) < 3 or not np.isfinite(self.cosines).all() or np.any(np.diff(self.cosines) <= 0):
                raise ValueError("Strictly increasing finite cosine grid required")
            if self.cosines[0] != -1.0 or self.cosines[-1] != 1.0:
                raise ValueError("Differential tables must cover the complete finite angular domain [-1,1]")
            if self.differential.shape != (len(self.velocities), len(self.cosines)):
                raise ValueError("Differential table shape must be [speed,cos_theta]")
            if self.differential.size > 4_000_000:
                raise MemoryError("Differential model exceeds the declared four-million-value native bound")
        elif self.kind == "constant_isotropic":
            _positive(spec["cross_section_cm2_per_g"], "isotropic cross section", zero=True)
        elif self.kind == "velocity_dependent_isotropic":
            function = spec.get("function", {})
            if function.get("kind") not in {"power_law", "broken_power_law", "screened", "callable"}:
                raise ValueError("Explicit power_law, broken_power_law, screened or retained callable function required")
            if function["kind"] == "callable" and (velocity_function is None or not function.get("source_sha256")):
                raise ValueError("Callable velocity law needs its retained source identity and callable")
            if function["kind"] != "callable":
                _positive(function["velocity_scale_kms"], "isotropic velocity scale")
                _positive(function["normalization_cm2_per_g"], "velocity-function normalization", zero=True)
        elif self.kind == "screened_rutherford":
            _positive(spec["cross_section_cm2_per_g"], "screened normalization", zero=True)
            required_definition = "zero_velocity_total" if self.supplied_cross_section == "total" else "moment_at_each_speed"
            if spec.get("normalization_definition") != required_definition:
                raise ValueError(f"Rutherford {self.supplied_cross_section} requires explicit normalization_definition={required_definition}")
            _positive(spec["velocity_scale_kms"], "Rutherford screening velocity")
        else:
            raise ValueError("Unsupported elastic SIDM differential model")

    @classmethod
    def constant_isotropic(cls, cross_section_cm2_per_g, *, convention="total", particle_convention="directed"):
        return cls({"schema": "sidm-elastic-model-v2", "kind": "constant_isotropic",
                    "cross_section_cm2_per_g": cross_section_cm2_per_g,
                    "supplied_cross_section": convention, "particle_convention": particle_convention})

    def _supplied_isotropic(self, speed):
        if self.kind == "constant_isotropic":
            return np.full_like(speed, self.specification["cross_section_cm2_per_g"])
        f = self.specification["function"]
        if f["kind"] == "callable":
            result = np.asarray(self._callable(speed), dtype=np.float64)
        else:
            x = speed / float(f["velocity_scale_kms"])
            amplitude = float(f["normalization_cm2_per_g"])
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                if f["kind"] == "power_law":
                    result = amplitude * np.maximum(x, float(f.get("speed_floor_fraction", 0.0))) ** float(f["power"])
                elif f["kind"] == "broken_power_law":
                    smoothness = float(f.get("smoothness", 2.0))
                    if not math.isfinite(smoothness) or smoothness <= 0:
                        raise ValueError("Finite positive broken-law smoothness required")
                    result = amplitude * x ** float(f["low_power"]) * (1 + x ** smoothness) ** ((float(f["high_power"]) - float(f["low_power"])) / smoothness)
                else:
                    result = amplitude / (1 + x * x) ** float(f.get("power", 2.0))
        return _finite_nonnegative(result, "evaluated velocity-dependent cross section")

    def _isotropic_total(self, speed):
        """Convert the declared isotropic moment to total cm^2/g.

        For an isotropic distribution, sigma_T=sigma_tot,
        sigma_modified=sigma_tot/2 and sigma_V=2*sigma_tot/3.
        """
        conversion = {"total": 1., "ordinary_transfer": 1., "modified_transfer_identical": 2., "viscosity": 1.5}
        return self._supplied_isotropic(speed) * conversion[self.supplied_cross_section]

    def _table_row(self, speed):
        value = float(speed)
        if value < self.velocities[0] or value > self.velocities[-1]:
            low = value < self.velocities[0]
            policy = self.low_velocity_policy if low else self.high_velocity_policy
            if policy == "error":
                raise ValueError("SIDM relative speed lies outside retained table coverage")
            if policy == "zero":
                return np.zeros_like(self.cosines)
            return self.differential[0 if low else -1].copy()
        upper = min(int(np.searchsorted(self.velocities, value, side="right")), len(self.velocities) - 1)
        lower = max(0, upper - 1)
        if lower == upper:
            return self.differential[lower].copy()
        fraction = math.log(value / self.velocities[lower]) / math.log(self.velocities[upper] / self.velocities[lower])
        return (1 - fraction) * self.differential[lower] + fraction * self.differential[upper]

    def differential_cross_section_per_mass(self, speed_kms, cos_theta):
        speed, mu = np.broadcast_arrays(_finite_nonnegative(speed_kms, "relative speed"), np.asarray(cos_theta, dtype=float))
        if not np.isfinite(mu).all() or np.any(np.abs(mu) > 1):
            raise ValueError("Scattering cosine must lie in [-1,1]")
        if self.kind in {"constant_isotropic", "velocity_dependent_isotropic"}:
            result = self._isotropic_total(speed) / _FOUR_PI
        elif self.kind == "screened_rutherford":
            a = 0.5 * (speed / float(self.specification["velocity_scale_kms"])) ** 2
            shape = 1 / (1 + a * (1 - mu)) ** 2
            amplitude = float(self.specification["cross_section_cm2_per_g"])
            result = amplitude * shape / _FOUR_PI
            if self.supplied_cross_section != "total":
                # The input fixes the stated moment at each speed; normalize
                # using that moment of the actual angular distribution.
                result = result * np.asarray([self._rutherford_normalizer(float(v)) for v in speed.flat]).reshape(speed.shape)
        else:
            result = np.asarray([np.interp(float(u), self.cosines, self._table_row(float(v)))
                                 for v, u in zip(speed.flat, mu.flat)]).reshape(speed.shape)
        return float(result) if result.ndim == 0 else result

    def _rutherford_normalizer(self, speed):
        a = 0.5 * (speed / float(self.specification["velocity_scale_kms"])) ** 2
        def integral(power, limit):
            x = a * limit
            if x < .25:
                total = 0.
                for order in range(128):
                    term = (-x) ** order * (order + 1) / (order + power + 1)
                    total += term
                    if abs(term) <= np.finfo(float).eps * max(1., abs(total)):
                        return .5 * limit ** (power + 1) * total
                raise ValueError("Rutherford small-anisotropy moment failed finite series allocation")
            if power == 0:
                return limit / (2 * (1 + x))
            if power == 1:
                return (math.log1p(x) - x / (1 + x)) / (2 * a * a)
            return (x + x / (1 + x) - 2 * math.log1p(x)) / (2 * a ** 3)
        first = integral(1, 2.)
        moments = {"ordinary_transfer": first,
                   "modified_transfer_identical": 2 * integral(1, 1.) - first + 2 * (integral(0, 2.) - integral(0, 1.)),
                   "viscosity": 2 * first - integral(2, 2.)}
        moment = moments[self.supplied_cross_section]
        if moment <= 0 or not math.isfinite(moment):
            raise ValueError("Rutherford normalization moment lost finite support")
        return 1 / moment

    def _integration_row(self, speed, split=None):
        # Table moments integrate the exact retained piecewise-linear density.
        # Gauss order 4 is exact for each segment times all moment polynomials.
        if self.kind == "differential_table":
            edges = np.unique(np.r_[self.cosines, 0.])
        elif self.kind in {"constant_isotropic", "velocity_dependent_isotropic"}:
            edges = np.array([-1., 0., 1.])
        else:
            a = 0.5 * (float(speed) / float(self.specification["velocity_scale_kms"])) ** 2
            # Resolve forward support without concentrating only on theta.
            x = np.geomspace(max(np.finfo(float).eps, 1 / (1 + a) * 1.e-6), 2., 1024)
            edges = np.unique(np.r_[-1., 0., 1. - x, 1.])
        if split is not None:
            edges = np.unique(np.r_[edges, math.cos(split), -math.cos(split)])
        nodes, weights = np.polynomial.legendre.leggauss(4)
        left, right = edges[:-1], edges[1:]
        mu = (left[:, None] + right[:, None]) / 2 + (right - left)[:, None] / 2 * nodes
        measure = math.pi * (right - left)[:, None] * weights * self.differential_cross_section_per_mass(speed, mu)
        return mu, measure

    def angular_moments(self, speed_kms, split_angle_radians=0.):
        """Integrate 2*pi*dmu*d(sigma/m)/dOmega with each stated weight.

        Ordinary transfer uses 1-mu, identical transfer uses 1-|mu|, and
        viscosity uses 1-mu^2. The chosen particle convention sets the
        small-angle transport coordinate x to 1-mu or 1-|mu|.
        """
        speed = float(_finite_nonnegative(speed_kms, "relative speed"))
        maximum = math.pi / 2 if self.particle_convention == "folded_identical" else math.pi
        split = float(split_angle_radians)
        if not math.isfinite(split) or not 0 <= split <= maximum:
            raise ValueError("SIDM split angle is outside the declared particle convention")
        mu, measure = self._integration_row(speed, split)
        x = 1 - (np.abs(mu) if self.particle_convention == "folded_identical" else mu)
        small = x < (1 - math.cos(split))
        total = float(np.sum(measure))
        first = float(np.sum(measure * x))
        small_first = float(np.sum(measure * x * small))
        small_second = float(np.sum(measure * x * x * small))
        return AngularMoments(total, float(np.sum(measure * (1 - mu))), float(np.sum(measure * (1 - np.abs(mu)))),
                              float(np.sum(measure * (1 - mu * mu))), float(np.sum(measure * x * x)),
                              float(np.sum(measure * small)) / total if total else 0.,
                              small_first / total if total else 0., small_second / total if total else 0.,
                              small_second / small_first if small_first else 0.)

    def _moment_vector(self, speed_kms, member):
        speeds = _finite_nonnegative(speed_kms, "relative speed")
        result = np.asarray([getattr(self.angular_moments(float(v)), member) for v in speeds.flat]).reshape(speeds.shape)
        return float(result) if result.ndim == 0 else result

    def total_cross_section_per_mass(self, speed_kms):
        if self.kind in {"constant_isotropic", "velocity_dependent_isotropic"}:
            result = self._isotropic_total(_finite_nonnegative(speed_kms, "relative speed"))
            return float(result) if result.ndim == 0 else result
        return self._moment_vector(speed_kms, "total")

    def momentum_transfer_cross_section_per_mass(self, speed_kms):
        """Ordinary transfer integral, integral (1-mu) d(sigma/m)."""
        return self._moment_vector(speed_kms, "ordinary_transfer")

    def modified_transfer_cross_section_per_mass(self, speed_kms):
        """Identical-particle transfer integral, integral (1-|mu|) d(sigma/m)."""
        return self._moment_vector(speed_kms, "modified_transfer_identical")

    def viscosity_cross_section_per_mass(self, speed_kms):
        return self._moment_vector(speed_kms, "viscosity")

    def sample_angle(self, speed_kms, uniform_draw, *, minimum_angle_radians=0.):
        """Return cosine from the normalized retained conditional distribution.

        Zero cross section has no probability distribution and is rejected.
        Native full tables use the same stable piecewise-quadratic CDF inversion.
        """
        speed = float(_finite_nonnegative(speed_kms, "relative speed"))
        u = float(uniform_draw)
        if not math.isfinite(u) or not 0 <= u <= 1:
            raise ValueError("Uniform angle draw must lie in [0,1]")
        folded = self.particle_convention == "folded_identical"
        minimum = float(minimum_angle_radians)
        maximum = math.pi / 2 if folded else math.pi
        if not math.isfinite(minimum) or not 0 <= minimum < maximum:
            raise ValueError("Minimum angle must lie in [0, pi/2) folded or [0, pi) directed")
        upper = math.cos(minimum)
        # A near-endpoint angle may round to the endpoint in cosine space.
        # Reject a zero-width domain rather than inventing a deterministic draw.
        if (folded and upper <= 0) or (not folded and upper <= -1):
            raise ValueError("Conditional angular domain has no representable width")
        if self.kind in {"constant_isotropic", "velocity_dependent_isotropic"}:
            if self.total_cross_section_per_mass(speed) <= 0:
                raise ValueError("Zero SIDM kernel has no sampling distribution")
            return (u * upper if folded else -1 + u * (upper + 1))
        if self.kind == "screened_rutherford" and not folded:
            if self.specification["cross_section_cm2_per_g"] <= 0:
                raise ValueError("Zero SIDM kernel has no sampling distribution")
            a = 0.5 * (speed / float(self.specification["velocity_scale_kms"])) ** 2
            large_mass = (1 + upper) / (2 * (1 + a * (1 - upper)))
            quantile = u * large_mass
            return 1 - 2 * (1 - quantile) / (1 + 2 * a * quantile)
        edges = self.cosines if self.kind == "differential_table" else self._integration_edges(speed)
        if folded:
            edges = np.unique(np.r_[0., np.abs(edges), upper])
            edges = edges[(edges >= 0) & (edges <= upper)]
            density = np.asarray(self.differential_cross_section_per_mass(speed, edges)) + self.differential_cross_section_per_mass(speed, -edges)
        else:
            edges = np.unique(np.r_[edges, upper])
            edges = edges[(edges >= -1) & (edges <= upper)]
            density = np.asarray(self.differential_cross_section_per_mass(speed, edges))
        width = np.diff(edges)
        mass = 0.5 * (density[:-1] + density[1:]) * width
        cumulative = np.r_[0., np.cumsum(mass)]
        if cumulative[-1] <= 0:
            raise ValueError("Conditional SIDM distribution has no support")
        target = u * cumulative[-1]
        segment = min(int(np.searchsorted(cumulative, target, side="right")) - 1, len(width) - 1)
        segment = max(0, segment)
        remaining = target - cumulative[segment]
        slope = (density[segment + 1] - density[segment]) / width[segment]
        if remaining <= 0:
            delta = 0.
        else:
            radical = math.sqrt(max(0., density[segment] ** 2 + 2 * slope * remaining))
            denominator = density[segment] + radical
            delta = 2 * remaining / denominator if denominator > 0 else width[segment]
        return float(np.clip(edges[segment] + delta, edges[segment], edges[segment + 1]))

    def _integration_edges(self, speed):
        scale = float(self.specification["velocity_scale_kms"])
        a = 0.5 * (speed / scale) ** 2
        return np.unique(np.r_[-1., 0., 1. - np.geomspace(max(np.finfo(float).eps, 1.e-8 / (1 + a)), 2., 4096), 1.])

    def to_native_table(self, path, *, speed_kms=None, cos_theta=None):
        """Write immutable native table and metadata; no evolution is executed.

        Supplied arbitrary tables define the retained interpolant exactly.
        Analytic laws are refined by midpoint error until the declared bound is
        met, with finite allocation limits and explicit low/high speed policies.
        """
        destination = Path(path).expanduser().resolve()
        if self.kind == "differential_table" and speed_kms is None and cos_theta is None:
            velocities, cosines, differential = self.velocities, self.cosines, self.differential
        else:
            if speed_kms is None:
                coverage = self.specification.get("native_speed_coverage_kms")
                if not coverage or len(coverage) != 2 or not 0 < coverage[0] < coverage[1]:
                    raise ValueError("Generated native model requires explicit positive speed coverage")
                speed_kms = np.geomspace(coverage[0], coverage[1], 65)
            velocities = _finite_nonnegative(speed_kms, "native speed coverage")
            cosines = np.asarray(np.linspace(-1., 1., 65) if cos_theta is None else cos_theta, float)
            if velocities.ndim != 1 or len(velocities) < 2 or np.any(velocities <= 0) or np.any(np.diff(velocities) <= 0):
                raise ValueError("Increasing positive native speed grid required")
            if cosines.ndim != 1 or len(cosines) < 3 or not np.isfinite(cosines).all() or cosines[0] != -1 or cosines[-1] != 1 or np.any(np.diff(cosines) <= 0):
                raise ValueError("Complete increasing native angular grid required")
            for _ in range(16):
                differential = self.differential_cross_section_per_mass(velocities[:, None], cosines[None, :])
                mid_v = np.sqrt(velocities[:-1] * velocities[1:])
                mid_mu = 0.5 * (cosines[:-1] + cosines[1:])
                exact_v = self.differential_cross_section_per_mass(mid_v[:, None], cosines[None, :])
                exact_mu = self.differential_cross_section_per_mass(velocities[:, None], mid_mu[None, :])
                error_v = np.abs(exact_v - 0.5 * (differential[:-1] + differential[1:]))
                error_mu = np.abs(exact_mu - 0.5 * (differential[:, :-1] + differential[:, 1:]))
                add_v = np.any(error_v > self.absolute_tolerance / _FOUR_PI + self.relative_tolerance * exact_v, axis=1)
                add_mu = np.any(error_mu > self.absolute_tolerance / _FOUR_PI + self.relative_tolerance * exact_mu, axis=0)
                if not add_v.any() and not add_mu.any():
                    break
                velocities = np.sort(np.r_[velocities, mid_v[add_v]])
                cosines = np.sort(np.r_[cosines, mid_mu[add_mu]])
                if len(velocities) * len(cosines) > 4_000_000:
                    raise MemoryError("Native model refinement exceeds the four-million-value bound")
            else:
                raise ValueError("Native differential interpolation failed its declared midpoint error controls")
        policy_codes = {"error": 0, "constant": 1, "zero": 2}
        header = ["SIDM_DIFFERENTIAL_TABLE_V2", self.model_hash,
                  f"{len(velocities)} {len(cosines)} {policy_codes[self.low_velocity_policy]} {policy_codes[self.high_velocity_policy]}",
                  f"{self.absolute_tolerance:.17g} {self.relative_tolerance:.17g}",
                  " ".join(f"{float(v):.17g}" for v in velocities), " ".join(f"{float(mu):.17g}" for mu in cosines)]
        body = "\n".join(" ".join(f"{float(v):.17g}" for v in row) for row in differential)
        content = ("\n".join(header) + "\n" + body + "\n").encode("ascii")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != content:
            raise FileExistsError("Existing native differential table differs; use a new retained model path")
        if not destination.exists():
            try:
                atomic_text(destination, content.decode("ascii"), overwrite=False)
            except FileExistsError:
                # Another writer may have published this immutable model.
                if destination.read_bytes() != content:
                    raise FileExistsError("Concurrent native differential table differs; use a new retained model path")
        table_hash = hashlib.sha256(content).hexdigest()
        return {"schema": "sidm-native-differential-table-v2", "path": str(destination), "sha256": table_hash,
                "model_hash": self.model_hash, "shape": [len(velocities), len(cosines)],
                "speed_coverage_kms": [float(velocities[0]), float(velocities[-1])],
                "interpolation": "linear_cosine_and_log_speed", "supplied_cross_section": self.supplied_cross_section,
                "low_velocity_policy": self.low_velocity_policy, "high_velocity_policy": self.high_velocity_policy,
                "error_control": "declared_piecewise_interpolant" if self.kind == "differential_table" else "adaptive_midpoint_absolute_and_relative"}


def read_native_differential_model(path, *, model_sha256, table_sha256, particle_convention, memory_bytes):
    """Read the exact native interpolant without generating a replacement table."""
    from .foundation import file_sha256
    path=Path(path).expanduser().resolve()
    if type(memory_bytes)is not int or memory_bytes<=0 or path.stat().st_size*16>memory_bytes:
        raise MemoryError('Retained native differential model exceeds its allocated parse workspace')
    if file_sha256(path)!=table_sha256:raise ValueError('Retained native SIDM differential table changed')
    lines=path.read_text(encoding='ascii').splitlines()
    if len(lines)<8 or lines[0]!='SIDM_DIFFERENTIAL_TABLE_V2' or lines[1]!=model_sha256:
        raise ValueError('Native differential table/model identity does not match actual evolution')
    nv,nmu,low,high=map(int,lines[2].split());absolute,relative=map(float,lines[3].split())
    if nv<2 or nmu<3 or nv*nmu>4_000_000 or low not in (0,1,2) or high not in (0,1,2) or len(lines)!=6+nv:
        raise ValueError('Native differential table dimensions/policies are invalid')
    # Converting every token rejects trailing junk that fromstring can ignore.
    speed=np.asarray(lines[4].split(),float);cosine=np.asarray(lines[5].split(),float)
    differential=np.asarray([np.asarray(row.split(),float) for row in lines[6:]])
    if speed.shape!=(nv,) or cosine.shape!=(nmu,) or differential.shape!=(nv,nmu):
        raise ValueError('Native differential table has truncated or extra numeric columns')
    policies={0:'error',1:'constant',2:'zero'}
    result=ElasticSIDMModel({'schema':'sidm-elastic-model-v2','kind':'differential_table',
        'particle_convention':particle_convention,'supplied_cross_section':'total',
        'speed_kms':speed.tolist(),'cos_theta':cosine.tolist(),'differential_cm2_per_g_per_sr':differential.tolist(),
        'low_velocity_policy':policies[low],'high_velocity_policy':policies[high],
        'interpolation':{'absolute_tolerance_cm2_per_g':absolute,'relative_tolerance':relative}})
    if file_sha256(path)!=table_sha256:raise ValueError('Native differential table changed while reading')
    return result


def sidm_model_main():
    """Materialize a declared differential model, without native execution."""
    import argparse
    from .foundation import atomic_json, file_sha256
    parser = argparse.ArgumentParser(description='Write a pinned native differential SIDM table from an explicit physical model')
    parser.add_argument('model', type=Path)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--grid', type=Path, help='JSON containing explicit speed_kms and cos_theta arrays for analytic models')
    parser.add_argument('--output', type=Path, required=True, help='Immutable model/table/source descriptor')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep previous SIDM model descriptor')
    model_path = args.model.expanduser().resolve()
    original = file_sha256(model_path)
    spec = json.loads(model_path.read_text())
    model = ElasticSIDMModel(spec)
    grid_path = args.grid.expanduser().resolve() if args.grid else None
    grid_hash = file_sha256(grid_path) if grid_path is not None else None
    grid = json.loads(grid_path.read_text()) if grid_path is not None else {}
    if grid_path is not None and (not isinstance(grid, dict) or set(grid) != {'speed_kms', 'cos_theta'}):
        raise ValueError('Declare exactly the native speed and full cosine grids')
    table = model.to_native_table(args.table, **grid)
    if file_sha256(model_path) != original:
        raise ValueError('Physical SIDM model changed during table materialization')
    sources = {str(model_path): original, str(Path(__file__).resolve()): file_sha256(__file__)}
    if grid_path is not None:
        if file_sha256(grid_path) != grid_hash:
            raise ValueError('Native SIDM grid changed during table materialization')
        sources[str(grid_path)] = grid_hash
    atomic_json(args.output, {'schema': 'sidm-model-materialization-v1', 'model': spec, 'table': table,
        'source_hashes': sources, 'execution_status': 'COMPLETE', 'science_acceptance': 'NOT_ESTABLISHED'})
    return 0


# Folded Born Moller reference, including identical final-state normalization

def moller_primitive(cosine,anisotropy):
    """Folded Born Moller angular mass on [0, |cos(theta)|], in units of sigma0.

    anisotropy = (v_relative/v_screen)^2/2 is dimensionless. At zero speed,
    this mass is u/2, so the total is sigma0/2, retaining the identical
    final-state symmetry normalization.
    """
    u=np.asarray(cosine,dtype=float)
    a=float(anisotropy)
    if not np.isfinite(a) or a<0 or not np.isfinite(a*a) or not np.isfinite(u).all() or np.any((u<0)|(u>1)):
        raise ValueError('Finite folded Born Moller domain required')
    if a==0:return u/2
    plus=1+a*(1-u);minus=1+a*(1+u)
    return u/(plus*minus)-np.log1p(2*a*u/plus)/(4*a*(1+a))


def moller_curvature_upper_bound(upper_cosine,anisotropy,conditional_mass):
    """Triangle bound on derivative of the positive folded density.

    On [0,u], D_plus>=1+a(1-u), D_minus>=1+a. Divide by the
    physical truncated cross-section mass, rather than renormalizing each term.
    """
    u=np.asarray(upper_cosine,float);a=float(anisotropy)
    if not np.isfinite(u).all() or np.any((u<0)|(u>1)) or not math.isfinite(a) or a<0 or not math.isfinite(a*a):
        raise ValueError('Finite folded Born Moller curvature domain required')
    conditional_mass=_positive(conditional_mass,'Moller truncated angular mass')
    inverse_plus=1/(1+a*(1-u));inverse_center=1/(1+a)
    # Multiply bounded factors before dividing by the truncated mass. This
    # preserves the same triangle bound without cubing a large denominator.
    forward=(a*inverse_plus)*inverse_plus*inverse_plus
    backward=(a*inverse_center)*inverse_center*inverse_center
    interference=(a*inverse_plus*inverse_center)**2*u
    bound=(forward+backward+interference)/conditional_mass
    if not np.isfinite(bound).all():raise ValueError('Born Moller curvature bound exceeds numeric range')
    return bound


# Explicit public settings mapped to native engine parameters

# Support lengths are proper kpc, sigma/m is cm^2/g, and clock tolerances
# are seconds. Absolute conservation floors match the native pair estimator:
# energy in Msun*(km/s)^2 and momentum in Msun*km/s. The canonical-control
# branch rescales these floors with its velocity coordinate, not with h.
FLOAT_PARAMETERS={
    'kernel_support_proper_kpc':'SIDMReferenceKernelRadiusProperKpc',
    'cross_section_normalization_cm2_per_g':'SIDMReferenceCrossSectionNormalizationCm2PerG',
    'maximum_pair_probability':'SIDMReferenceMaximumEventProbability',
    'macro_mass_relative_tolerance':'SIDMReferenceMacroMassTolerance',
    'conservation_relative_tolerance':'SIDMReferenceConservationTolerance',
    'energy_absolute_floor':'SIDMReferenceEnergyFloor',
    'momentum_absolute_floor':'SIDMReferenceMomentumFloor',
    'cdf_absolute_tolerance':'SIDMReferenceCDFTolerance',
    'kernel_origin_relative_bound':'SIDMReferenceKernelOriginRelativeBound',
    'maximum_particle_probability':'SIDMReferenceMaximumParticleProbability',
    'maximum_transfer_depth':'SIDMReferenceMaximumTransferDepth',
    'maximum_neglected_moment_ratio':'SIDMReferenceMaximumNeglectedMomentRatio',
    'split_angle_radians':'SIDMReferenceSplitAngleRadians',
    'clock_absolute_tolerance_seconds':'SIDMReferenceClockAbsoluteToleranceSeconds',
    'clock_relative_tolerance':'SIDMReferenceClockRelativeTolerance',
    'maximum_scale_factor_fractional_change':'SIDMReferenceMaximumScaleFactorFractionalChange',
    'exchange_memory_mib':'SIDMReferenceExchangeMemoryMiB',
    'coarse_kernel_support_proper_kpc':'SIDMReferenceCoarseKernelRadiusProperKpc',
    'maximum_kernel_coarse_support_fraction':'SIDMReferenceMaximumKernelCoarseSupportFraction',
    'maximum_omitted_scattering_upper_bound':'SIDMReferenceMaximumOmittedScatteringUpperBound',
}


INTEGER_PARAMETERS={
    'maximum_particles':'SIDMReferenceMaximumParticles',
    'maximum_events_per_step':'SIDMReferenceMaximumEventsPerStep',
    'maximum_substeps':'SIDMReferenceMaximumSubsteps',
    'maximum_clock_panels':'SIDMReferenceMaximumClockPanels',
    'maximum_local_pairs':'SIDMReferenceMaximumLocalPairs',
    'maximum_matching_rounds':'SIDMReferenceMaximumMatchingRounds',
}


EXACT_PARAMETERS={'operator_seed':'SIDMOperatorSeed','maximum_pair_visits':'SIDMReferenceMaximumPairVisits',
    'maximum_exchange_records':'SIDMReferenceMaximumExchangeRecords'}


CHOICES={
    'backend':('SIDMReferenceBackendKind',{'all_pair_reference':0,'distributed_spatial':1}),
    'scheme':('SIDMReferenceSchemeKind',{'rare':0,'frequent':1,'hybrid':2}),
    'kernel_kind':('SIDMReferenceKernelKind',{'uniform_ball':0,'cubic_spline':1}),
    'particle_convention':('SIDMReferenceParticleConvention',{'directed':0,'folded_identical':1}),
}


SIDM_CONFIGURATION_KEYS=frozenset({'schema','enabled','physical_model','event_identity_namespace',
    *FLOAT_PARAMETERS,*INTEGER_PARAMETERS,*EXACT_PARAMETERS,*CHOICES})


SIDM_OPTIONAL_KEYS=frozenset({'event_audit_population'})


def _positive(value,name,zero=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or (value<0 if zero else value<=0):
        raise ValueError('Explicit finite SIDM '+('nonnegative ' if zero else 'positive ')+name+' required')
    return float(value)


def sidm_physical_model(settings):
    reference=settings['physical_model']
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:
        raise ValueError('Native SIDM physical angular model needs its exact retained path and SHA256')
    path=Path(reference['path']).expanduser().resolve()
    if path.stat().st_size>256*1024**2:raise MemoryError('Native SIDM physical model exceeds the bounded 256-MiB table metadata allocation')
    if file_sha256(path)!=reference['sha256']:raise ValueError('Native SIDM physical angular model changed')
    model=json.loads(path.read_text())
    if model.get('schema')=='sidm-elastic-model-v2':
        elastic=ElasticSIDMModel(model)
        if elastic.particle_convention!=settings['particle_convention']:
            raise ValueError('Native differential-model particle convention differs from the requested pair operator')
        return model,{'role':'sidm_angular_model','path':str(path),'sha256':reference['sha256'],'size_bytes':path.stat().st_size}
    if model.get('schema')!='sidm-physical-angular-model-v1' or model.get('kind') not in {'isotropic','screened_rutherford','born_moller'}:
        raise ValueError('Native SIDM currently supports declared isotropic or screened-Rutherford models')
    if model['kind']!='isotropic':_positive(model['velocity_scale_kms'],'physical screening velocity')
    if model['kind']=='born_moller':
        if model.get('regime')!='nonrelativistic_born' or model.get('spin')!='unpolarized_spin_half' or model.get('normalization')!='rutherford_sigma0_including_identical_final_state_factor':
            raise ValueError('Born Moller requires explicit spin, approximation and symmetry normalization')
        if settings['particle_convention']!='folded_identical':
            raise ValueError('Born Moller requires the folded identical-particle convention')
    return model,{'role':'sidm_angular_model','path':str(path),'sha256':reference['sha256'],'size_bytes':path.stat().st_size}


def sidm_native_parameters(settings,*,physics='dm',mpi_ranks=None,max_mem_rank_mib=None,ngenic_creation=False):
    """Return native parameters from the same model preparation used by manifests."""
    parameters, _, _ = _sidm_native_configuration(
        settings, physics=physics, mpi_ranks=mpi_ranks,
        max_mem_rank_mib=max_mem_rank_mib, ngenic_creation=ngenic_creation)
    return parameters


def _sidm_native_configuration(settings,*,physics='dm',mpi_ranks=None,max_mem_rank_mib=None,ngenic_creation=False):
    """Validate once, returning native parameters, physical model and its assets.

    Manifests reuse this result instead of loading the model and generating
    the same adaptive differential table a second time.
    """
    if not isinstance(settings,dict) or type(settings.get('enabled')) is not bool:
        raise ValueError('Native SIDM enabled state must be explicit and boolean')
    if not settings['enabled']:
        if set(settings)-{'enabled','schema'}:raise ValueError('Disabled SIDM configuration cannot silently retain ignored physics settings')
        return {}, None, []
    if settings.get('schema')!='native-sidm-configuration-v1':raise ValueError('Versioned native SIDM configuration required')
    missing=SIDM_CONFIGURATION_KEYS-set(settings)
    unknown=set(settings)-SIDM_CONFIGURATION_KEYS-SIDM_OPTIONAL_KEYS
    if missing or unknown:raise ValueError('Native SIDM fields missing='+','.join(sorted(missing))+' unknown='+','.join(sorted(unknown)))
    if physics not in {'dm','hydro','hydro_stellar','hydro_stellar_bh'}:
        raise ValueError('Native SIDM requires a declared collisionless or supported hydrodynamical gravity/source profile')
    if physics!='dm' and settings['backend']!='distributed_spatial':
        raise ValueError('Hydro+SIDM uses the distributed DM tree source; the isolated all-pair control is fine-DM-only')
    values={}
    zero_fields={'cross_section_normalization_cm2_per_g','macro_mass_relative_tolerance',
        'maximum_kernel_coarse_support_fraction','maximum_omitted_scattering_upper_bound'}
    for field,native in FLOAT_PARAMETERS.items():values[native]=_positive(settings[field],field,field in zero_fields)
    for field,native in INTEGER_PARAMETERS.items():
        value=settings[field]
        if type(value) is not int or not 1<=value<=2**31-1:raise ValueError('Native SIDM exact positive int32 '+field+' required')
        values[native]=value
    if settings['maximum_clock_panels']<2:raise ValueError('Native SIDM physical clock requires at least two panels')
    for field,native in EXACT_PARAMETERS.items():
        value=settings[field]
        if type(value) is not int or not 0<=value<2**64 or field!='operator_seed' and value==0:
            raise ValueError('Native SIDM '+field+' must retain its exact uint64 value')
        values[native]=str(value)
    for field,(native,choices) in CHOICES.items():
        if settings[field] not in choices:raise ValueError('Unknown native SIDM '+field)
        values[native]=choices[settings[field]]
    for field in ('maximum_pair_probability','maximum_particle_probability','maximum_transfer_depth',
                  'kernel_origin_relative_bound','clock_relative_tolerance'):
        if settings[field]>=1:raise ValueError('Native SIDM '+field+' must be smaller than one')
    if settings['maximum_neglected_moment_ratio']>=2:raise ValueError('Native SIDM neglected angular moment ratio must be smaller than two')
    if settings['maximum_kernel_coarse_support_fraction']>1:raise ValueError('Native SIDM kernel coarse fraction cannot exceed one')
    maximum_angle=math.pi/2 if settings['particle_convention']=='folded_identical' else math.pi
    if settings['split_angle_radians']>=maximum_angle:raise ValueError('Native SIDM split angle leaves an empty declared large-angle domain')
    namespace=settings['event_identity_namespace']
    if not isinstance(namespace,str) or not 1<=len(namespace)<=128 or any(ord(c)<33 or ord(c)>126 for c in namespace):
        raise ValueError('Native SIDM event namespace must be one bounded printable parameter-file token')
    values['SIDMEventIdentityNamespace']=namespace
    audit=settings.get('event_audit_population',{'selection':'all'})
    if not isinstance(audit,dict) or audit.get('selection') not in {'all','aggregate_only','hashed_particle_ids'}:
        raise ValueError('Native SIDM immutable event audit needs all, aggregate_only or hashed_particle_ids selection')
    selection=audit['selection']
    if selection=='hashed_particle_ids':
        if set(audit)!={'selection','modulus','residue'} or type(audit['modulus']) is not int or not 2<=audit['modulus']<=2**31-1 or type(audit['residue']) is not int or not 0<=audit['residue']<audit['modulus']:
            raise ValueError('SIDM audit hashed exact uint64 IDs need positive bounded modulus and its residue')
        values['SIDMEventAuditModulus']=audit['modulus'];values['SIDMEventAuditResidue']=audit['residue']
    else:
        if set(audit)!={'selection'}:raise ValueError('Full/aggregate SIDM audits cannot retain ignored population arguments')
        values['SIDMEventAuditModulus']=1 if selection=='all' else 0;values['SIDMEventAuditResidue']=0
    model,asset=sidm_physical_model(settings)
    assets=[asset]
    values['SIDMAngularModelHash']=canonical_digest(model)
    values['SIDMNativeTablePath']='NONE'
    values['SIDMNativeTableSHA256']='NONE'
    if model.get('schema')=='sidm-elastic-model-v2':
        if settings['cross_section_normalization_cm2_per_g'] not in {0,1}:
            raise ValueError('A dimensional differential table fixes its cross section; the legacy normalization must be one, or zero for a collisionless control')
        values['SIDMReferenceAngularKind']=3
        values['SIDMReferenceScreeningVelocityKmPerS']=0.
        if not ngenic_creation:
            # N-GenIC receives only the linear transfer bundle, not scattering.
            table=_native_model_table(settings,model)
            values['SIDMNativeTablePath']=table['path']
            values['SIDMNativeTableSHA256']=table['sha256']
            assets.append({'role':'sidm_differential_table',**table,'size_bytes':Path(table['path']).stat().st_size})
    else:
        values['SIDMReferenceAngularKind']={'isotropic':0,'screened_rutherford':1,'born_moller':2}[model['kind']]
        values['SIDMReferenceScreeningVelocityKmPerS']=0. if model['kind']=='isotropic' else model['velocity_scale_kms']
    if mpi_ranks is not None:
        if type(mpi_ranks) is not int or not 1<=mpi_ranks<=65536:raise ValueError('Native SIDM event identities require 1..65536 MPI ranks')
        if settings['backend']=='all_pair_reference' and mpi_ranks!=1:raise ValueError('All-pair SIDM reference backend requires one MPI rank')
    if max_mem_rank_mib is not None and settings['exchange_memory_mib']>max_mem_rank_mib*.25:
        raise MemoryError('Native SIDM exchange workspace exceeds the actual per-rank native allocation')
    # Linear IC generation stays governed by its selected transfer bundle; the
    # nonlinear SIDM source is compiled into evolution, not the N-GenIC binary.
    return ({} if ngenic_creation else values), model, assets


def _native_model_table(settings,model):
    """Materialize a native input under external build state, not in configs."""
    from .resources import runtime_paths

    source=Path(settings['physical_model']['path']).expanduser().resolve()
    directory=runtime_paths().build_root/'sidm-models'
    destination=directory/(source.stem+'.'+canonical_digest(model)[:20]+'.native-sidm-table.dat')
    return ElasticSIDMModel(model).to_native_table(destination)


def sidm_manifest_record(settings):
    """Describe the actual prepared model, native controls and retained inputs."""
    parameters,model,assets=_sidm_native_configuration(settings)
    if model is None:return {'enabled':False,'kind':'collisionless'}
    return {'enabled':True,'kind':'elastic_sidm','schema':'native-sidm-model-record-v1',
        'configuration':settings,'configuration_hash':canonical_digest(settings),'physical_model':model,
        'angular_model_hash':canonical_digest(model),'native_parameters':parameters,'assets':assets,
        'transfer_policy':'nonlinear interactions do not replace the configured linear transfer bundle',
        'supported_domain':'synchronized equal-mass fine DM; coarse support guarded; gas/stars/BHs coupled through native gravity and separate source kicks; differential tables with full-angle/frequent/hybrid transport',
        'randomness':{'generator':'Random123_Philox4x64_10','key':['run_seed','canonical_uint64_particle_pair','operator_interval','substep','draw_purpose','hybrid_stage'],'rank_independent_draws':True,'trajectory_bitwise_identity_across_mpi_counts':False},
        'production_timestep_policy':'FORCE_EQUAL_TIMESTEPS; SIDM physical source subcycles inside each synchronized gravity interval',
        'production_timestep_cost':'global gravity/hydro step restricted by the fastest particle; source exchange waits for every conflict-free round',
        'coarse_material_approximation':'gravity included; no unequal-resolution scattering; compact-support fraction and omitted hazard bounded for every fine particle',
        'event_audit':settings.get('event_audit_population',{'selection':'all'}),
        'exposure_definition':'cumulative microscopic angular-domain optical depth, including rare and frequent contributions; not a frequent update count',
        'science_acceptance':'NOT_ESTABLISHED'}


SIDM_CONTROL_FIELDS={
    'sidm_kernel':'kernel_support_proper_kpc','sidm_kernel_shape':'kernel_kind',
    'sidm_kernel_origin_bound':'kernel_origin_relative_bound','sidm_timestep':'maximum_pair_probability',
    'sidm_angular_cdf':'cdf_absolute_tolerance','sidm_frequent_approximation':'maximum_transfer_depth',
    'sidm_split_angle':'split_angle_radians','sidm_stochastic_realization':'operator_seed',
}


def sidm_campaign_bindings(controls):
    """Use existing specification controls to alter actual public native answers.

    Returns materializer and measured-manifest bindings, not source measurements
    or permission to omit any requested independent physical control.
    """
    if not isinstance(controls,list) or len(set(controls))!=len(controls) or any(control not in SIDM_CONTROL_FIELDS for control in controls):
        raise ValueError('Distinct actual SIDM campaign control dimensions required')
    native={'phase_seed':[{'native_path':['seed']}]}
    measured={}
    for control in controls:
        field=SIDM_CONTROL_FIELDS[control]
        native[control]=[{'native_path':['sidm',field]}]
        measured[control]={'manifest_path':['sidm_parameters',field]}
    # Operator seed is retained even when the varied dimension is a different
    # numerical control. The materializer has a dedicated operator binding.
    measured['sidm_stochastic_realization']={'manifest_path':['sidm_parameters','operator_seed']}
    native['sidm_stochastic_realization']=[{'native_path':['sidm','operator_seed']}]
    return {'native_configuration_bindings':native,'native_parameter_bindings':measured}


# Independent OpenGadget3 differential-convention mapping

def opengadget3_model_parameters(model_specification, *, velocity_unit_kms, split_angle_radians, scheme,
                                retained_differential_table=None):
    """Map supported public OpenGadget3 model names with exact units/conventions.

    OpenGadget3 functions return d(sigma/m)/dcos(theta), while the GADGET-4
    native model returns d(sigma/m)/dOmega. The azimuth factor is explicit.
    Arbitrary functions/tables use the authored registered reference-table
    adapter inside the pinned OpenGadget3 source. Public mSIDM independently
    computes angular integrals, transport, rates and scattering realizations.
    """
    velocity_unit_kms = _positive(velocity_unit_kms, 'OpenGadget3 velocity unit')
    if scheme not in {"rare", "frequent", "hybrid"}:
        raise ValueError("Matched native velocity units and SIDM scheme required")
    spec = dict(model_specification)
    kind = spec.get("kind")
    environment = {}
    particle_convention=spec.get("particle_convention","folded_identical" if kind=="born_moller" else "directed")
    if particle_convention not in {"directed", "folded_identical"}:
        raise ValueError("Explicit supported independent particle convention required")
    if spec.get("schema") == "sidm-elastic-model-v2":
        model = ElasticSIDMModel(spec)
        if kind == "constant_isotropic":
            cross_section = float(model.total_cross_section_per_mass(0.))
            # OpenGadget3 returns d(sigma/m)/dmu = sigma_0; integrating the
            # full mu interval [-1,1] gives total cross section 2*sigma_0.
            name, params = "isotropic_scattering", {"sigma_0": cross_section / 2.}
        elif kind == "screened_rutherford" and spec["supplied_cross_section"] == "total":
            name, params = "rutherford_scattering", {"sigma_0": spec["cross_section_cm2_per_g"], "w": spec["velocity_scale_kms"] / velocity_unit_kms}
        else:
            if not isinstance(retained_differential_table,dict) or set(retained_differential_table)!={"path","sha256"}:
                raise ValueError("Independent arbitrary differential model requires its materialized retained table and SHA256")
            table_path=Path(retained_differential_table['path']).expanduser().resolve()
            if file_sha256(table_path)!=retained_differential_table['sha256']:
                raise ValueError("Independent differential table identity changed")
            name,params="research_table_scattering",{"velocity_unit_kms":velocity_unit_kms}
            environment={"RESEARCH_SIDM_REFERENCE_TABLE":str(table_path)}
    elif spec.get("schema") == "sidm-physical-angular-model-v1" and kind in {"isotropic", "screened_rutherford", "born_moller"}:
        if "cross_section_normalization_cm2_per_g" not in spec:
            raise ValueError("Legacy angular model adapter needs the actual native dimensional normalization")
        normalization = _positive(spec["cross_section_normalization_cm2_per_g"], 'independent-reference normalization', zero=True)
        if kind == "isotropic":
            name, params = "isotropic_scattering", {"sigma_0": normalization / 2.}
        else:
            name = "rutherford_scattering" if kind == "screened_rutherford" else "moeller_scattering"
            scale = _positive(spec["velocity_scale_kms"], 'independent-reference screening velocity')
            params = {"sigma_0": normalization, "w": scale / velocity_unit_kms}
    else:
        raise ValueError("Supported explicit independent-engine angular model required")
    if type(split_angle_radians) not in (int, float) or not math.isfinite(split_angle_radians) or not 0 < split_angle_radians < math.pi / 2:
        raise ValueError("Independent forward/backward split angle must lie strictly in (0,pi/2)")
    cutoff = split_angle_radians if scheme == "hybrid" else 0. if scheme == "rare" else math.pi / 2
    return {"engine": "OpenGadget3", "MSIDM_CrossSection": name, "cross_section_parameters": params, "MSIDM_DCS_params": ";".join(f"{key}={float(value):.17g}" for key,value in params.items()),
            "environment":environment,"retained_differential_table":retained_differential_table,
            "MSIDM_thetaC_fwd": cutoff, "MSIDM_thetaC_bwd": cutoff if particle_convention=="folded_identical" else 0.,
            "differential_convention": "dsigma_per_mass_dcos_theta=2*pi*dsigma_per_mass_dOmega",
            "velocity_parameter_units": "native_code_velocity", "cross_section_parameter_units": "cm2_per_g",
            "particle_convention":particle_convention,
            "required_reference_build_features":["mSIDM","mSIDM_DIFFERENTIAL_CROSS_SECTION","DCS_USE_ZSLICE"]+(["rSIDM"] if scheme=="rare" else ["fSIDM"] if scheme=="frequent" else ["rSIDM","fSIDM"])+(["rSIDM_SANGLE90"] if particle_convention=="folded_identical" else []),
            "execution_status": "NOT_RUN"}
