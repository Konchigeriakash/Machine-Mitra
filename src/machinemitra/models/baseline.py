"""Normal-only baseline model for Machine Mitra vibration analytics.

Fits per-feature median and Median Absolute Deviation (MAD) strictly on
normal operating data. Supports safe JSON persistence and machine-specific isolation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from machinemitra.features.extractor import WindowFeatures


BASELINE_SCHEMA_VERSION = "1.0"

# Standard time- and frequency-domain features monitored for baseline
DEFAULT_BASELINE_FEATURES: tuple[str, ...] = (
    "rms_x",
    "rms_y",
    "rms_z",
    "rms_mag",
    "peak_x",
    "peak_y",
    "peak_z",
    "peak_mag",
    "peak_to_peak_x",
    "peak_to_peak_y",
    "peak_to_peak_z",
    "peak_to_peak_mag",
    "crest_factor_x",
    "crest_factor_y",
    "crest_factor_z",
    "crest_factor_mag",
    "kurtosis_x",
    "kurtosis_y",
    "kurtosis_z",
    "kurtosis_mag",
    "dominant_frequency_hz",
    "total_spectral_energy",
)


@dataclass(frozen=True)
class FeatureStats:
    """Robust statistical moments for a single vibration feature.

    Attributes:
        median: Median value under normal operation.
        mad: Median Absolute Deviation (MAD = median(|x - median|)).
        scale: Normal-consistent dispersion scale (1.4826 * MAD), protected
            against zero with a safe minimum epsilon.
    """

    median: float
    mad: float
    scale: float


@dataclass(frozen=True)
class NormalBaseline:
    """Fitted baseline distribution representing normal operating condition.

    Attributes:
        machine_id: Machine identifier this baseline was trained on.
        feature_stats: Mapping of feature names to their robust baseline parameters.
        feature_names: Ordered tuple of monitored feature names.
        sample_count: Number of normal operating windows used to fit this baseline.
        fitted_at: ISO 8601 UTC timestamp of baseline fitting.
        version: Serialization schema version.
    """

    machine_id: str
    feature_stats: dict[str, FeatureStats]
    feature_names: tuple[str, ...]
    sample_count: int
    fitted_at: str
    version: str = BASELINE_SCHEMA_VERSION

    @classmethod
    def fit(
        cls,
        normal_data: Union[pd.DataFrame, Sequence[WindowFeatures]],
        *,
        machine_id: str = "default_machine",
        feature_names: Sequence[str] | None = None,
        min_scale: float = 1e-3,
        min_scale_relative: float = 0.01,
    ) -> NormalBaseline:
        """Fit a normal-only baseline from training data.

        Args:
            normal_data: DataFrame or sequence of WindowFeatures representing
                confirmed normal operating windows.
            machine_id: Identifier for the asset (e.g. "Fan-01").
            feature_names: Optional explicit list of feature column names. If None,
                uses DEFAULT_BASELINE_FEATURES plus any present energy bands.
            min_scale: Minimum absolute dispersion scale to prevent division by zero
                and excessive sensitivity to normal measurement noise for constant or
                quiescent features. Default is 1e-3 (1 mg for acceleration features).
            min_scale_relative: Minimum relative dispersion scale expressed as a fraction
                of the feature median (default 0.01 = 1%). Prevents zero-MAD features from
                producing massive deviation scores under legitimate sensor noise.

        Returns:
            Fitted NormalBaseline instance.

        Raises:
            ValueError: If normal_data is empty or contains non-finite values.
            ValueError: If requested feature columns are missing from normal_data.
            ValueError: If min_scale <= 0 or min_scale_relative < 0.
        """
        if min_scale <= 0:
            raise ValueError(f"min_scale must be strictly positive, got {min_scale}.")
        if min_scale_relative < 0:
            raise ValueError(
                f"min_scale_relative must be non-negative, got {min_scale_relative}."
            )
        # 1. Convert input to DataFrame if necessary
        if isinstance(normal_data, pd.DataFrame):
            df = normal_data
        elif isinstance(normal_data, Sequence):
            if len(normal_data) == 0:
                raise ValueError("Cannot fit baseline from an empty sequence of WindowFeatures.")
            if isinstance(normal_data[0], WindowFeatures):
                df = pd.DataFrame([f.to_dict() for f in normal_data])
            else:
                raise TypeError(
                    f"Unsupported sequence element type: {type(normal_data[0])}. "
                    "Expected WindowFeatures."
                )
        else:
            raise TypeError(
                f"Unsupported data type: {type(normal_data)}. Expected pd.DataFrame or Sequence[WindowFeatures]."
            )

        if len(df) == 0:
            raise ValueError("Training data contains zero samples. Cannot fit baseline.")
        if len(df) < 2:
            raise ValueError(
                f"Insufficient training samples ({len(df)}). At least 2 normal windows "
                "are required to estimate baseline variance."
            )

        # 2. Determine feature columns to fit
        if feature_names is not None:
            selected_features = list(feature_names)
        else:
            # Include standard features and any dynamic energy band columns
            selected_features = [col for col in DEFAULT_BASELINE_FEATURES if col in df.columns]
            band_cols = [col for col in df.columns if col.startswith("energy_")]
            for bc in band_cols:
                if bc not in selected_features:
                    selected_features.append(bc)

        if len(selected_features) == 0:
            raise ValueError("No matching feature columns found in normal training data.")

        # Check for missing requested features
        missing = [f for f in selected_features if f not in df.columns]
        if missing:
            raise ValueError(
                f"Training data is missing requested feature(s): {', '.join(sorted(missing))}."
            )

        # 3. Compute Median and MAD per feature
        stats: dict[str, FeatureStats] = {}
        for feat in selected_features:
            series = pd.to_numeric(df[feat], errors="coerce")
            if series.isna().any():
                raise ValueError(
                    f"Feature '{feat}' contains null, missing, or non-numeric values in training data."
                )
            arr = series.to_numpy(dtype=float)
            if not np.isfinite(arr).all():
                raise ValueError(f"Feature '{feat}' contains non-finite values in training data.")

            med = float(np.median(arr))
            abs_dev = np.abs(arr - med)
            mad = float(np.median(abs_dev))
            # Normal-consistent scale: 1.4826 * MAD with physical lower-bound protection.
            # Protects against zero-MAD explosion from minor sensor noise.
            scale_floor = max(min_scale, min_scale_relative * abs(med))
            scale = float(max(1.4826 * mad, scale_floor))

            stats[feat] = FeatureStats(median=med, mad=mad, scale=scale)

        fitted_timestamp = datetime.now(timezone.utc).isoformat()

        return cls(
            machine_id=machine_id,
            feature_stats=stats,
            feature_names=tuple(selected_features),
            sample_count=len(df),
            fitted_at=fitted_timestamp,
            version=BASELINE_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize baseline parameters to a structured dictionary."""
        return {
            "version": self.version,
            "machine_id": self.machine_id,
            "fitted_at": self.fitted_at,
            "sample_count": self.sample_count,
            "feature_names": list(self.feature_names),
            "parameters": {
                name: {
                    "median": s.median,
                    "mad": s.mad,
                    "scale": s.scale,
                }
                for name, s in self.feature_stats.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NormalBaseline:
        """Construct a NormalBaseline from a validated dictionary.

        Raises:
            ValueError: If dictionary format or version is incompatible or corrupt.
        """
        if not isinstance(data, Mapping):
            raise ValueError("Baseline data must be a dictionary.")

        version = str(data.get("version", ""))
        if not version.startswith("1."):
            raise ValueError(
                f"Incompatible baseline version: '{version}'. Expected version 1.x."
            )

        machine_id = str(data.get("machine_id", "unknown_machine"))
        fitted_at = str(data.get("fitted_at", ""))
        sample_count = int(data.get("sample_count", 0))

        raw_params = data.get("parameters")
        if not isinstance(raw_params, Mapping) or len(raw_params) == 0:
            raise ValueError("Baseline parameters are missing or empty.")

        stats: dict[str, FeatureStats] = {}
        for name, p in raw_params.items():
            if not isinstance(p, Mapping):
                raise ValueError(f"Malformed parameter block for feature '{name}'.")
            try:
                med = float(p["median"])
                mad = float(p["mad"])
                scale = float(p["scale"])
            except (KeyError, TypeError, ValueError) as err:
                raise ValueError(
                    f"Invalid or missing median/mad/scale for feature '{name}': {p}"
                ) from err

            if scale <= 0 or not np.isfinite(scale):
                raise ValueError(
                    f"Invalid non-positive or non-finite scale for feature '{name}': {scale}."
                )
            stats[name] = FeatureStats(median=med, mad=mad, scale=scale)

        feature_names = tuple(data.get("feature_names", list(stats.keys())))

        return cls(
            machine_id=machine_id,
            feature_stats=stats,
            feature_names=feature_names,
            sample_count=sample_count,
            fitted_at=fitted_at,
            version=version,
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize baseline to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> NormalBaseline:
        """Deserialize baseline from a JSON string.

        Raises:
            ValueError: If JSON is malformed or invalid.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as err:
            raise ValueError(f"Malformed JSON string for NormalBaseline: {err}") from err
        return cls.from_dict(data)

    def save(self, file_path: Union[str, Path]) -> Path:
        """Persist baseline parameters to a JSON file.

        Args:
            file_path: Destination path for the JSON file.

        Returns:
            Resolved Path instance of the saved file.
        """
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> NormalBaseline:
        """Load and validate baseline parameters from a JSON file.

        Args:
            file_path: Path to the stored baseline JSON.

        Returns:
            Validated NormalBaseline instance.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If file content is corrupt or incompatible.
        """
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"Baseline file not found: {source}")
        text = source.read_text(encoding="utf-8")
        return cls.from_json(text)
