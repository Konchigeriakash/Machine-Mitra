"""Explainable anomaly detection engine for Machine Mitra.

Compares vibration feature vectors against a fitted NormalBaseline, calculates
robust deviations using median and MAD, ranks contributing physical features,
and classifies operating condition into NORMAL, WARNING, or ANOMALOUS states.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from machinemitra.features.extractor import WindowFeatures
from machinemitra.models.baseline import NormalBaseline


class AnomalyState(str, Enum):
    """Categorical operating condition determined by anomaly evaluation."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    ANOMALOUS = "ANOMALOUS"


@dataclass(frozen=True)
class FeatureContribution:
    """Individual feature deviation metrics explaining an anomaly.

    Attributes:
        feature_name: Name of the vibration feature.
        deviation_score: Robust z-score (|actual - median| / scale).
        actual_value: Measured feature value in the evaluated window.
        baseline_median: Learned normal median for this feature.
        baseline_mad: Learned normal Median Absolute Deviation.
        signed_deviation: Signed difference (actual - median).
    """

    feature_name: str
    deviation_score: float
    actual_value: float
    baseline_median: float
    baseline_mad: float
    signed_deviation: float


@dataclass(frozen=True)
class AnomalyResult:
    """Structured, explainable outcome of an anomaly detection evaluation.

    Attributes:
        machine_id: Asset identifier.
        state: Classified operational state (NORMAL, WARNING, ANOMALOUS).
        anomaly_score: Aggregate robust deviation score (RMS over all features).
        warning_threshold: Configured threshold for WARNING state on aggregate RMS.
        alert_threshold: Configured threshold for ANOMALOUS state on aggregate RMS.
        is_anomaly: True if state is ANOMALOUS.
        contributing_features: Tuple of feature contributions ranked by deviation score.
        feature_deviations: Mapping of all feature names to their deviation scores.
        window_idx: Optional window index from the source telemetry.
        timestamp_ms: Optional window start timestamp in milliseconds.
        feature_warning_threshold: Optional individual feature deviation threshold for WARNING.
        feature_alert_threshold: Optional individual feature deviation threshold for ANOMALOUS.
    """

    machine_id: str
    state: AnomalyState
    anomaly_score: float
    warning_threshold: float
    alert_threshold: float
    is_anomaly: bool
    contributing_features: tuple[FeatureContribution, ...]
    feature_deviations: dict[str, float]
    window_idx: int | None = None
    timestamp_ms: float | None = None
    feature_warning_threshold: float | None = None
    feature_alert_threshold: float | None = None

    @property
    def top_contributor(self) -> FeatureContribution | None:
        """The single feature exhibiting the largest deviation from normal."""
        return self.contributing_features[0] if self.contributing_features else None

    def top_contributors(self, count: int = 3) -> tuple[FeatureContribution, ...]:
        """Return the top N features exhibiting the largest deviation from normal."""
        return self.contributing_features[:count]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a flat dictionary for tabular analytics."""
        base: dict[str, Any] = {
            "machine_id": self.machine_id,
            "state": self.state.value,
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "warning_threshold": self.warning_threshold,
            "alert_threshold": self.alert_threshold,
            "feature_warning_threshold": self.feature_warning_threshold,
            "feature_alert_threshold": self.feature_alert_threshold,
            "window_idx": self.window_idx,
            "timestamp_ms": self.timestamp_ms,
        }
        if self.top_contributor is not None:
            base["top_feature"] = self.top_contributor.feature_name
            base["top_deviation_score"] = self.top_contributor.deviation_score
            base["top_actual_value"] = self.top_contributor.actual_value
            base["top_baseline_median"] = self.top_contributor.baseline_median
            base["top_signed_deviation"] = self.top_contributor.signed_deviation

        return base


class ExplainableAnomalyDetector:
    """Anomaly detector combining normal-only baseline statistics with explainability.

    Evaluates feature vectors against a trained NormalBaseline using robust
    Median/MAD scaling, ranks the physical vibration metrics driving any drift,
    and identifies anomalies without black-box opacity.

    Threshold Interaction (Dual-Criterion):
    ---------------------------------------
    The detector uses a complementary dual-criterion classification scheme:
    1. Aggregate RMS Score:
       score = sqrt(mean(z_j ** 2))
       Detects broadband, multi-component degradation where several features drift
       simultaneously (e.g. progressive bearing wear, structural looseness), even if
       no single feature is catastrophic.
    2. Maximum Individual Feature Deviation:
       max_dev = max(z_j)
       Prevents localized mechanical faults (e.g. impact shock affecting only kurtosis
       or single-axis displacement) from being diluted by quiescent normal features.
    The final state (NORMAL, WARNING, ANOMALOUS) is assigned as the maximum severity
    triggered by either criterion.
    """

    def __init__(
        self,
        baseline: NormalBaseline,
        *,
        warning_threshold: float = 2.5,
        alert_threshold: float = 4.0,
        feature_warning_threshold: float | None = 4.0,
        feature_alert_threshold: float | None = 6.0,
        top_k: int = 3,
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            baseline: A fitted NormalBaseline instance.
            warning_threshold: Deviation score threshold for WARNING state on aggregate RMS (default 2.5).
            alert_threshold: Deviation score threshold for ANOMALOUS state on aggregate RMS (default 4.0).
            feature_warning_threshold: Optional threshold on any single feature deviation for WARNING state (default 4.0).
            feature_alert_threshold: Optional threshold on any single feature deviation for ANOMALOUS state (default 6.0).
            top_k: Number of highest-deviating features prioritized for reporting (default 3).

        Raises:
            ValueError: If thresholds are non-positive or inconsistent, or top_k < 1.
        """
        if warning_threshold <= 0:
            raise ValueError(
                f"warning_threshold must be strictly positive, got {warning_threshold}."
            )
        if alert_threshold <= warning_threshold:
            raise ValueError(
                f"alert_threshold ({alert_threshold}) must be strictly greater than "
                f"warning_threshold ({warning_threshold})."
            )
        if feature_warning_threshold is not None and feature_warning_threshold <= 0:
            raise ValueError(
                f"feature_warning_threshold must be strictly positive, got {feature_warning_threshold}."
            )
        if feature_alert_threshold is not None:
            if feature_alert_threshold <= 0:
                raise ValueError(
                    f"feature_alert_threshold must be strictly positive, got {feature_alert_threshold}."
                )
            if (
                feature_warning_threshold is not None
                and feature_alert_threshold <= feature_warning_threshold
            ):
                raise ValueError(
                    f"feature_alert_threshold ({feature_alert_threshold}) must be strictly greater than "
                    f"feature_warning_threshold ({feature_warning_threshold})."
                )
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")

        self.baseline = baseline
        self.warning_threshold = float(warning_threshold)
        self.alert_threshold = float(alert_threshold)
        self.feature_warning_threshold = (
            float(feature_warning_threshold) if feature_warning_threshold is not None else None
        )
        self.feature_alert_threshold = (
            float(feature_alert_threshold) if feature_alert_threshold is not None else None
        )
        self.top_k = int(top_k)

    def predict(
        self,
        sample: Union[WindowFeatures, pd.Series, Mapping[str, Any]],
    ) -> AnomalyResult:
        """Evaluate an incoming vibration feature sample against the normal baseline.

        Args:
            sample: A WindowFeatures instance, pd.Series, or dictionary mapping
                feature names to numerical values.

        Returns:
            Structured AnomalyResult containing state, score, and ranked feature contributions.

        Raises:
            ValueError: If required baseline features are missing or non-finite.
            TypeError: If sample format is unsupported.
        """
        window_idx: int | None = None
        timestamp_ms: float | None = None

        # 1. Normalize sample into a dictionary
        if isinstance(sample, WindowFeatures):
            sample_dict = sample.to_dict()
            window_idx = sample.window_idx
            timestamp_ms = sample.start_time_ms
        elif isinstance(sample, pd.Series):
            sample_dict = sample.to_dict()
            window_idx = sample.get("window_idx")
            timestamp_ms = sample.get("start_time_ms")
        elif isinstance(sample, Mapping):
            sample_dict = dict(sample)
            window_idx = sample.get("window_idx")
            timestamp_ms = sample.get("start_time_ms")
        else:
            raise TypeError(
                f"Unsupported sample type: {type(sample)}. Expected WindowFeatures, pd.Series, or dict."
            )

        # 2. Verify all baseline features are present and finite
        missing = [f for f in self.baseline.feature_names if f not in sample_dict]
        if missing:
            raise ValueError(
                f"Sample is missing required baseline feature(s): {', '.join(sorted(missing))}."
            )

        contributions: list[FeatureContribution] = []
        deviations: dict[str, float] = {}

        for feat in self.baseline.feature_names:
            raw_val = sample_dict[feat]
            try:
                val = float(raw_val)
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"Feature '{feat}' has non-numeric value: {raw_val}."
                ) from err

            if not np.isfinite(val):
                raise ValueError(
                    f"Feature '{feat}' contains non-finite value: {val}."
                )

            stats = self.baseline.feature_stats[feat]
            signed_diff = val - stats.median
            z_score = float(abs(signed_diff) / stats.scale)

            deviations[feat] = z_score
            contributions.append(
                FeatureContribution(
                    feature_name=feat,
                    deviation_score=z_score,
                    actual_value=val,
                    baseline_median=stats.median,
                    baseline_mad=stats.mad,
                    signed_deviation=signed_diff,
                )
            )

        # 3. Rank features by deviation score descending
        contributions.sort(key=lambda c: c.deviation_score, reverse=True)

        # 4. Compute overall anomaly score using root-mean-square of all feature deviations
        all_scores = np.array([c.deviation_score for c in contributions], dtype=float)
        overall_score = float(np.sqrt(np.mean(all_scores**2)))

        # 5. Classify state based on aggregate RMS and maximum feature deviation
        # Aggregate evaluation (broadband multi-feature drift)
        if overall_score >= self.alert_threshold:
            state = AnomalyState.ANOMALOUS
        elif overall_score >= self.warning_threshold:
            state = AnomalyState.WARNING
        else:
            state = AnomalyState.NORMAL

        # Maximum individual feature evaluation (prevents dilution of severe localized faults)
        max_dev = contributions[0].deviation_score if contributions else 0.0
        if self.feature_alert_threshold is not None and max_dev >= self.feature_alert_threshold:
            state = AnomalyState.ANOMALOUS
        elif (
            self.feature_warning_threshold is not None
            and max_dev >= self.feature_warning_threshold
            and state == AnomalyState.NORMAL
        ):
            state = AnomalyState.WARNING

        return AnomalyResult(
            machine_id=self.baseline.machine_id,
            state=state,
            anomaly_score=overall_score,
            warning_threshold=self.warning_threshold,
            alert_threshold=self.alert_threshold,
            is_anomaly=(state == AnomalyState.ANOMALOUS),
            contributing_features=tuple(contributions),
            feature_deviations=deviations,
            window_idx=int(window_idx) if window_idx is not None else None,
            timestamp_ms=float(timestamp_ms) if timestamp_ms is not None else None,
            feature_warning_threshold=self.feature_warning_threshold,
            feature_alert_threshold=self.feature_alert_threshold,
        )

    def predict_batch(
        self,
        samples: Union[Sequence[WindowFeatures], pd.DataFrame],
    ) -> list[AnomalyResult]:
        """Batch predict anomaly states across multiple feature samples.

        Args:
            samples: Sequence of WindowFeatures or a pandas DataFrame.

        Returns:
            List of AnomalyResult instances.
        """
        if isinstance(samples, pd.DataFrame):
            return [self.predict(row) for _, row in samples.iterrows()]
        if isinstance(samples, Sequence):
            return [self.predict(s) for s in samples]
        raise TypeError(
            f"Unsupported samples type: {type(samples)}. Expected Sequence[WindowFeatures] or pd.DataFrame."
        )


def results_to_dataframe(results: Sequence[AnomalyResult]) -> pd.DataFrame:
    """Convert a sequence of AnomalyResult instances to a pandas DataFrame."""
    if len(results) == 0:
        return pd.DataFrame()
    return pd.DataFrame([r.to_dict() for r in results])
