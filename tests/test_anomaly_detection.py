"""Unit tests for normal-only baseline modeling and explainable anomaly detection."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from machinemitra.features import (
    DEFAULT_FREQUENCY_BANDS,
    WindowFeatures,
    extract_features_from_windows,
)
from machinemitra.models import (
    DEFAULT_BASELINE_FEATURES,
    AnomalyResult,
    AnomalyState,
    ExplainableAnomalyDetector,
    FeatureContribution,
    NormalBaseline,
    results_to_dataframe,
)
from machinemitra.processing.windowing import Window


def make_synthetic_window_sequence(
    window_count: int,
    fs: float = 200.0,
    noise_level: float = 0.02,
    fault_type: str | None = None,
) -> list[Window]:
    """Generate synthetic windows of normal or faulty operation for testing."""
    windows: list[Window] = []
    n_samples = int(fs * 1.0)  # 1.0s window
    dt_ms = 1000.0 / fs
    rng = np.random.default_rng(42)

    for i in range(window_count):
        t_s = np.arange(n_samples) / fs
        # Normal baseline: 25 Hz rotational vibration with low noise
        x = 0.4 * np.sin(2 * np.pi * 25 * t_s) + rng.normal(0, noise_level, n_samples)
        y = 0.2 * np.cos(2 * np.pi * 25 * t_s) + rng.normal(0, noise_level, n_samples)
        z = 0.98 + 0.1 * np.sin(2 * np.pi * 50 * t_s) + rng.normal(0, noise_level, n_samples)

        if fault_type == "unbalance":
            # Severe 1X rotational unbalance on X and Y
            x += 2.5 * np.sin(2 * np.pi * 25 * t_s)
            y += 1.8 * np.cos(2 * np.pi * 25 * t_s)
        elif fault_type == "looseness":
            # Broadband structural looseness across all axes
            x += rng.normal(0, 0.8, n_samples)
            y += rng.normal(0, 0.8, n_samples)
            z += rng.normal(0, 0.8, n_samples)
        elif fault_type == "warning_drift":
            # Moderate drift (approaching warning threshold)
            x += 0.3 * np.sin(2 * np.pi * 25 * t_s)

        start_time_ms = float(i * 1000.0)
        timestamps = start_time_ms + np.arange(n_samples) * dt_ms
        df = pd.DataFrame(
            {
                "timestamp_ms": timestamps,
                "accel_x_g": x,
                "accel_y_g": y,
                "accel_z_g": z,
            }
        )
        windows.append(
            Window(
                window_idx=i,
                start_time_ms=start_time_ms,
                end_time_ms=float(timestamps[-1]),
                duration_s=float((timestamps[-1] - timestamps[0]) / 1000.0),
                fs=fs,
                data=df,
            )
        )

    return windows


class TestNormalBaselineFitting:
    """Tests verifying normal-only baseline fitting mechanics."""

    def test_fit_from_window_features(self) -> None:
        windows = make_synthetic_window_sequence(window_count=30, fault_type=None)
        features = extract_features_from_windows(windows)

        baseline = NormalBaseline.fit(features, machine_id="Fan-Unit-01")

        assert baseline.machine_id == "Fan-Unit-01"
        assert baseline.sample_count == 30
        assert len(baseline.feature_stats) >= len(DEFAULT_BASELINE_FEATURES)

        # Baseline RMS X should be around 0.4 / sqrt(2) ≈ 0.283
        rms_x_stats = baseline.feature_stats["rms_x"]
        assert rms_x_stats.median == pytest.approx(0.285, abs=0.03)
        assert rms_x_stats.mad > 0.0
        assert rms_x_stats.scale > 0.0

    def test_fit_from_dataframe(self) -> None:
        windows = make_synthetic_window_sequence(window_count=20, fault_type=None)
        features = extract_features_from_windows(windows)
        df = pd.DataFrame([f.to_dict() for f in features])

        baseline = NormalBaseline.fit(df, machine_id="Motor-02")
        assert baseline.machine_id == "Motor-02"
        assert baseline.sample_count == 20

    def test_zero_mad_safe_handling(self) -> None:
        # Create dataset where one feature is perfectly constant (MAD = 0.0)
        df = pd.DataFrame(
            {
                "rms_x": [0.5, 0.5, 0.5, 0.5, 0.5],
                "rms_y": [0.1, 0.2, 0.15, 0.22, 0.18],
                "rms_z": [0.98, 0.98, 0.98, 0.98, 0.98],
                "rms_mag": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        # Default fitting applies physical floor: max(min_scale, 0.01 * median)
        baseline = NormalBaseline.fit(df, feature_names=["rms_x", "rms_y", "rms_z", "rms_mag"])

        assert baseline.feature_stats["rms_x"].mad == 0.0
        # Scale must be protected with physical noise floor: max(1e-3, 0.01 * 0.5) = 0.005
        assert baseline.feature_stats["rms_x"].scale > 0.0
        assert baseline.feature_stats["rms_x"].scale == pytest.approx(0.005)

        # Explicit min_scale override works when desired
        custom_baseline = NormalBaseline.fit(
            df,
            feature_names=["rms_x", "rms_y", "rms_z", "rms_mag"],
            min_scale=1e-6,
            min_scale_relative=0.0,
        )
        assert custom_baseline.feature_stats["rms_x"].scale == 1e-6

    def test_empty_or_single_sample_training_rejected(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            NormalBaseline.fit(pd.DataFrame())
        assert "zero samples" in str(exc_info.value)

        single_sample_df = pd.DataFrame({"rms_x": [0.5]})
        with pytest.raises(ValueError) as exc_info:
            NormalBaseline.fit(single_sample_df)
        assert "Insufficient training samples" in str(exc_info.value)


class TestExplainableAnomalyDetector:
    """Tests verifying detection of normal, warning, and anomalous states."""

    @pytest.fixture
    def fitted_detector(self) -> ExplainableAnomalyDetector:
        # Fit baseline strictly on 40 normal windows
        normal_windows = make_synthetic_window_sequence(window_count=40, fault_type=None)
        normal_features = extract_features_from_windows(normal_windows)
        baseline = NormalBaseline.fit(normal_features, machine_id="Fan-Unit-A")
        return ExplainableAnomalyDetector(
            baseline,
            warning_threshold=2.5,
            alert_threshold=4.0,
            top_k=3,
        )

    def test_normal_data_evaluates_as_normal(
        self,
        fitted_detector: ExplainableAnomalyDetector,
    ) -> None:
        # Evaluate on fresh held-out normal windows (evaluation track)
        held_out_normal = make_synthetic_window_sequence(window_count=10, fault_type=None)
        held_out_features = extract_features_from_windows(held_out_normal)

        results = fitted_detector.predict_batch(held_out_features)
        assert len(results) == 10

        for r in results:
            assert isinstance(r, AnomalyResult)
            assert r.state == AnomalyState.NORMAL
            assert r.is_anomaly is False
            assert r.anomaly_score < fitted_detector.warning_threshold

    def test_severe_unbalance_detected_and_explained(
        self,
        fitted_detector: ExplainableAnomalyDetector,
    ) -> None:
        # Evaluate on held-out unbalance fault windows
        fault_windows = make_synthetic_window_sequence(window_count=5, fault_type="unbalance")
        fault_features = extract_features_from_windows(fault_windows)

        results = fitted_detector.predict_batch(fault_features)
        assert len(results) == 5

        for r in results:
            assert r.state == AnomalyState.ANOMALOUS
            assert r.is_anomaly is True
            assert r.anomaly_score >= fitted_detector.alert_threshold

            # Explainability check: Top contributing feature must be unbalance-related (X or Y axis)
            top = r.top_contributor
            assert top is not None
            assert top.deviation_score > 5.0  # Significant sigma drift
            assert any(axis in top.feature_name for axis in ["x", "y", "mag"])
            # Ensure signed deviation is positive (vibration increased)
            assert top.signed_deviation > 0.0

            top_3 = r.top_contributors(3)
            assert len(top_3) == 3
            assert top_3[0] == top
            assert top_3[0].deviation_score >= top_3[1].deviation_score >= top_3[2].deviation_score

    def test_moderate_drift_triggers_warning_state(
        self,
        fitted_detector: ExplainableAnomalyDetector,
    ) -> None:
        warning_windows = make_synthetic_window_sequence(window_count=5, fault_type="warning_drift")
        warning_features = extract_features_from_windows(warning_windows)

        results = fitted_detector.predict_batch(warning_features)
        assert len(results) == 5

        # At least some windows should be in WARNING or ANOMALOUS (score >= warning_threshold)
        elevated = [r for r in results if r.state in (AnomalyState.WARNING, AnomalyState.ANOMALOUS)]
        assert len(elevated) > 0
        assert elevated[0].anomaly_score >= fitted_detector.warning_threshold

    def test_missing_features_rejected(
        self,
        fitted_detector: ExplainableAnomalyDetector,
    ) -> None:
        incomplete_sample = {"rms_x": 0.5, "rms_y": 0.3}  # Missing other baseline features
        with pytest.raises(ValueError) as exc_info:
            fitted_detector.predict(incomplete_sample)
        assert "missing required baseline feature" in str(exc_info.value)

    def test_non_finite_feature_values_rejected(
        self,
        fitted_detector: ExplainableAnomalyDetector,
    ) -> None:
        normal_windows = make_synthetic_window_sequence(window_count=1, fault_type=None)
        feat = extract_features_from_windows(normal_windows)[0]
        feat_dict = feat.to_dict()
        feat_dict["rms_x"] = np.nan

        with pytest.raises(ValueError) as exc_info:
            fitted_detector.predict(feat_dict)
        assert "non-finite" in str(exc_info.value)

    def test_results_to_dataframe(
        self,
        fitted_detector: ExplainableAnomalyDetector,
    ) -> None:
        normal_windows = make_synthetic_window_sequence(window_count=3, fault_type=None)
        features = extract_features_from_windows(normal_windows)
        results = fitted_detector.predict_batch(features)

        df = results_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "state" in df.columns
        assert "anomaly_score" in df.columns
        assert "top_feature" in df.columns

    def test_single_extreme_feature_not_diluted(
        self,
        fitted_detector: ExplainableAnomalyDetector,
    ) -> None:
        """Verify an extreme spike on a single feature triggers ANOMALOUS despite 21 normal features."""
        # Start with normal baseline median values for all features
        sample = {
            feat: stats.median
            for feat, stats in fitted_detector.baseline.feature_stats.items()
        }
        # Spike a single feature (rms_x) to 10 robust sigmas
        rms_x_stats = fitted_detector.baseline.feature_stats["rms_x"]
        sample["rms_x"] = rms_x_stats.median + 10.0 * rms_x_stats.scale

        result = fitted_detector.predict(sample)

        # Aggregate RMS with 1 feature at 10.0 and 21 features at 0.0:
        # sqrt((10^2) / 22) ≈ 2.13, which is LESS than warning_threshold (2.5)
        assert result.anomaly_score < fitted_detector.warning_threshold
        # But individual feature threshold prevents dilution into NORMAL
        assert result.state == AnomalyState.ANOMALOUS
        assert result.is_anomaly is True
        assert result.top_contributor is not None
        assert result.top_contributor.feature_name == "rms_x"
        assert result.top_contributor.deviation_score == pytest.approx(10.0, rel=1e-3)

    def test_multiple_moderate_deviations_trigger_aggregate(
        self,
        fitted_detector: ExplainableAnomalyDetector,
    ) -> None:
        """Verify multi-feature moderate drift triggers warning via RMS without individual extreme spikes."""
        sample = {
            feat: stats.median
            for feat, stats in fitted_detector.baseline.feature_stats.items()
        }
        # Shift 16 features moderately by 3.2 robust sigmas (below individual thresholds: warning 4.0, alert 6.0)
        feature_list = list(fitted_detector.baseline.feature_stats.keys())
        for feat in feature_list[:16]:
            st = fitted_detector.baseline.feature_stats[feat]
            sample[feat] = st.median + 3.2 * st.scale

        result = fitted_detector.predict(sample)

        # Max individual deviation is 3.2 (below both 4.0 warning and 6.0 alert feature thresholds)
        assert result.top_contributor is not None
        assert result.top_contributor.deviation_score == pytest.approx(3.2, rel=1e-3)
        if fitted_detector.feature_warning_threshold is not None:
            assert result.top_contributor.deviation_score < fitted_detector.feature_warning_threshold

        # But aggregate RMS: sqrt((16 * 3.2^2) / 25) ≈ 2.56 >= 2.5 (warning_threshold)
        assert result.anomaly_score >= fitted_detector.warning_threshold
        assert result.state == AnomalyState.WARNING

    def test_normal_variation_stays_normal(
        self,
        fitted_detector: ExplainableAnomalyDetector,
    ) -> None:
        """Verify held-out normal windows with random noise trigger neither aggregate nor individual thresholds."""
        normal_windows = make_synthetic_window_sequence(window_count=20, fault_type=None)
        features = extract_features_from_windows(normal_windows)
        results = fitted_detector.predict_batch(features)

        assert len(results) == 20
        for r in results:
            assert r.state == AnomalyState.NORMAL
            assert r.is_anomaly is False
            assert r.anomaly_score < fitted_detector.warning_threshold
            assert r.top_contributor is not None
            if fitted_detector.feature_warning_threshold is not None:
                assert r.top_contributor.deviation_score < fitted_detector.feature_warning_threshold

    def test_zero_mad_with_measurement_noise_not_anomalous(self) -> None:
        """Verify zero-MAD features with realistic sensor measurement noise do not cause false alarms."""
        df = pd.DataFrame(
            {
                "rms_x": [0.5, 0.5, 0.5, 0.5, 0.5],
                "rms_y": [0.20, 0.21, 0.19, 0.20, 0.22],
                "rms_z": [0.98, 0.99, 0.97, 0.98, 0.98],
                "rms_mag": [1.10, 1.11, 1.09, 1.10, 1.12],
            }
        )
        baseline = NormalBaseline.fit(df, feature_names=["rms_x", "rms_y", "rms_z", "rms_mag"])
        detector = ExplainableAnomalyDetector(baseline)

        # 0.5 mg (0.0005 g) measurement noise on constant feature rms_x
        noisy_normal_sample = {
            "rms_x": 0.5005,
            "rms_y": 0.20,
            "rms_z": 0.98,
            "rms_mag": 1.10,
        }
        res_normal = detector.predict(noisy_normal_sample)
        assert res_normal.state == AnomalyState.NORMAL
        assert res_normal.is_anomaly is False
        assert res_normal.anomaly_score < detector.warning_threshold
        assert res_normal.feature_deviations["rms_x"] < 1.0

        # Real mechanical anomaly on rms_x (+20% increase, 0.60 g)
        fault_sample = {
            "rms_x": 0.60,
            "rms_y": 0.20,
            "rms_z": 0.98,
            "rms_mag": 1.10,
        }
        res_fault = detector.predict(fault_sample)
        assert res_fault.state == AnomalyState.ANOMALOUS
        assert res_fault.is_anomaly is True
        assert res_fault.top_contributor is not None
        assert res_fault.top_contributor.feature_name == "rms_x"
        assert res_fault.top_contributor.deviation_score >= 10.0


class TestBaselinePersistence:
    """Tests verifying safe, versioned JSON serialization of baselines."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        windows = make_synthetic_window_sequence(window_count=25, fault_type=None)
        features = extract_features_from_windows(windows)
        original_baseline = NormalBaseline.fit(features, machine_id="Powerloom-Loom-4")

        # Save to JSON
        save_path = tmp_path / "baselines" / "loom4_normal.json"
        saved_file = original_baseline.save(save_path)
        assert saved_file.exists()

        # Load back
        loaded_baseline = NormalBaseline.load(saved_file)
        assert loaded_baseline.machine_id == original_baseline.machine_id
        assert loaded_baseline.sample_count == original_baseline.sample_count
        assert loaded_baseline.version == original_baseline.version
        assert set(loaded_baseline.feature_names) == set(original_baseline.feature_names)

        # Numerical equivalence
        for feat in original_baseline.feature_names:
            orig_stat = original_baseline.feature_stats[feat]
            load_stat = loaded_baseline.feature_stats[feat]
            assert orig_stat.median == pytest.approx(load_stat.median)
            assert orig_stat.mad == pytest.approx(load_stat.mad)
            assert orig_stat.scale == pytest.approx(load_stat.scale)

        # Detector consistency
        detector_orig = ExplainableAnomalyDetector(original_baseline)
        detector_load = ExplainableAnomalyDetector(loaded_baseline)

        test_windows = make_synthetic_window_sequence(window_count=5, fault_type="unbalance")
        test_features = extract_features_from_windows(test_windows)

        res_orig = detector_orig.predict_batch(test_features)
        res_load = detector_load.predict_batch(test_features)

        for ro, rl in zip(res_orig, res_load):
            assert ro.anomaly_score == pytest.approx(rl.anomaly_score)
            assert ro.state == rl.state
            assert ro.top_contributor.feature_name == rl.top_contributor.feature_name

    def test_incompatible_version_rejected(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "bad_version.json"
        bad_json.write_text('{"version": "99.0", "parameters": {}}', encoding="utf-8")
        with pytest.raises(ValueError) as exc_info:
            NormalBaseline.load(bad_json)
        assert "Incompatible baseline version" in str(exc_info.value)

    def test_corrupt_json_rejected(self, tmp_path: Path) -> None:
        corrupt_json = tmp_path / "corrupt.json"
        corrupt_json.write_text("{this is not valid json", encoding="utf-8")
        with pytest.raises(ValueError) as exc_info:
            NormalBaseline.load(corrupt_json)
        assert "Malformed JSON" in str(exc_info.value)
