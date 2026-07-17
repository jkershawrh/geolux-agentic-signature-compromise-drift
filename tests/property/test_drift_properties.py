from hypothesis import given, settings
from hypothesis import strategies as st

from domain.enums import DriftCategory
from domain.geometry import DriftMeasurement


class TestDriftMeasurementProperties:
    @given(
        geo=st.floats(min_value=0.0, max_value=10.0),
        euc=st.floats(min_value=0.0, max_value=10.0),
        cos=st.floats(min_value=-1.0, max_value=1.0),
        mag=st.floats(min_value=0.0, max_value=1.0),
        prob=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_drift_measurement_all_ranges_valid(self, geo, euc, cos, mag, prob):
        drift = DriftMeasurement(
            agent_id="test",
            baseline_signature_id="s1",
            current_signature_id="s2",
            geodesic_distance=geo,
            euclidean_distance=euc,
            cosine_similarity=cos,
            drift_category=DriftCategory.SEMANTIC,
            drift_magnitude=mag,
            per_dimension_drift={},
            is_significant=False,
            compromise_probability=prob,
        )
        assert drift.geodesic_distance >= 0
        assert drift.euclidean_distance >= 0
        assert -1.0 <= drift.cosine_similarity <= 1.0
        assert 0.0 <= drift.drift_magnitude <= 1.0
        assert 0.0 <= drift.compromise_probability <= 1.0
