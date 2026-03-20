import pytest

from framework.pipeline import BenchmarkPipeline


def test_pipeline_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="greater than 0"):
        BenchmarkPipeline(
            model_name="llama3",
            accuracy_weight=0,
            consistency_weight=0,
            efficiency_weight=0,
        )


def test_set_weights_rejects_negative_values():
    pipeline = BenchmarkPipeline(
        model_name="llama3",
        accuracy_weight=0.5,
        consistency_weight=0.3,
        efficiency_weight=0.2,
    )

    with pytest.raises(ValueError, match=">= 0"):
        pipeline.set_weights(accuracy=-0.1)


def test_set_weights_normalizes_valid_values():
    pipeline = BenchmarkPipeline(
        model_name="llama3",
        accuracy_weight=0.5,
        consistency_weight=0.3,
        efficiency_weight=0.2,
    )

    pipeline.set_weights(accuracy=3, consistency=1, efficiency=0)

    assert pipeline.weights["accuracy"] == 0.75
    assert pipeline.weights["consistency"] == 0.25
    assert pipeline.weights["efficiency"] == 0.0
    assert sum(pipeline.weights.values()) == pytest.approx(1.0)
