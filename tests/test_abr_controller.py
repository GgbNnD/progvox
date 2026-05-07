from models.abr_controller import NetworkObservation, RuleBasedABRConfig, RuleBasedABRController, observations_from_trace
from transport.scheduler import TracePoint, default_video_call_layers


def test_rule_based_abr_downshifts_on_loss_and_delay():
    layers = default_video_call_layers()
    controller = RuleBasedABRController(layers, RuleBasedABRConfig(fps=30.0, stable_upshift_windows=1))

    high = NetworkObservation(timestamp_ms=0.0, throughput_kbps=700.0, rtt_ms=60.0, loss_rate=0.0)
    first = controller.decide(0, high)
    congested = NetworkObservation(timestamp_ms=33.3, throughput_kbps=90.0, rtt_ms=220.0, loss_rate=0.06)
    second = controller.decide(1, congested)

    assert first.selected_level >= 2
    assert second.selected_level < first.selected_level
    assert "loss" in second.reason or "delay" in second.reason


def test_rule_based_abr_hysteresis_holds_single_upshift():
    layers = default_video_call_layers()
    controller = RuleBasedABRController(layers, RuleBasedABRConfig(fps=30.0, stable_upshift_windows=3))

    first = controller.decide(0, NetworkObservation(0.0, 700.0, 60.0, 0.0))
    second = controller.decide(1, NetworkObservation(33.3, 700.0, 60.0, 0.0))

    assert first.selected_level == 0
    assert second.selected_level == 0
    assert "hold_upshift" in second.reason


def test_observations_from_trace_derives_delay_from_weak_bandwidth():
    trace = [
        TracePoint(bandwidth_kbps=500.0, duration_ms=33.3, loss_rate=0.01),
        TracePoint(bandwidth_kbps=3.0, duration_ms=33.3, loss_rate=0.05),
    ]
    observations = observations_from_trace(trace, fps=30.0)

    assert observations[0].rtt_ms < observations[1].rtt_ms
    assert observations[1].queue_delay_ms > 0.0
