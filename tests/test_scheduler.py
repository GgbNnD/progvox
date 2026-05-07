from transport.scheduler import (
    GreedyScheduler,
    SlidingWindowScheduler,
    TokenLayerSpec,
    TracePoint,
    default_video_call_layers,
    make_bursty_trace,
    simulate,
)


def test_default_video_call_layers_match_tokenizer_geometry():
    layers = default_video_call_layers()

    assert [layer.bits for layer in layers] == [144, 576, 2304, 9216]
    assert [layer.level for layer in layers] == [0, 1, 2, 3]
    assert layers[0].quality_psnr < layers[-1].quality_psnr


def test_sliding_window_reduces_base_layer_starvation():
    layers = [
        TokenLayerSpec(level=0, bits=200, quality_psnr=10.0),
        TokenLayerSpec(level=1, bits=2000, quality_psnr=20.0),
    ]
    trace = [TracePoint(bandwidth_kbps=1.0, duration_ms=100.0) for _ in range(12)]

    greedy = simulate(GreedyScheduler(), trace, layers, num_frames=6, fps=10.0, playback_delay_ms=400.0)
    sliding = simulate(SlidingWindowScheduler(window_frames=5), trace, layers, num_frames=6, fps=10.0, playback_delay_ms=400.0)

    assert greedy.summary()["stall_rate"] > sliding.summary()["stall_rate"]
    assert sliding.frames[1].max_layer == 0


def test_simulation_summary_contains_transport_metrics():
    layers = default_video_call_layers()
    trace = make_bursty_trace(slots=12, duration_ms=33.33, seed=3)
    result = simulate(SlidingWindowScheduler(), trace, layers, num_frames=6, fps=30.0, playback_delay_ms=100.0)
    summary = result.summary()

    assert summary["policy"] == "sliding_window"
    assert summary["frames"] == 6
    assert 0.0 <= summary["stall_rate"] <= 1.0
    assert summary["sent_bitrate_kbps"] > 0.0
    assert -1 in summary["layer_counts"]
