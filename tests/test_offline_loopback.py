import torch

from transport.offline_loopback import payload_to_token_map, token_map_to_payload


def test_token_map_payload_roundtrip():
    token_map = torch.arange(16, dtype=torch.long).view(1, 4, 4)

    restored = payload_to_token_map(token_map_to_payload(token_map))

    assert restored.shape == (1, 4, 4)
    assert torch.equal(restored, token_map)


def test_token_map_payload_rejects_bad_length():
    payload = token_map_to_payload(torch.zeros(1, 2, 2, dtype=torch.long))

    try:
        payload_to_token_map(payload[:-1])
    except ValueError as exc:
        assert "length" in str(exc)
    else:
        raise AssertionError("expected invalid payload length")
