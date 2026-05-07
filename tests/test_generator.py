import torch

from models.generator import (
    DetailSynthesisCNN,
    PatchDiscriminator,
    TinyConditionalDiffusion,
    build_generator,
)


def test_cnn_generator_preserves_image_shape_and_range():
    condition = torch.rand(2, 3, 64, 64)
    model = DetailSynthesisCNN()

    output = model(condition)

    assert output.shape == condition.shape
    assert output.min() >= 0
    assert output.max() <= 1


def test_patch_discriminator_returns_patch_logits():
    condition = torch.rand(2, 3, 64, 64)
    image = torch.rand(2, 3, 64, 64)
    discriminator = PatchDiscriminator()

    logits = discriminator(condition, image)

    assert logits.shape[0] == 2
    assert logits.shape[1] == 1
    assert logits.shape[-1] < image.shape[-1]


def test_diffusion_loss_and_short_sampler_are_valid():
    condition = torch.rand(2, 3, 32, 32)
    target = torch.rand(2, 3, 32, 32)
    model = TinyConditionalDiffusion()

    loss = model.training_loss(condition, target)
    sample = model.sample(condition, steps=2)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert sample.shape == condition.shape
    assert sample.min() >= 0
    assert sample.max() <= 1


def test_generator_factory_supports_required_variants():
    assert isinstance(build_generator("pure_cnn"), DetailSynthesisCNN)
    assert isinstance(build_generator("cnn_gan"), DetailSynthesisCNN)
    assert isinstance(build_generator("tiny_diffusion"), TinyConditionalDiffusion)
