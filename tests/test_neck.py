import torch

from mvt.models.neck.mvt_neck import (
    MVTNeck,
    PixelCorrelation,
    PointwiseCrossCorrelation,
    FeatureFusor,
)


def test_pixel_correlation_shape():

    template = torch.randn(
        2, 128, 8, 8
    )

    search = torch.randn(
        2, 128, 16, 16
    )

    correlation = PixelCorrelation()(
        template,
        search,
    )

    assert correlation.shape == (
        2, 64, 16, 16
    )


def test_mvt_neck_shape():

    search = torch.randn(
        2, 128, 16, 16
    )

    template = torch.randn(
        2, 128, 8, 8
    )

    neck = MVTNeck(
        num_channels=128
    )

    output_search, output_template = neck(
        search,
        template,
    )

    assert output_search.shape == search.shape
    assert output_template.shape == template.shape


def test_pwca_shape():

    template = torch.randn(
        2, 128, 8, 8
    )

    search = torch.randn(
        2, 128, 16, 16
    )

    pwca = PointwiseCrossCorrelation(
        num_channels=64
    )

    output = pwca(
        template,
        search,
    )

    assert output.shape == (
        2, 64, 16, 16
    )


def test_feature_fusor_shape():

    template = torch.randn(
        2, 128, 8, 8
    )

    search = torch.randn(
        2, 128, 16, 16
    )

    fusor = FeatureFusor(
        input_channels=128,
        correlation_channels=64,
        output_channels=256,
    )

    output = fusor(
        template,
        search,
    )

    assert output.shape == (
        2, 256, 16, 16
    )