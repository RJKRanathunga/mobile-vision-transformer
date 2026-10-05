import torch
from torch import nn


class BNAdjust(nn.Module):
    """
    Separate BatchNorm adjustment for search and template features.

    Search:
        [B, 128, 16, 16]

    Template:
        [B, 128, 8, 8]
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()

        self.bn_search = nn.BatchNorm2d(num_channels)
        self.bn_template = nn.BatchNorm2d(num_channels)

    def forward(
        self,
        search: torch.Tensor,
        template: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        search = self.bn_search(search)
        template = self.bn_template(template)

        return search, template


class PixelCorrelation(nn.Module):
    """
    Pixel-wise correlation between template and search features.

    Template:
        [B, C, Ht, Wt]

    Search:
        [B, C, Hs, Ws]

    Output:
        [B, Ht*Wt, Hs, Ws]
    """

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, channels, search_h, search_w = search.shape

        # [B, C, Ht, Wt]
        # -> [B, Ht*Wt, C]
        template_matrix = template.flatten(2).transpose(1, 2)

        # [B, C, Hs, Ws]
        # -> [B, C, Hs*Ws]
        search_matrix = search.flatten(2)

        # [B, Ht*Wt, C]
        # ×
        # [B, C, Hs*Ws]
        #
        # -> [B, Ht*Wt, Hs*Ws]
        correlation = torch.matmul(
            template_matrix,
            search_matrix,
        )

        # -> [B, Ht*Wt, Hs, Ws]
        correlation = correlation.view(
            batch_size,
            -1,
            search_h,
            search_w,
        )

        return correlation


class ChannelAttention(nn.Module):
    """
    Channel attention used after pixel-wise correlation.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 1,
    ) -> None:
        super().__init__()

        reduced_channels = channels // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(
            channels,
            reduced_channels,
            kernel_size=1,
        )

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(
            reduced_channels,
            channels,
            kernel_size=1,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        attention = self.avg_pool(x)
        attention = self.fc1(attention)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)

        return x * attention


class PointwiseCrossCorrelation(nn.Module):
    """
    Pointwise Correlation + Channel Attention.

    For MVT-Small:

        Template = [B, 128, 8, 8]
        Search   = [B, 128, 16, 16]

    Therefore:

        8 * 8 = 64

    correlation channels are produced.

    Output:

        [B, 64, 16, 16]
    """

    def __init__(
        self,
        num_channels: int,
    ) -> None:
        super().__init__()

        self.pixel_correlation = PixelCorrelation()

        self.channel_attention = ChannelAttention(
            channels=num_channels,
        )

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
    ) -> torch.Tensor:

        correlation = self.pixel_correlation(
            template,
            search,
        )

        correlation = self.channel_attention(
            correlation,
        )

        return correlation


class FeatureFusor(nn.Module):
    """
    Feature fusor used after the MVT backbone.

    Backbone output:

        template: [B, 128, 8, 8]
        search:   [B, 128, 16, 16]

    Correlation:

        [B, 64, 16, 16]

    Final output:

        [B, 256, 16, 16]
    """

    def __init__(
        self,
        input_channels: int = 128,
        correlation_channels: int = 64,
        output_channels: int = 256,
    ) -> None:
        super().__init__()

        self.pwca = PointwiseCrossCorrelation(
            num_channels=correlation_channels,
        )

        self.channel_adjust = nn.Conv2d(
            correlation_channels,
            output_channels,
            kernel_size=1,
        )

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
    ) -> torch.Tensor:

        correlation = self.pwca(
            template,
            search,
        )

        output = self.channel_adjust(
            correlation,
        )

        return output


class MVTNeck(nn.Module):
    """
    MVT neck.

    The official implementation separates the BN adjustment from
    the feature fusor.

    This module performs only the BN adjustment.
    """

    def __init__(
        self,
        num_channels: int = 128,
    ) -> None:
        super().__init__()

        self.bn_adjust = BNAdjust(
            num_channels=num_channels,
        )

    def forward(
        self,
        search: torch.Tensor,
        template: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        return self.bn_adjust(
            search,
            template,
        )