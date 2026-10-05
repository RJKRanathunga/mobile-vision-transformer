import torch
from torch import nn


class FocalLoss(nn.Module):
    """
    CenterNet-style focal loss used by the official MVT implementation.

    Args:
        alpha: Focusing parameter for prediction confidence.
        beta: Weighting parameter for negative locations.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prediction: Predicted heatmap probabilities.
                        Shape [B, 1, H, W].
                        Values should be in (0, 1).

            target: Ground-truth Gaussian heatmap.
                    Shape [B, 1, H, W].
                    Values in [0, 1].

        Returns:
            Scalar focal loss.
        """

        positive_index = target.gt(0.1).float()
        negative_index = target.lt(0.1).float()

        negative_weights = torch.pow(
            1.0 - target,
            self.beta,
        )

        # Prevent log(0).
        prediction = torch.clamp(
            prediction,
            min=1e-12,
            max=1.0 - 1e-12,
        )

        positive_loss = (
            torch.log(prediction)
            * torch.pow(1.0 - prediction, self.alpha)
            * positive_index
        )

        negative_loss = (
            torch.log(1.0 - prediction)
            * torch.pow(prediction, self.alpha)
            * negative_weights
            * negative_index
        )

        num_positive = positive_index.sum()

        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss = -negative_loss
        else:
            loss = -(positive_loss + negative_loss) / num_positive

        return loss