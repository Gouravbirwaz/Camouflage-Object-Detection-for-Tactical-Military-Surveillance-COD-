"""
Loss functions for DGNet:
- Binary Cross-Entropy Loss (L_BCE): Pixel-level segmentation accuracy.
- Gradient Loss (L_gradient = |G_pred - G_gt|): Boundary supervision loss.
- Total Loss (L_total = L_BCE + lambda * L_gradient).
"""

class DGNetLoss:
    """
    Combined BCE and Gradient supervision loss function for camouflaged target detection.
    """
    def __init__(self, lambda_weight: float = 0.5):
        self.lambda_weight = lambda_weight

    def __call__(self, pred_mask, gt_mask, pred_grad, gt_grad):
        """Calculates combined L_total loss."""
        pass
