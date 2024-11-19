import torch
import torch.nn as nn

class CrossEntropyMasked(nn.Module):
    def __init__(self):
        """
        Custom masked loss class for sequence-to-sequence models.
        """
        super(CrossEntropyMasked, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')  # Element-wise loss

    def forward(self, logits, targets, mask):
        """
        Compute masked loss.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, seq_len, vocab_size)
            targets (torch.Tensor): Target sequences of shape (batch_size, seq_len)
            mask (torch.Tensor): Binary mask of shape (batch_size, seq_len), where 1 indicates valid tokens.

        Returns:
            torch.Tensor: Scalar loss normalized by the number of valid tokens.
        """
        # Flatten the tensors for loss computation
        logits_flat = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
        targets_flat = targets.view(-1)                # (batch_size * seq_len)
        mask_flat = mask.view(-1)                      # (batch_size * seq_len)

        # Compute element-wise loss
        loss = self.loss_fn(logits_flat, targets_flat)  # (batch_size * seq_len)

        # Apply the mask to zero out padding tokens
        loss = loss * mask_flat  # Zero out loss for padding tokens

        # Total loss sum over all the non-masked entries in output over the batch
        return loss.sum()
