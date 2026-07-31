import torch


_SUPPORTED_TOKEN_DTYPES = {torch.long, torch.uint16}


def sample_next_token_batch(
    token_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be a 1D tensor")

    if token_ids.dtype not in _SUPPORTED_TOKEN_DTYPES:
        raise ValueError("token_ids must have dtype torch.long or torch.uint16")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    if context_length <= 0:
        raise ValueError("context_length must be greater than 0")

    if len(token_ids) <= context_length:
        raise ValueError("token_ids must be longer than context_length")

    start_indices = torch.randint(
        low=0,
        high=len(token_ids) - context_length,
        size=(batch_size,),
    )

    x_batch = torch.stack([
        token_ids[start.item():start.item() + context_length]
        for start in start_indices
    ])

    y_batch = torch.stack([
        token_ids[start.item() + 1:start.item() + context_length + 1]
        for start in start_indices
    ])

    # Embedding lookup requires torch.long IDs. Keeping mapped streams uint16
    # until this point avoids materializing an entire large corpus as int64.
    x_batch = x_batch.to(device=device, dtype=torch.long)
    y_batch = y_batch.to(device=device, dtype=torch.long)

    return x_batch, y_batch
