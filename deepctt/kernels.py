from torch import exp, einsum, compile, Tensor


@compile(mode="reduce-overhead", fullgraph=True)
def deep_gsn_kernel_single(
    X: Tensor,
    sigma0: float = 1,
    sigma: float = 1,
    epsilon: float = 0,
    d_embd: int = 1,
) -> Tensor:
    """Computes a combination of the Gsn kernel on the embeddings with
    the Gsn kernel on the original input points.
    First define kappa(x,y) := exp(-||x-y||_2^2/sigma0) to be the Gsn kernel on the embeddings
    Then define q(x,y) := exp(-||x-y||_2^2/sigma) to be the Gsn kernel on the original inputs

    k_{omega}(X,X) := {(1-epsilon) kappa(X[:d_embd],X[:d_embd]) + epsilon} * q(X[d_embd:],X[d_embd:])
                                                ^^^     ^^^                         ^^^     ^^^
                                            embd(X) embd(X)                     orig(X) orig(X)

    Args:
        X: tensor of shape [B,S,E]
        sigma0: positive kernel bandwidth for the embeddings
        sigma: positive kernel bandwidth for the original inputs
        epsilon: weight for the embeddings kernel
        d_embd: number of dimensions of the embeddings

    Warning: there are some numerical instabilities in the computation of the kernel,
        leading sometimes to (small) negative square distances.

    """
    # Compute the squared Euclidean distance between the embeddings
    embd = X[:, :, :d_embd]
    embd_sq = einsum("bse,bse->bs", embd, embd)
    embd_embd = einsum("bse,bte->bst", embd, embd)
    embd1_sq = embd_sq.unsqueeze(embd_sq.dim())
    embd2_sq = embd_sq.unsqueeze(embd_sq.dim() - 1)
    embd_dist_sq_ = embd1_sq - 2 * embd_embd + embd2_sq

    # Compute the squared Euclidean distance between the original
    orig = X[:, :, d_embd:]
    orig_sq = einsum("bse,bse->bs", orig, orig)
    orig_orig = einsum("bse,bte->bst", orig, orig)
    orig1_sq = orig_sq.unsqueeze(orig_sq.dim())
    orig2_sq = orig_sq.unsqueeze(orig_sq.dim() - 1)
    orig_dist_sq_ = orig1_sq - 2 * orig_orig + orig2_sq

    return ((1 - epsilon) * exp(-embd_dist_sq_ / sigma0) + epsilon) * exp(
        -orig_dist_sq_ / sigma
    )


@compile(mode="reduce-overhead")
def square_dist_double(X1: Tensor, X2: Tensor, start: int, end: int) -> Tensor:
    """Assumes X1 and X2 have the same shape except for the second to last dimension"""
    # Extract dimensions [start, end)
    embd1 = X1[..., start:end]
    embd2 = X2[..., start:end]
    # Step 1: Compute embd^2
    embd1_sq = einsum("...se,...se->...s", embd1, embd1)
    embd2_sq = einsum("...te,...te->...t", embd2, embd2)
    # Step 2: Compute embd1 * embd2
    embd1_embd2 = einsum("...se,...te->...st", embd1, embd2)
    # shapes: [..., S, 1], [..., S, T], [..., 1, T]
    embd1_sq = embd1_sq.unsqueeze(embd1_sq.dim())
    embd2_sq = embd2_sq.unsqueeze(embd2_sq.dim() - 1)
    embd_dist_sq = embd1_sq - 2 * embd1_embd2 + embd2_sq
    return embd_dist_sq


@compile(mode="reduce-overhead")
def deep_gsn_kernel_double(
    X1: Tensor,
    X2: Tensor,
    sigma0: float = 1,
    sigma: float = 1,
    epsilon: float = 0,
    d_embd: int = 1,
) -> Tensor:
    """Computes a combination of the Gsn kernel on the embeddings with
    the Gsn kernel on the original input points.
    First define kappa(x,y) := exp(-||x-y||_2^2/sigma0) to be the Gsn kernel on the embeddings
    Then define q(x,y) := exp(-||x-y||_2^2/sigma) to be the Gsn kernel on the original inputs

    k_{omega}(X1,X2) := {(1-epsilon) kappa(X1[:d_embd],X2[:d_embd]) + epsilon} * q(X1[d_embd:],X2[d_embd:])
                                                ^^^     ^^^                         ^^^     ^^^
                                            embd(X1) embd(X2)                     orig(X1) orig(X2)

    Assumes that X1 and X2 have the same shape except for the second to last dimension,
    which is S for X1 and T for X2. The last dimension is E for both X1 and X2.

    Args:
        X1: tensor of shape [..., S, E]
        X2 (optional): tensor of shape [..., T, E]
            if none, then use X2 = X1
        sigma0: positive kernel bandwidth for the embeddings
        sigma: positive kernel bandwidth for the original inputs
        epsilon: weight for the embeddings kernel
        d_embd: number of dimensions of the embeddings

    Warning: there are some numerical instabilities in the computation of the kernel,
        leading sometimes to (small) negative square distances.

    """
    E = X1.shape[-1]
    # Compute the squared Euclidean distance between the embeddings
    embd_dist_sq = square_dist_double(X1, X2, 0, d_embd)
    embd_dist_sq_ = embd_dist_sq / sigma0
    # Compute the squared Euclidean distance between the original
    orig_dist_sq = square_dist_double(X1, X2, d_embd, E)
    orig_dist_sq_ = orig_dist_sq / sigma
    # Compute the combined kernel
    return ((1 - epsilon) * exp(-embd_dist_sq_) + epsilon) * exp(-orig_dist_sq_)
