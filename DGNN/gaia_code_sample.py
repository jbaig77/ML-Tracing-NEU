def nonbacktracking_loop_penalty(
    edge_index: torch.Tensor,
    p_edge: torch.Tensor,
    N_nodes: int,
    K: int = 6,
    R: int = 8,
    weighting: str = "sqrt",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Hutchinson estimate of tr( B^3 * sum_{m=0..K} B^{m} ), with B the non-backtracking (Hashimoto) operator.
    This counts **all** non-backtracking closed walks of length >=3 (odd + even), while still
    ignoring immediate edge reversals.

    Args:
      edge_index: (2,E) long
      p_edge:     (E,) float in [0,1] (soft edge predictions)
      N_nodes:    number of nodes, for normalization
      K:          number of extra powers beyond 3 (>=0)  => total powers 3..(3+K)
      R:          Hutchinson probe count
      weighting:  "sqrt" or "child" for B weights (see builder)
    """
    device, dtype = edge_index.device, p_edge.dtype
    E = p_edge.numel()
    if E == 0:
        return torch.zeros((), device=device, dtype=dtype)

    # Build B (2E x 2E) sparse
    B = _build_hashimoto_nonbacktracking(edge_index, p_edge, N=N_nodes, weighting=weighting).coalesce()
    D = B.size(0)  # 2E

    if D == 0 or B._nnz() == 0:
        return torch.zeros((), device=device, dtype=dtype)

    # --- Stability: scale B so spectral radius < 1 (cheap bound via max row L1 norm) ---
    row_l1 = _row_abs_sums_sparse(B)
    max_l1 = torch.maximum(row_l1.max(), torch.tensor(1.0, device=device, dtype=dtype))
    scale = (0.9 / max_l1).clamp(max=1.0)  # keep <=1 so we never inflate
    B = torch.sparse_coo_tensor(B.indices(), B.values() * scale, size=B.size(), device=device, dtype=dtype).coalesce()

    acc = p_edge.new_zeros(())
    for _ in range(int(R)):
        # Rademacher probe
        z = torch.empty((D,), device=device, dtype=dtype).bernoulli_().mul_(2).add_(-1)

        # y3 = B^3 z
        y1 = _sparse_mm(B, z)
        y2 = _sparse_mm(B, y1)
        yk = _sparse_mm(B, y2)  # this is B^3 z

        # accumulate z^T (B^3 + B^4 + ... + B^{3+K}) z
        term = (z * yk).sum()
        for _m in range(int(K)):
            yk = _sparse_mm(B, yk)  # multiply by B once each time -> includes ALL lengths
            term = term + (z * yk).sum()

        acc = acc + term

    val = acc / float(R)

    # normalization (similar scale as other penalties)
    with torch.no_grad():
        E_est = p_edge.sum()  # soft total edge mass
        denom = (N_nodes + 0.5 * E_est).clamp_min(1.0)
    return val / denom
