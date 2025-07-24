import torch
def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)
def to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)
def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 63, m_d & 15 | d >> 2 & 48], dim=-1)
    min = torch.cat([m & 63, m_d >> 4 | m >> 2 & 48], dim=-1)
    return sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8))
def load_grid_tensor(grid_shape, grid_hex):
    grid_bytes = torch.tensor(list(grid_hex))
    grid_words = grid_bytes.view(-1, 2).flip(1)
    grid = grid_words.contiguous().view(-1).to(torch.int16).view(*grid_shape)
    return grid