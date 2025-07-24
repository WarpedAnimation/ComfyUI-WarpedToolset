import torch
from .quant import dequantize as gq
from .reader import GGMLQuantizationType, GGML_QUANT_SIZES
from tqdm import tqdm
TORCH_COMPATIBLE_QTYPES = {None, GGMLQuantizationType.F32,
    GGMLQuantizationType.F16}
def is_torch_compatible(tensor):
    return tensor is None or getattr(tensor, 'tensor_type', None
        ) in TORCH_COMPATIBLE_QTYPES
def is_quantized(tensor):
    return not is_torch_compatible(tensor)
def dequantize(data, qtype, oshape, dtype=None):
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)
def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    qtype = getattr(tensor, 'tensor_type', None)
    oshape = getattr(tensor, 'tensor_shape', tensor.shape)
    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)
    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == 'target' else dequant_dtype
        return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(
            dtype)
    else:
        tqdm.write(f'Processing tensor: qtype: {qtype}, {oshape}')
        new = gq(tensor.cpu().numpy(), qtype)
        return torch.from_numpy(new).to(tensor.device, dtype=dtype)
def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)
def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d = blocks[:, :2].view(torch.float16).to(dtype)
    x = blocks[:, 2:].view(torch.int8)
    return x * d
def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d = blocks[:, :2].view(torch.float16).to(dtype)
    qs = blocks[:, 2:]
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0,
        4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 15).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs
def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d = blocks[:, :2].view(torch.float16).to(dtype)
    m = blocks[:, 2:4].view(torch.float16).to(dtype)
    qs = blocks[:, 4:]
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0,
        4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 15).reshape(n_blocks, -1)
    return d * qs + m
from .quant2b import to_uint32
def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d = blocks[:, :2].view(torch.float16).to(dtype)
    qh = blocks[:, 2:6]
    qs = blocks[:, 6:]
    qh = to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype
        =torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4
        ], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 15).reshape(n_blocks, -1)
    qs = (ql | qh << 4).to(torch.int8) - 16
    return d * qs
def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d = blocks[:, :2].view(torch.float16).to(dtype)
    m = blocks[:, 2:4].view(torch.float16).to(dtype)
    qh = blocks[:, 4:8]
    qs = blocks[:, 8:]
    qh = to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device,
        dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0,
        4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 15).reshape((n_blocks, -1))
    qs = ql | qh << 4
    return d * qs + m
QK_K = 256
from .quant2b import split_block_dims
def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K //
        16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d
        .device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 15).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6],
        device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 3).reshape((n_blocks, -1, 32))
    q = (ql | qh << 4).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))
def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4],
        device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6
        ], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = lscales & 15 | (hscales & 3) << 4
    scales = scales.to(torch.int8) - 32
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6],
        device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in
        range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = qh.reshape((n_blocks, 16, QK_K // 16)) & 1 ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)
    return (dl * q).reshape((n_blocks, QK_K))
def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 15)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8
        ).reshape((1, 1, 4, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> shift & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))
K_SCALE_SIZE = 12
from .quant2b import get_scale_min
def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, 
        QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d
        .device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in
        range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 15).reshape((n_blocks, -1, 32))
    qh = (qh & 1).reshape((n_blocks, -1, 32))
    q = ql | qh << 4
    return (d * q - dm).reshape((n_blocks, QK_K))
def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d
        .device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 15).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))
def dequantize_blocks_TQ2_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    qs, d = split_block_dims(blocks, QK_K // 4)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6],
        device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs & 3).reshape((n_blocks, -1)) - 1
    return d * qs
def dequantize_blocks_TQ1_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    qs, qh, d = split_block_dims(blocks, (QK_K - 4 * QK_K // 64) // 5, QK_K //
        64)
    d = d.view(torch.float16).to(dtype)
    qs0, qs1 = qs[..., :32], qs[..., 32:]
    qs0 = qs0.reshape((n_blocks, -1, 1, 32)) * torch.tensor([1, 3, 9, 27, 
        81], device=d.device, dtype=torch.uint8).reshape((1, 1, 5, 1))
    qs0 = qs0.reshape((n_blocks, -1))
    qs1 = qs1.reshape((n_blocks, -1, 1, 16)) * torch.tensor([1, 3, 9, 27, 
        81], device=d.device, dtype=torch.uint8).reshape((1, 1, 5, 1))
    qs1 = qs1.reshape((n_blocks, -1))
    qh = qh.reshape((n_blocks, -1, 1, 4)) * torch.tensor([1, 3, 9, 27],
        device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = qh.reshape((n_blocks, -1))
    qs = torch.cat([qs0, qs1, qh], dim=-1)
    qs = (qs * 3 >> 8) - 1
    return d * qs
def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    kvalues = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13,
        25, 38, 53, 69, 89, 113], dtype=torch.float32, device=blocks.device)
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0,
        4], device=blocks.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 15).reshape((n_blocks, -1)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], 16), 2, qs)
    qs = qs.squeeze(-1).to(dtype)
    return d * qs
def dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    kvalues = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13,
        25, 38, 53, 69, 89, 113], dtype=torch.float32, device=blocks.device)
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = scales_h.view(torch.int16)
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> torch.tensor([0, 4],
        device=blocks.device, dtype=torch.uint8).reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, 1, -1)) >> torch.tensor([(2 * i) for
        i in range(QK_K // 32)], device=blocks.device, dtype=torch.uint8
        ).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & 15
    scales_h = scales_h.reshape((n_blocks, -1)) & 3
    scales = (scales_l | scales_h << 4) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))
    shifts_q = torch.tensor([0, 4], device=blocks.device, dtype=torch.uint8
        ).reshape(1, 1, 2, 1)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shifts_q
    qs = (qs & 15).reshape((n_blocks, -1, 32)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], qs.shape[2],
        16), 3, qs)
    qs = qs.squeeze(-1).to(dtype)
    return (dl * qs).reshape(n_blocks, -1)
from .quant2b import load_grid_tensor
def dequantize_blocks_IQ3_S(blocks, block_size, type_size, dtype=None):
    grid_shape = 512, 4
    grid_hex = (
        b'00000100020005000700100011001200140016002000210025003300400042004500470051005300600062007100740077000001010102010401100111011501200123012701310135014401610165017201000201020502070210021302160221022502300234024202450247025102530270027302030311031503200322033103330336034403500352036703710375030004130417042104240432044004430451047004020504052005220526053305410545054705660573050606110613063106520671060007020704072007220726073307500754070010011002100410101011101310151017102010221031103410361054105610611072100011011103110611101114112111301133114111501152117011761100121212151217122012241232124012431255126012721201130413071310131313211327133013341341136213701303140514121414143114331442144614501454140115101513152115301532155115201624162716441646160117031710171217211735174117621770170020012003200520072010201220142016202120232027203020322041204320452050205220672070207320752000210221102113211721222125213121342142215121012204220722212223223022372241225322572271227422002302230523112322232423312333234223502366230124072420242324322435244124722475240425112522253725402553257025002602260726212655266126052711272627302743275027023011301330153017302230313033303530423044304730513063307130013103310531143121312331403160317231763100321232203232323432503201331033143321332333273330334133433347335533733303341134163422343134523460346434013510351235253532354435563573351636413601370337203722373537004004401240204024402740324041405040704002410741114113412241304135414341514155410142034210421542214233424042574262427042044311431343204322433143354300440244244437444044714405450745214562451346344660461047154730474347514702501050145022504050445047505250665074500151035105511251215132517251005211522352305236525352025307531053275344535153655373530154045420543254465412552655515553554256025704572257116013601560316033606060006120612761646112623462426255626262706200631463216340632564436462640065036534656065056640661167136700700470077020702270367040705470627002711171247143714571017204721072167221723072517202733273357353730174057413742074507422754275027631760077'
        )
    n_blocks = blocks.shape[0]
    d, qs, qh, signs, scales = split_block_dims(blocks, 2, QK_K // 4, QK_K //
        32, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    scales = scales.reshape((n_blocks, -1, 1)) >> torch.tensor([0, 4],
        device=blocks.device, dtype=torch.uint8).reshape((1, 1, 2))
    scales = (scales & 15).reshape((n_blocks, -1))
    db = d * (1 + 2 * scales.to(dtype))
    db = db.reshape((n_blocks, -1, 1, 1))
    signs = signs.reshape((n_blocks, -1, 1)) >> torch.arange(8, device=
        blocks.device, dtype=torch.uint8).reshape((1, 1, 8))
    signs = signs & 1
    signs = signs.reshape((n_blocks, -1, 4, 8))
    qh_shifts = torch.arange(8, device=blocks.device, dtype=torch.uint8).view(
        1, 1, 8)
    qh = qh.reshape((n_blocks, -1, 1)) >> qh_shifts & 1
    qh = qh.reshape((n_blocks, -1)).to(torch.int16)
    qs = qs.to(torch.int64) | qh << 8
    grid = load_grid_tensor(grid_shape, grid_hex)
    grid = grid.unsqueeze(0).expand(n_blocks, -1, -1)
    return (db * signs).reshape(n_blocks, -1)
def dequantize_blocks_IQ3_XXS(blocks, block_size, type_size, dtype=None):
    grid_shape = 256, 4
    grid_hex = (
        b'0000020004001100130017002000220031004200730075000101030110011201210125013001320141015401700100020202040211022002220231023302370251025702750201030703100312032503700313043704440457047304750401050705320552053506640610071407160743076107011003101010121021102310301032103410471050100011021111112011221101120312101212122112301272120013021320133113461366130114051450142015241546157115051622174017002002201120132020202220262031204220012103210521102112212121302163216721702100220222112217222022222237224022552201231023142370237423352453240325272541257425012703271627452701301030123021302330503065307230003102312031313144314631013203321032253252327232113333333034473472340035063522355535143636366336333760370440174035403740534057407441204237424042604266420743454304445144644425454345704505471047124730471250415070500051065126515551145232527252025353531054235427547254025531555056245742572460446046606460216161611762646230633663446405655265336603672167037005700770107032705270267140711272457252720073157333736073217441740075027524753076'
        )
    n_blocks = blocks.shape[0]
    d, qs, scales = split_block_dims(blocks, 2, QK_K // 4)
    d = d.view(torch.float16).to(dtype)
    scales = to_uint32(scales)
    db = d * (0.5 + (scales >> 28)) * 0.5
    db = db.reshape((n_blocks, -1, 1, 1))
    bit_shifts = torch.tensor([0, 7, 14, 21], device=blocks.device, dtype=
        torch.uint8).reshape((1, 1, 4))
    signs = scales.reshape((n_blocks, -1, 1)) >> bit_shifts
    db = db.reshape((n_blocks, -1, 1, 1))
    signs = signs.reshape((n_blocks, -1, 1)) >> torch.tensor([i for i in
        range(8)], device=blocks.device, dtype=torch.uint8).reshape((1, 1, 8))
    signs = signs & 1
    db = db.reshape((n_blocks, -1, 1, 1))
    sign_shifts = torch.arange(8, device=blocks.device, dtype=torch.uint8
        ).view(1, 1, 8)
    signs = signs.reshape((n_blocks, -1, 1)) >> sign_shifts
    signs = (signs & 1).float()
    signs = torch.where(signs == 0, torch.tensor(1.0, device=blocks.device),
        torch.tensor(-1.0, device=blocks.device))
    signs = signs.reshape(n_blocks, -1, 4, 8)
    qs = qs.reshape(n_blocks, -1, 1, 1)
    grid = load_grid_tensor(grid_shape, grid_hex)
    grid = grid.unsqueeze(0).expand(n_blocks, -1, -1)
    return (db * signs).reshape(n_blocks, -1)
dequantize_functions = {GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0, GGMLQuantizationType
    .Q5_1: dequantize_blocks_Q5_1, GGMLQuantizationType.Q5_0:
    dequantize_blocks_Q5_0, GGMLQuantizationType.Q4_1:
    dequantize_blocks_Q4_1, GGMLQuantizationType.Q4_0:
    dequantize_blocks_Q4_0, GGMLQuantizationType.Q6_K:
    dequantize_blocks_Q6_K, GGMLQuantizationType.Q5_K:
    dequantize_blocks_Q5_K, GGMLQuantizationType.Q4_K:
    dequantize_blocks_Q4_K, GGMLQuantizationType.Q3_K:
    dequantize_blocks_Q3_K, GGMLQuantizationType.Q2_K:
    dequantize_blocks_Q2_K, GGMLQuantizationType.TQ2_0:
    dequantize_blocks_TQ2_0, GGMLQuantizationType.TQ1_0:
    dequantize_blocks_TQ1_0, GGMLQuantizationType.IQ4_NL:
    dequantize_blocks_IQ4_NL, GGMLQuantizationType.IQ4_XS:
    dequantize_blocks_IQ4_XS, GGMLQuantizationType.IQ3_S:
    dequantize_blocks_IQ3_S, GGMLQuantizationType.IQ3_XXS:
    dequantize_blocks_IQ3_XXS}