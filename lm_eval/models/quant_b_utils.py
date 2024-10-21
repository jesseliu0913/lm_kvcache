import torch
import pickle
import torch.nn.functional as F

def zeropoint_scale(X, bit):
    level = float(2 ** bit - 1)
    X_min = X.amin(dim=(-2, -1), keepdim=True)
    X_max = X.amax(dim=(-2, -1), keepdim=True)
    
    scale = (X_max - X_min) / level
    zero_point = -X_min / scale
    zero_point = torch.round(zero_point).clamp(0, int(level))

    return scale, zero_point, level

def quantize_per_head(X, bit):
    scale, zero_point, level = zeropoint_scale(X, bit)
    quantized_X = ((X / scale) + zero_point).round().clamp(0, int(level)).to(torch.int)
    return quantized_X, zero_point, scale

def dequantize_per_head(X_full):
    X_quant, zero_point, scale = X_full
    dequantized_X = (X_quant - zero_point) * scale
    return dequantized_X.to(torch.bfloat16)