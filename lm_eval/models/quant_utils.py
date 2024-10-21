import torch
import pickle
import torch.nn.functional as F


# def cal_cosine(X):
#     batch_size, num_heads, seq_len, head_dim = X.size()
#     norm_X = F.normalize(X, p=2, dim=-1)
#     norm_X = norm_X.reshape(-1, seq_len, head_dim)
#     cosine_sim = torch.bmm(norm_X, norm_X.transpose(1, 2))
#     cosine_sim = cosine_sim.view(batch_size, num_heads, seq_len, seq_len)
#     cosine_sim = cosine_sim.mean(dim=(-1, -2))
#     return cosine_sim


# def cal_outliers(matrix, percent=1):
#     matrix_fp32 = matrix.to(torch.float32)
    
#     batch_size, num_heads, seq_length, hidden_size = matrix_fp32.shape

#     medians = matrix_fp32.median(dim=-1, keepdim=True).values
#     medians = medians.median(dim=-2, keepdim=True).values
#     abs_deviation = torch.abs(matrix_fp32 - medians)
#     mad = abs_deviation.median(dim=-1, keepdim=True).values
#     mad = mad.median(dim=-2, keepdim=True).values

#     num_elements = seq_length * hidden_size
#     num_outliers = max(1, int(num_elements * (percent / 100)))

#     flattened_deviation = abs_deviation.reshape(batch_size, num_heads, -1)
#     sorted_deviation, _ = torch.sort(flattened_deviation, dim=-1, descending=True)
#     deviation_thresh = sorted_deviation[:, :, num_outliers - 1:num_outliers].view(batch_size, num_heads, 1, 1)

#     outliers = abs_deviation >= deviation_thresh

#     outlier_values = matrix[outliers].clone()
#     outlier_indices = outliers.nonzero(as_tuple=True)

#     matrix[outliers] = medians.expand_as(matrix_fp32)[outliers].to(matrix.dtype)
    
#     return matrix, outlier_values, outlier_indices

# def revert_outliers(matrix, outlier_values, outlier_indices):
#     matrix[outlier_indices] = outlier_values
#     return matrix

# def zeropoint_scale(X, bit):
#     level = float(2 ** bit - 1)
#     X_min = X.amin(dim=(-2, -1), keepdim=True)
#     X_max = X.amax(dim=(-2, -1), keepdim=True)
    
#     scale = (X_max - X_min) / level
#     zero_point = -X_min / scale
#     zero_point = torch.round(zero_point).clamp(0, int(level))

#     return scale, zero_point, level

# def quantize_per_head(X, bit):
#     X_remove = X.clone()
#     X_new, outlier_values, outlier_indices = cal_outliers(X_remove)
#     scale, zero_point, level = zeropoint_scale(X_new, bit)
#     quantized_X = ((X_new / scale) + zero_point).round().clamp(0, int(level)).to(torch.int)
#     return quantized_X, zero_point, scale, outlier_values, outlier_indices

# def dequantize_per_head(X_full):
#     X_quant, zero_point, scale, outlier_values, outlier_indices = X_full
#     dequantized_X = (X_quant - zero_point) * scale
#     dequantized_X = revert_outliers(dequantized_X, outlier_values, outlier_indices)
#     return dequantized_X.to(torch.float16)


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
    return dequantized_X.to(torch.float16)

