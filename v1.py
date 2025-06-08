import numpy as np
import matplotlib.pyplot as plt

def fp8_to_decimal(bits):
    sign = (bits >> 7) & 1 
    exponent = (bits >> 3) & 0xF 
    mantissa = bits & 0x7 

    bias = 7
    
    if exponent == 0xF:
        if mantissa == 0:
            return float('inf') if sign == 0 else -float('inf')
        else:
            return float('nan')
    
    if exponent == 0:
        mantissa_value = mantissa / 8.0
        value = (-1)**sign * 2.0**(-6) * mantissa_value
    else:
        mantissa_value = 1.0 + mantissa / 8.0
        value = (-1)**sign * 2.0**(exponent - bias) * mantissa_value
    
    return value

def plot_fp8_distribution():

    x_values = np.arange(256)
    y_values = np.array([fp8_to_decimal(i) for i in x_values])
    
    finite_values = y_values[np.isfinite(y_values)]
    if len(finite_values) > 0:
        y_min, y_max = finite_values.min(), finite_values.max()
    else:
        y_min, y_max = -1, 1
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, s=10, alpha=0.6, color='blue')
    
    inf_indices = np.isinf(y_values)
    nan_indices = np.isnan(y_values)
    if inf_indices.any():
        plt.scatter(x_values[inf_indices], np.sign(y_values[inf_indices]) * y_max,
                    s=50, marker='x', color='red', label='Infinity')
    if nan_indices.any():
        plt.scatter(x_values[nan_indices], [0]*nan_indices.sum(),
                    s=50, marker='*', color='green', label='NaN')
    
    plt.xlabel('8-bit Integer Value (0 to 255)')
    plt.ylabel('FP8 Decimal Value')
    plt.title('FP8 (1-4-3) Decimal Values for All 8-bit Combinations')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.ylim(y_min * 1.1, y_max * 1.1)
    
    if inf_indices.any() or nan_indices.any():
        plt.legend()
    
    plt.savefig('fp8_distribution.png')
    
    plt.show()

plot_fp8_distribution()
