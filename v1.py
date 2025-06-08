import numpy as np
import matplotlib.pyplot as plt

def fp8_to_decimal(bits):
    """
    Convert an 8-bit integer to FP8 decimal (1-4-3 format).
    Returns decimal value, handling normalized, denormalized, infinity, and NaN cases.
    """
    # Extract sign, exponent, mantissa
    sign = (bits >> 7) & 1  # 1 bit
    exponent = (bits >> 3) & 0xF  # 4 bits
    mantissa = bits & 0x7  # 3 bits
    
    # Bias for 4-bit exponent
    bias = 7
    
    # Handle special cases
    if exponent == 0xF:  # Exponent all 1s
        if mantissa == 0:
            return float('inf') if sign == 0 else -float('inf')
        else:
            return float('nan')
    
    # Compute value
    if exponent == 0:  # Denormalized
        # No implicit 1, exponent fixed at -6 (bias - 1)
        mantissa_value = mantissa / 8.0  # 2^-3
        value = (-1)**sign * 2.0**(-6) * mantissa_value
    else:  # Normalized
        # Implicit 1, mantissa as fraction
        mantissa_value = 1.0 + mantissa / 8.0  # 1 + 2^-3
        value = (-1)**sign * 2.0**(exponent - bias) * mantissa_value
    
    return value

def plot_fp8_distribution():
    """
    Generate all 8-bit combinations, convert to FP8 decimals, and plot.
    X-axis: Integer value (0-255). Y-axis: FP8 decimal value.
    """
    # Generate all possible 8-bit values
    x_values = np.arange(256)  # 0 to 255
    y_values = np.array([fp8_to_decimal(i) for i in x_values])
    
    # Filter out inf and nan for plotting range
    finite_values = y_values[np.isfinite(y_values)]
    if len(finite_values) > 0:
        y_min, y_max = finite_values.min(), finite_values.max()
    else:
        y_min, y_max = -1, 1  # Fallback range
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, s=10, alpha=0.6, color='blue')
    
    # Plot infinities and NaNs separately (optional, for visibility)
    inf_indices = np.isinf(y_values)
    nan_indices = np.isnan(y_values)
    if inf_indices.any():
        plt.scatter(x_values[inf_indices], np.sign(y_values[inf_indices]) * y_max,
                    s=50, marker='x', color='red', label='Infinity')
    if nan_indices.any():
        plt.scatter(x_values[nan_indices], [0]*nan_indices.sum(),
                    s=50, marker='*', color='green', label='NaN')
    
    # Set labels and title
    plt.xlabel('8-bit Integer Value (0 to 255)')
    plt.ylabel('FP8 Decimal Value')
    plt.title('FP8 (1-4-3) Decimal Values for All 8-bit Combinations')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust y-axis to show finite range clearly
    plt.ylim(y_min * 1.1, y_max * 1.1)  # Add 10% padding
    
    # Add legend if special cases exist
    if inf_indices.any() or nan_indices.any():
        plt.legend()
    
    plt.show()

# Run the plotting function
plot_fp8_distribution()

