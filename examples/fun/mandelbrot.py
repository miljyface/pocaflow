import numpy as np
import mathcube as rs
import time


def complex_matmul(z_real, z_imag, c_real, c_imag, max_iter=50):
    #Mandelbrot iteration: z = z^2 + c
    #Using matrix ops: z^2 = (a+bi)^2 = (a^2-b^2) + 2abi
    
    for _ in range(max_iter):
        # z^2 real part: a^2 - b^2
        z_real_sq = z_real ** 2 - z_imag ** 2
        # z^2 imag part: 2ab
        z_imag_sq = 2 * z_real * z_imag
        
        # z = z^2 + c
        z_real = z_real_sq + c_real
        z_imag = z_imag_sq + c_imag
        
        # Check divergence
        if np.abs(z_real) > 2 or np.abs(z_imag) > 2:
            return _
    
    return max_iter


def generate_mandelbrot(width=80, height=24, max_iter=50):
    """Generate ASCII Mandelbrot set."""
    chars = " .:-=+*#%@"
    
    # Complex plane bounds
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.0, 1.0
    
    for y in range(height):
        row = []
        for x in range(width):
            # Map to complex plane
            c_real = x_min + (x / width) * (x_max - x_min)
            c_imag = y_min + (y / height) * (y_max - y_min)
            
            # Mandelbrot iteration
            iterations = complex_matmul(0, 0, c_real, c_imag, max_iter)
            
            # Map to ASCII character
            char_idx = int((iterations / max_iter) * (len(chars) - 1))
            row.append(chars[char_idx])
        
        print(''.join(row))
    
    print("\n" + "="*80)
    print("Mandelbrot Set")
    print("="*80)

generate_mandelbrot()
