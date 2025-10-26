import numpy as np
import pocaflow as rs
import time
import os


def clear():
    os.system("clear" if os.name != "nt" else "cls")


WIDTH, HEIGHT = 80, 24


def generate_wave(t, freq_matrix):
    time_vec = np.linspace(0, 2*np.pi, WIDTH).reshape(1, -1).astype(np.float32)
    frequencies = np.array([[freq] for freq in freq_matrix], dtype=np.float32)
    phase_matrix = rs.matmul(frequencies, time_vec)
    wave = np.sum(np.sin(phase_matrix + t), axis=0)
    return wave


def render_waveform(wave):
    screen = [[' ' for _ in range(WIDTH)] for _ in range(HEIGHT)]
    wave_normalized = ((wave - wave.min()) / (wave.max() - wave.min())) * (HEIGHT - 1)
    
    for x, y_val in enumerate(wave_normalized):
        y = int(y_val)
        if 0 <= y < HEIGHT:
            screen[y][x] = '.'
    
    for row in screen:
        print(''.join(row))


def main():
    t = 0.0
    frequencies = [1.0, 2.0, 3.5, 5.0]
    
    try:
        while True:
            clear()
            
            wave = generate_wave(t, frequencies)
            render_waveform(wave)
            
            print("\n" + "="*80)
            print("Waveform Visualizer | Ctrl+C to exit")
            print("="*80)
            
            t += 0.1
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n\nWaveform stopped!")


if __name__ == "__main__":
    main()
