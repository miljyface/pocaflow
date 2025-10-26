import numpy as np
import mathcube as rs
import time
import os

def clear():
    os.system("clear" if os.name != "nt" else "cls")

# cube vertices f64 for precision
# this script has to use f64 for some reason
# probably has to do with the tiny ass 8x3 matrix losing precision
CUBE = np.array([
    [-1, -1, -1],
    [-1, -1,  1],
    [-1,  1, -1],
    [-1,  1,  1],
    [ 1, -1, -1],
    [ 1, -1,  1],
    [ 1,  1, -1],
    [ 1,  1,  1]
], dtype=np.float64) *0.5


# cube edges
EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]

SCREEN_WIDTH = 80
SCREEN_HEIGHT = 30


def rotation_matrix(rx, ry, rz):
    """Build combined rotation matrix (f64)"""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ], dtype=np.float64)
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ], dtype=np.float64)
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    temp = rs.matmul_f64(Ry, Rx)
    rot = rs.matmul_f64(Rz, temp)
    return rot


def perspective_project(point, fov=5.0):
    """Project 3D to 2D"""
    x, y, z = point
    z = max(z, -fov + 0.5)
    f = fov / (fov + z)
    
    screen_x = int(f * x * 20 + SCREEN_WIDTH // 2)
    screen_y = int(f * y * 20 + SCREEN_HEIGHT // 2)
    
    return screen_x, screen_y, z


def bresenham_line(screen, x0, y0, x1, y1):
    """Bresenham line algorithm"""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    while True:
        if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            screen[y][x] = '.'
        
        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def render_cube(transformed_points, screen):
    """Render cube to screen"""
    for a, b in EDGES:
        pt_a = transformed_points[a]
        pt_b = transformed_points[b]
        
        x0, y0, z0 = perspective_project(pt_a)
        x1, y1, z1 = perspective_project(pt_b)
        
        bresenham_line(screen, x0, y0, x1, y1)


def main():
    """Main loop"""
    rx, ry, rz = 0.0, 0.0, 0.0
    frame = 0
    start_time = time.time()
    
    try:
        while True:
            clear()
            
            screen = [[' ' for _ in range(SCREEN_WIDTH)] for _ in range(SCREEN_HEIGHT)]
            
            rot = rotation_matrix(rx, ry, rz)
            transformed = rs.matmul_f64(CUBE, rot)
            
            render_cube(transformed, screen)
            
            for row in screen:
                print(''.join(row))
            
            rx += 0.04
            ry += 0.03
            rz += 0.02
            
            frame += 1
            elapsed = time.time() - start_time
            fps = frame / elapsed if elapsed > 0 else 0
            
            print("\n" + "="*80)
            print(f"GPU-Accelerated Spinning Cube")
            print(f"Frame: {frame} | FPS: {fps:.1f} | Ctrl+C to exit")
            print("="*80)
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n\nDone! Total frames:", frame)

main()
