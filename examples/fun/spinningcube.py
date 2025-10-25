import numpy as np
import rust_linalg as rs
import time
import os
import math

def clear():
    os.system("cls" if os.name == "nt" else "clear")

cube = np.array([
    [-1, -1, -1],
    [-1, -1,  1],
    [-1,  1, -1],
    [-1,  1,  1],
    [ 1, -1, -1],
    [ 1, -1,  1],
    [ 1,  1, -1],
    [ 1,  1,  1]
], dtype=np.float64)

edges = [
    (0,1), (0,2), (0,4),
    (1,3), (1,5),
    (2,3), (2,6),
    (3,7),
    (4,5), (4,6),
    (5,7),
    (6,7)
]

def proj_point(p):
    d = 4
    factor = d/(d + p[2])
    x = int(factor * p[0] * 8 + 40)
    y = int(factor * p[1] * 8 + 12)
    return x, y

def print_cube(points):
    screen = [[' ' for _ in range(80)] for _ in range(24)]
    # edges
    for (a, b) in edges:
        x1, y1 = proj_point(points[a])
        x2, y2 = proj_point(points[b])
        # Bresenham or simple draw
        dx = abs(x2-x1)
        dy = abs(y2-y1)
        steps = max(dx, dy, 1)
        for i in range(steps+1):
            x = int(round(x1 + (x2-x1) * i / steps))
            y = int(round(y1 + (y2-y1) * i / steps))
            if 0 <= x < 80 and 0 <= y < 24:
                screen[y][x] = '#'
    print("\n".join("".join(row) for row in screen))

def get_rot(rx, ry, rz):
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ])
    rot = rs.matmul(Rz, rs.matmul(Ry, Rx))
    return rot

rx, ry, rz = 0.0, 0.0, 0.0
while True:
    clear()
    rot = get_rot(rx, ry, rz)
    pts = rs.matmul(cube, rot)
    print_cube(pts)
    rx += 0.03  # tune for speed
    ry += 0.02
    rz += 0.026
    time.sleep(0.01)
