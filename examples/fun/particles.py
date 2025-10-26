import numpy as np
import mathcube as rs
import time
import os


def clear():
    os.system("clear" if os.name != "nt" else "cls")


N_PARTICLES = 100
WIDTH, HEIGHT = 80, 24


class ParticleSystem:
    def __init__(self, n_particles):
        # Position matrix (N x 2)
        self.positions = np.random.rand(n_particles, 2).astype(np.float32)
        self.positions[:, 0] *= WIDTH
        self.positions[:, 1] *= HEIGHT
        
        # Velocity matrix (N x 2)
        self.velocities = (np.random.rand(n_particles, 2) - 0.5).astype(np.float32) * 2
    
    def update(self, dt=0.05):
        """Update particle positions using GPU matmul for forces."""
        # Gravity matrix (N x 2)
        gravity = np.array([[0, 0.1]] * len(self.positions), dtype=np.float32)
        
        # Update velocities: v = v + g*dt
        self.velocities += gravity * dt
        
        # Update positions: p = p + v*dt
        self.positions += self.velocities * dt
        
        # Boundary collisions
        for i in range(len(self.positions)):
            if self.positions[i, 0] < 0 or self.positions[i, 0] >= WIDTH:
                self.velocities[i, 0] *= -0.8
                self.positions[i, 0] = np.clip(self.positions[i, 0], 0, WIDTH-1)
            
            if self.positions[i, 1] < 0 or self.positions[i, 1] >= HEIGHT:
                self.velocities[i, 1] *= -0.8
                self.positions[i, 1] = np.clip(self.positions[i, 1], 0, HEIGHT-1)
    
    def render(self):
        """Render particles to screen."""
        screen = [[' ' for _ in range(WIDTH)] for _ in range(HEIGHT)]
        
        for pos in self.positions:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                screen[y][x] = '.'
        
        for row in screen:
            print(''.join(row))


def main():
    particles = ParticleSystem(N_PARTICLES)
    
    try:
        while True:
            clear()
            particles.update()
            particles.render()
            
            print("\n" + "="*80)
            print("GPU Particle Physics | Ctrl+C to exit")
            print("="*80)
            
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\n\nParticles stopped!")

main()
