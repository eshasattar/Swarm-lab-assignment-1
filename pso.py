import random
import numpy as np
import matplotlib.pyplot as plt

# 1. PROBLEM SETUP & UNIQUE SEED (Roll No: 062)
ROLL_NUMBER = 62
random.seed(ROLL_NUMBER)
np.random.seed(ROLL_NUMBER)

GRID_SIZE = 20
NUM_OBSTACLES = 40

# Generate obstacles ensuring start and goal are free
obstacles = set()
while len(obstacles) < NUM_OBSTACLES:
    obs = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    obstacles.add(obs)

# Generate unique Start and Goal points on non-obstacle cells
while True:
    start = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    if start not in obstacles:
        break

while True:
    goal = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    if goal not in obstacles and goal != start:
        break

print(f"Roll Number Seed: {ROLL_NUMBER}")
print(f"Start Point: {start}")
print(f"Goal Point: {goal}")
print(f"Generated Obstacles Count: {len(obstacles)}")

# 2. PSO ALGORITHM FOR PATH PLANNING

class Particle:
    def __init__(self, start, goal, num_waypoints=4):
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.num_waypoints = num_waypoints
        
        # Intermediate waypoints between start and goal
        self.position = []
        for i in range(num_waypoints):
            alpha = (i + 1) / (num_waypoints + 1)
            base_pt = (1 - alpha) * self.start + alpha * self.goal
            # Add minor random offset for diversity
            offset = np.random.uniform(-1.5, 1.5, size=2)
            self.position.append(base_pt + offset)
            
        self.velocity = [np.zeros(2) for _ in range(num_waypoints)]
        self.best_position = list(self.position)
        self.best_cost = float('inf')

def get_full_path(particle):
    path = [tuple(particle.start)]
    for wp in particle.position:
        path.append((int(np.clip(round(wp[0]), 0, GRID_SIZE - 1)), 
                     int(np.clip(round(wp[1]), 0, GRID_SIZE - 1))))
    path.append(tuple(particle.goal))
    return path

def calculate_cost(particle, obstacles):
    path = get_full_path(particle)
    cost = 0.0
    penalty = 0.0
    
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])
        
        # Distance cost
        segment_length = np.linalg.norm(p2 - p1)
        cost += segment_length
        
        # Check obstacle collision along the line segment
        steps = int(max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]), 1))
        for step in range(steps + 1):
            interp_x = int(round(p1[0] + (p2[0] - p1[0]) * (step / steps)))
            interp_y = int(round(p1[1] + (p2[1] - p1[1]) * (step / steps)))
            if (interp_x, interp_y) in obstacles:
                penalty += 50.0  # Heavy collision penalty
                
    return cost + penalty

def run_pso(start, goal, obstacles, iterations=50, num_particles=30):
    particles = [Particle(start, goal) for _ in range(num_particles)]
    global_best_position = None
    global_best_cost = float('inf')
    
    w, c1, c2 = 0.5, 1.5, 1.5
    
    for it in range(iterations):
        for particle in particles:
            cost = calculate_cost(particle, obstacles)
            
            if cost < particle.best_cost:
                particle.best_cost = cost
                particle.best_position = list(particle.position)
                
            if cost < global_best_cost:
                global_best_cost = cost
                global_best_position = list(particle.position)
                
        # Update velocities and positions
        for particle in particles:
            for i in range(particle.num_waypoints):
                r1, r2 = random.random(), random.random()
                cognitive = c1 * r1 * (particle.best_position[i] - particle.position[i])
                social = c2 * r2 * (global_best_position[i] - particle.position[i])
                particle.velocity[i] = w * particle.velocity[i] + cognitive + social
                particle.position[i] += particle.velocity[i]
                
    best_particle = Particle(start, goal)
    best_particle.position = global_best_position
    return get_full_path(best_particle), global_best_cost

final_path, final_cost = run_pso(start, goal, obstacles)
print(f"Optimized Path Found: {final_path}")
print(f"Total Path Cost: {final_cost:.2f}")


# 3. VISUALIZATION
fig, ax = plt.subplots(figsize=(8, 8))
grid_matrix = np.zeros((GRID_SIZE, GRID_SIZE))

for obs in obstacles:
    grid_matrix[obs[0], obs[1]] = 1

ax.imshow(grid_matrix, cmap='Greys', origin='lower')

# Plot path
path_x = [p[1] for p in final_path]
path_y = [p[0] for p in final_path]
ax.plot(path_x, path_y, color='blue', marker='o', linewidth=2.5, label='Optimized Path')

# Plot Start and Goal
ax.scatter(start[1], start[0], color='green', s=150, marker='s', label='Start', zorder=5)
ax.scatter(goal[1], goal[0], color='red', s=150, marker='*', label='Goal', zorder=5)

ax.set_title(f"PSO Path Planning (Seed: {ROLL_NUMBER}) - Cost: {final_cost:.2f}")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.legend(loc='upper right')
ax.grid(True)
plt.show()