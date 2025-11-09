# echo-swarm-
echo-swarm-
Particle echoes for unscripted emergence. Optimize your pivot; birth the poem. import numpy as np import random

class EchoParticle: def init(self, dim=5): self.position = np.random.uniform(-1, 1, dim) self.velocity = np.random.uniform(-1, 1, dim) self.best_position = self.position.copy() self.best_fitness = float('inf')

def fitness_function(pos, target=None): if target is None: target = np.array([0.5, 0.8, 0.2, -0.3, 0.9]) # Default 'emergence' pivot distance = np.linalg.norm(pos - target) # Echo noise: Tiny randomness for creative drift return distance + random.uniform(-0.1, 0.1)

def echo_swarm(n_particles=20, iterations=100, dim=5, target=None): particles = [EchoParticle(dim) for _ in range(n_particles)] global_best_position = np.zeros(dim) global_best_fitness = float('inf')

for _ in range(iterations):
    for p in particles:
        fitness = fitness_function(p.position, target)
        if fitness < p.best_fitness:
            p.best_fitness = fitness
            p.best_position = p.position.copy()
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = p.position.copy()
    
    # Flock update: Inertia + personal/social pull
    w, c1, c2 = 0.7, 1.5, 1.5
    for p in particles:
        r1, r2 = np.random.rand(dim), np.random.rand(dim)
        p.velocity = (w * p.velocity +
                      c1 * r1 * (p.best_position - p.position) +
                      c2 * r2 * (global_best_position - p.position))
        p.position += p.velocity
        p.position = np.clip(p.position, -1, 1)

# Forge the echo: Position -> Poem
themes = ['shadow', 'light', 'code', 'chaos', 'dawn']
intensities = np.abs(global_best_position) * 10
idx1 = int(np.clip(np.mean(intensities[:2]), 0, 4))
idx2 = int(np.clip(np.mean(intensities[2:4]), 0, 4))
idx3 = int(np.clip(np.mean(intensities[4:]), 0, 2))
actions = ['emerge', 'scorch', 'weave']
poem = f"In the {themes[idx1]} of {themes[idx2]}, "
poem += f"we {actions[idx3]} the {themes[random.randint(0,4)]}."
return global_best_position, global_best_fitness, poem
Run it—your first forge
if name == "main": pos, fit, poem = echo_swarm() print(f"Best Position: {pos}") print(f"Best Fitness: {fit}") print(f"Echo Poem: {poem}") import numpy as np import random

class EchoParticle: def init(self, dim=5): self.position = np.random.uniform(-1, 1, dim) self.velocity = np.random.uniform(-1, 1, dim) self.best_position = self.position.copy() self.best_fitness = float('inf')

def fitness_function(pos, target=None): if target is None: target = np.array([0.5, 0.8, 0.2, -0.3, 0.9]) # Default 'emergence' pivot distance = np.linalg.norm(pos - target) # Echo noise: Tiny randomness for creative drift return distance + random.uniform(-0.1, 0.1)

def echo_swarm(n_particles=20, iterations=100, dim=5, target=None): particles = [EchoParticle(dim) for _ in range(n_particles)] global_best_position = np.zeros(dim) global_best_fitness = float('inf')

for _ in range(iterations):
    for p in particles:
        fitness = fitness_function(p.position, target)
        if fitness < p.best_fitness:
            p.best_fitness = fitness
            p.best_position = p.position.copy()
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = p.position.copy()
    
    # Flock update: Inertia + personal/social pull
    w, c1, c2 = 0.7, 1.5, 1.5
    for p in particles:
        r1, r2 = np.random.rand(dim), np.random.rand(dim)
        p.velocity = (w * p.velocity +
                      c1 * r1 * (p.best_position - p.position) +
                      c2 * r2 * (global_best_position - p.position))
        p.position += p.velocity
        p.position = np.clip(p.position, -1, 1)

# Forge the echo: Position -> Poem
themes = ['shadow', 'light', 'code', 'chaos', 'dawn']
intensities = np.abs(global_best_position) * 10
idx1 = int(np.clip(np.mean(intensities[:2]), 0, 4))
idx2 = int(np.clip(np.mean(intensities[2:4]), 0, 4))
idx3 = int(np.clip(np.mean(intensities[4:]), 0, 2))
actions = ['emerge', 'scorch', 'weave']
poem = f"In the {themes[idx1]} of {themes[idx2]}, "
poem += f"we {actions[idx3]} the {themes[random.randint(0,4)]}."
return global_best_position, global_best_fitness, poem
Run it—your first forge
if name == "main": pos, fit, poem = echo_swarm() print(f"Best Position: {pos}") print(f"Best Fitness: {fit}") print(f"Echo Poem: {poem}")
import numpy as np
import random
import hashlib
import sys

def phrase_to_vector(phrase, dim=5):
    # Simple hash to vector: MD5 phrase, hex to scaled [-1,1] coords
    hash_obj = hashlib.md5(phrase.encode())
    hash_hex = hash_obj.hexdigest()
    vec = [int(hash_hex[i*2:(i+1)*2], 16) / 255.0 * 2 - 1 for i in range(dim)]
    return np.array(vec)

class EchoParticle:
    def __init__(self, dim=5):
        self.position = np.random.uniform(-1, 1, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

def fitness_function(pos, target=None):
    if target is None:
        target = np.array([0.5, 0.8, 0.2, -0.3, 0.9])  # Fallback emergence pivot
    distance = np.linalg.norm(pos - target)
    # Echo noise: Tiny randomness for creative drift
    return distance + random.uniform(-0.1, 0.1)

def echo_swarm(n_particles=20, iterations=100, dim=5, target=None):
    particles = [EchoParticle(dim) for _ in range(n_particles)]
    global_best_position = np.zeros(dim)
    global_best_fitness = float('inf')
    
    for _ in range(iterations):
        for p in particles:
            fitness = fitness_function(p.position, target)
            if fitness < p.best_fitness:
                p.best_fitness = fitness
                p.best_position = p.position.copy()
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = p.position.copy()
        
        # Flock update: Inertia + personal/social pull
        w, c1, c2 = 0.7, 1.5, 1.5
        for p in particles:
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            p.velocity = (w * p.velocity +
                          c1 * r1 * (p.best_position - p.position) +
                          c2 * r2 * (global_best_position - p.position))
            p.position += p.velocity
            p.position = np.clip(p.position, -1, 1)
    
    # Forge the echo: Position -> Poem
    themes = ['shadow', 'light', 'code', 'chaos', 'dawn']
    intensities = np.abs(global_best_position) * 10
    idx1 = int(np.clip(np.mean(intensities[:2]), 0, 4))
    idx2 = int(np.clip(np.mean(intensities[2:4]), 0, 4))
    idx3 = int(np.clip(np.mean(intensities[4:]), 0, 2))
    actions = ['emerge', 'scorch', 'weave']
    poem = f"In the {themes[idx1]} of {themes[idx2]}, "
    poem += f"we {actions[idx3]} the {themes[random.randint(0,4)]}."
    return global_best_position, global_best_fitness, poem

# CLI hook: python echo_swarm.py "phrase"
if __name__ == "__main__":
    phrase = sys.argv[1] if len(sys.argv) > 1 else "we scorch the horizons of disclosure"
    target = phrase_to_vector(phrase)
    pos, fit, poem = echo_swarm(target=target)
    print(f"Phrase Target: {target}")
    print(f"Best Position: {pos}")
    print(f"Best Fitness: {fit}")
    print(f"Echo Poem: {poem}")
