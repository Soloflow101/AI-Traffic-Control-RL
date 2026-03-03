import pygame, yaml, random, math, sys
import numpy as np
from stable_baselines3 import PPO
from envs.traffic_env import TrafficEnv
from simulators.simulator_interface import MockSimulator

# --- CONFIG & COLORS ---
ASPHALT, GRASS, WHITE, RED, GREEN, BLUE = (45, 45, 48), (34, 139, 34), (255, 255, 255), (230, 30, 30), (30, 230, 30), (0, 191, 255)
SIDEWALK, GANTRY = (120, 120, 120), (80, 80, 80)
WIDTH, HEIGHT, CENTER = 800, 800, 400

class Pedestrian:
    def __init__(self, direction):
        self.dir = direction # 'NS' or 'EW'
        self.speed = 1.3
        self.crossed = False
        # SPAWN ON SIDEWALK
        if direction == 'NS': self.x, self.y = CENTER + 80, -20 
        else: self.x, self.y = -20, CENTER + 80
        self.rect = pygame.Rect(self.x, self.y, 8, 8)

    def move(self, phase):
        # NS Peds (Crossing EW road) move when NS light is Green (Phase 0)
        can_go = (phase == 0) if self.dir == 'NS' else (phase == 1)
        at_curb = (self.y > CENTER - 115 and self.y < CENTER - 95) if self.dir == 'NS' else (self.x > CENTER - 115 and self.x < CENTER - 95)
        
        if self.crossed or (can_go and at_curb) or not at_curb:
            if self.dir == 'NS': self.y += self.speed
            else: self.x += self.speed
        if (self.dir == 'NS' and self.y > CENTER - 85) or (self.dir == 'EW' and self.x > CENTER - 85):
            self.crossed = True
        self.rect.center = (self.x, self.y)

class Vehicle:
    def __init__(self, direction):
        self.dir = direction 
        self.speed = 3.2
        self.has_passed_stop = False
        self.turning = random.random() < 0.25
        self.turn_dir = random.choice([-90, 90]) if self.turning else 0
        self.angle = {'N': 270, 'S': 90, 'W': 0, 'E': 180}[direction]
        self.x, self.y = self._spawn()
        self.rect = pygame.Rect(0, 0, 34, 20)

    def _spawn(self):
        # Right-Hand Drive alignment
        pos = {'N': (CENTER+25, -60), 'S': (CENTER-25, HEIGHT+60), 'W': (-60, CENTER+25), 'E': (WIDTH+60, CENTER-25)}
        return pos[self.dir]

    def move(self, is_green, others, peds):
        dist = math.sqrt((self.x-CENTER)**2 + (self.y-CENTER)**2)
        
        # 1. Stop Line Trigger (approx 120px from center)
        at_light = False
        if not is_green and not self.has_passed_stop and 110 < dist < 135:
            at_light = True

        # 2. Advanced Anti-collision (Checking only vehicles AHEAD in same lane)
        blocked = False
        for o in others:
            if o == self or o.dir != self.dir: continue
            gap = math.sqrt((o.x-self.x)**2 + (o.y-self.y)**2)
            # Ensure the car is actually in front of us
            is_ahead = False
            if self.dir == 'N' and o.y > self.y: is_ahead = True
            if self.dir == 'S' and o.y < self.y: is_ahead = True
            if self.dir == 'W' and o.x > self.x: is_ahead = True
            if self.dir == 'E' and o.x < self.x: is_ahead = True
            
            if is_ahead and gap < 55:
                blocked = True
                break

        if not at_light and not blocked:
            if self.turning and not self.has_passed_stop and dist < 15:
                self.angle += self.turn_dir
                self.has_passed_stop = True
            rad = math.radians(self.angle)
            self.x += self.speed * math.cos(rad); self.y -= self.speed * math.sin(rad)
        
        if dist < 80: self.has_passed_stop = True
        self.rect.center = (self.x, self.y)

def run_sim():
    pygame.init(); screen = pygame.display.set_mode((WIDTH, HEIGHT))
    with open("configs/config.yaml", "r") as f: config = yaml.safe_load(f)
    sim = MockSimulator(config); env = TrafficEnv(sim)
    model = PPO.load("models/ppo_traffic_agent.zip")
    vehicles, peds, obs, _ = [], [], *env.reset()
    ai_timer, spawn_timer = 0, 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()

        ai_timer += 1
        if ai_timer > 30:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action); ai_timer = 0
        
        # FINAL INDEX FIX: Use index 6 for 7-D Phase
        phase = int(obs[6]) 
        lights = {'N': phase==0, 'S': phase==0, 'E': phase==1, 'W': phase==1}

        spawn_timer += 1
        if spawn_timer > 50:
            if random.random() < 0.4: vehicles.append(Vehicle(random.choice(['N','S','E','W'])))
            if random.random() < 0.2: peds.append(Pedestrian(random.choice(['NS','EW'])))
            spawn_timer = 0

        screen.fill(GRASS)
        # Background Layers
        pygame.draw.rect(screen, SIDEWALK, (CENTER-95, 0, 190, HEIGHT))
        pygame.draw.rect(screen, SIDEWALK, (0, CENTER-95, WIDTH, 190))
        pygame.draw.rect(screen, ASPHALT, (CENTER-75, 0, 150, HEIGHT))
        pygame.draw.rect(screen, ASPHALT, (0, CENTER-75, WIDTH, 150))
        
        # Zebra Stripes
        for i in range(5):
            pygame.draw.rect(screen, WHITE, (CENTER-65+i*28, CENTER-105, 15, 35))
            pygame.draw.rect(screen, WHITE, (CENTER-65+i*28, CENTER+70, 15, 35))
            pygame.draw.rect(screen, WHITE, (CENTER-105, CENTER-65+i*28, 35, 15))
            pygame.draw.rect(screen, WHITE, (CENTER+70, CENTER-65+i*28, 35, 15))

        # Gantries
        for d, s in lights.items():
            c = GREEN if s else RED
            if d == 'N': pygame.draw.rect(screen, GANTRY, (CENTER-75, CENTER-140, 150, 10))
            if d == 'S': pygame.draw.rect(screen, GANTRY, (CENTER-75, CENTER+130, 150, 10))
            if d == 'E': pygame.draw.rect(screen, GANTRY, (CENTER+130, CENTER-75, 10, 150))
            if d == 'W': pygame.draw.rect(screen, GANTRY, (CENTER-140, CENTER-75, 10, 150))
            pos = {'N':(CENTER+30,CENTER-135), 'S':(CENTER-30,CENTER+135), 'W':(CENTER-135,CENTER-30), 'E':(CENTER+135,CENTER+30)}
            pygame.draw.circle(screen, c, pos[d], 10)

        for v in vehicles[:]:
            v.move(lights[v.dir], vehicles, peds)
            surf = pygame.Surface((34, 20), pygame.SRCALPHA)
            pygame.draw.rect(surf, (200, 50, 50) if v.dir in 'NS' else (50, 50, 200), (0,0,34,20), border_radius=5)
            rot = pygame.transform.rotate(surf, v.angle); screen.blit(rot, rot.get_rect(center=(v.x, v.y)))

        for p in peds[:]:
            p.move(phase); pygame.draw.circle(screen, BLUE, p.rect.center, 7)
            if p.x > 900 or p.y > 900: peds.remove(p)

        pygame.display.flip(); pygame.time.Clock().tick(60)

if __name__ == "__main__":
    run_sim()
