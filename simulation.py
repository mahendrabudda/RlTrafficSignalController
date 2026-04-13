import random
import time
import threading
import sys

import pygame

from rl_bridge import RLBridge


# ── signal timing constants ───────────────────────────────────────────────────
MIN_GREEN_TIME  = 5
MAX_GREEN_TIME  = 30
DEFAULT_YELLOW  = 3
CHECK_INTERVAL  = 1          # seconds between feedback() calls inside a green phase

# ── global state ──────────────────────────────────────────────────────────────
signals       = []
noOfSignals   = 4
currentGreen  = 0
currentYellow = 0

speeds = {"car": 2.25, "bus": 1.8, "truck": 1.8, "bike": 2.5}

x = {
    "right": [0, 0, 0],
    "down":  [755, 727, 697],
    "left":  [1400, 1400, 1400],
    "up":    [602, 627, 657],
}
y = {
    "right": [348, 370, 398],
    "down":  [0, 0, 0],
    "left":  [498, 466, 436],
    "up":    [800, 800, 800],
}

vehicles = {
    "right": {0: [], 1: [], 2: [], "crossed": 0},
    "down":  {0: [], 1: [], 2: [], "crossed": 0},
    "left":  {0: [], 1: [], 2: [], "crossed": 0},
    "up":    {0: [], 1: [], 2: [], "crossed": 0},
}

vehicleTypes     = {0: "car", 1: "bus", 2: "truck", 3: "bike"}
directionNumbers = {0: "right", 1: "down", 2: "left", 3: "up"}

vehicleCrossedCount = {"car": 0, "bus": 0, "truck": 0, "bike": 0}
totalCrossed        = 0
waitTime            = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

signalCoods      = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]
stopLines        = {"right": 590, "down": 330, "left": 800,  "up": 535}
defaultStop      = {"right": 580, "down": 320, "left": 810,  "up": 545}
stoppingGap      = 15
movingGap        = 15

pygame.init()
simulation = pygame.sprite.Group()

bridge = RLBridge(
    qtable_path="qtable.json",
    online_learning=True,
    epsilon=1,
    save_every=500,
)


# ── classes ───────────────────────────────────────────────────────────────────

class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red        = red
        self.yellow     = yellow
        self.green      = green
        self.signalText = ""


class Vehicle(pygame.sprite.Sprite):

    def __init__(self, lane, vehicleClass, direction_number, direction):
        pygame.sprite.Sprite.__init__(self)
        self.lane             = lane
        self.vehicleClass     = vehicleClass
        self.speed            = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction        = direction
        self.x                = x[direction][lane]
        self.y                = y[direction][lane]
        self.crossed          = 0
        vehicles[direction][lane].append(self)
        self.index            = len(vehicles[direction][lane]) - 1
        path                  = "images/" + direction + "/" + vehicleClass + ".png"
        self.image            = pygame.image.load(path)

        prev = vehicles[direction][lane]
        if len(prev) > 1 and prev[self.index - 1].crossed == 0:
            p = prev[self.index - 1]
            if direction == "right":
                self.stop = p.stop - p.image.get_rect().width  - stoppingGap
            elif direction == "left":
                self.stop = p.stop + p.image.get_rect().width  + stoppingGap
            elif direction == "down":
                self.stop = p.stop - p.image.get_rect().height - stoppingGap
            elif direction == "up":
                self.stop = p.stop + p.image.get_rect().height + stoppingGap
        else:
            self.stop = defaultStop[direction]

        # ── FIX: clamp self.stop so it never goes past the default stop line ──
        # Without this, following vehicles can get a stop position beyond the
        # junction line, causing them to freeze or stutter near the signal.
        if direction == "right":
            self.stop = min(self.stop, defaultStop[direction])
        elif direction == "left":
            self.stop = max(self.stop, defaultStop[direction])
        elif direction == "down":
            self.stop = min(self.stop, defaultStop[direction])
        elif direction == "up":
            self.stop = max(self.stop, defaultStop[direction])
        # ─────────────────────────────────────────────────────────────────────

        if direction == "right":
            x[direction][lane] -= self.image.get_rect().width  + stoppingGap
        elif direction == "left":
            x[direction][lane] += self.image.get_rect().width  + stoppingGap
        elif direction == "down":
            y[direction][lane] -= self.image.get_rect().height + stoppingGap
        elif direction == "up":
            y[direction][lane] += self.image.get_rect().height + stoppingGap

        simulation.add(self)

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        global totalCrossed
        d   = self.direction
        w   = self.image.get_rect().width
        h   = self.image.get_rect().height
        ln  = self.lane
        idx = self.index

        if d == "right":
            if self.crossed == 0 and self.x + w > stopLines[d]:
                self._markCrossed()
            can_move = (
                (self.x + w <= self.stop or self.crossed == 1 or
                 (currentGreen == 0 and currentYellow == 0)) and
                (idx == 0 or self.x + w < vehicles[d][ln][idx - 1].x - movingGap)
            )
            if can_move:
                self.x += self.speed

        elif d == "down":
            if self.crossed == 0 and self.y + h > stopLines[d]:
                self._markCrossed()
            can_move = (
                (self.y + h <= self.stop or self.crossed == 1 or
                 (currentGreen == 1 and currentYellow == 0)) and
                (idx == 0 or self.y + h < vehicles[d][ln][idx - 1].y - movingGap)
            )
            if can_move:
                self.y += self.speed

        elif d == "left":
            if self.crossed == 0 and self.x < stopLines[d]:
                self._markCrossed()
            can_move = (
                (self.x >= self.stop or self.crossed == 1 or
                 (currentGreen == 2 and currentYellow == 0)) and
                (idx == 0 or self.x > vehicles[d][ln][idx - 1].x +
                 vehicles[d][ln][idx - 1].image.get_rect().width + movingGap)
            )
            if can_move:
                self.x -= self.speed

        elif d == "up":
            if self.crossed == 0 and self.y < stopLines[d]:
                self._markCrossed()
            can_move = (
                (self.y >= self.stop or self.crossed == 1 or
                 (currentGreen == 3 and currentYellow == 0)) and
                (idx == 0 or self.y > vehicles[d][ln][idx - 1].y +
                 vehicles[d][ln][idx - 1].image.get_rect().height + movingGap)
            )
            if can_move:
                self.y -= self.speed

    def _markCrossed(self):
        global totalCrossed
        self.crossed                           = 1
        vehicles[self.direction]["crossed"]   += 1
        vehicleCrossedCount[self.vehicleClass] += 1
        totalCrossed                           += 1


# ── helpers ───────────────────────────────────────────────────────────────────

def getVehicleCount(direction):
    count = 0
    for lane in range(3):
        for v in vehicles[direction][lane]:
            if v.crossed == 0:
                count += 1
    return count


def greenTimeForAction(action):
    return MIN_GREEN_TIME + (MAX_GREEN_TIME - MIN_GREEN_TIME) // 2   # 17 s default


# ── signal control loop ───────────────────────────────────────────────────────

def initialize():
    signals.append(TrafficSignal(0, DEFAULT_YELLOW, MIN_GREEN_TIME))
    for _ in range(noOfSignals - 1):
        signals.append(TrafficSignal(99, DEFAULT_YELLOW, MIN_GREEN_TIME))
    repeat()


def repeat():
    global currentGreen, currentYellow

    while True:
        currentGreen = bridge.choose(vehicles)
        direction    = directionNumbers[currentGreen]

        green_time = greenTimeForAction(currentGreen)

        signals[currentGreen].green  = green_time
        signals[currentGreen].yellow = DEFAULT_YELLOW
        waitTime[currentGreen]       = 0.0
        currentYellow                = 0

        start   = time.time()
        elapsed = 0

        while elapsed < green_time:
            time.sleep(CHECK_INTERVAL)
            elapsed = time.time() - start

            for i in range(noOfSignals):
                if i != currentGreen:
                    waitTime[i] += CHECK_INTERVAL

            signals[currentGreen].green = max(0, green_time - int(elapsed))

            bridge.feedback(vehicles)

            if getVehicleCount(direction) == 0 and elapsed >= MIN_GREEN_TIME:
                break

        # Yellow phase
        currentYellow                = 1
        signals[currentGreen].yellow = DEFAULT_YELLOW
        y_start                      = time.time()

        while signals[currentGreen].yellow > 0:
            time.sleep(1)
            signals[currentGreen].yellow = max(
                0, DEFAULT_YELLOW - int(time.time() - y_start)
            )

        currentYellow = 0


# ── vehicle generator ─────────────────────────────────────────────────────────

def generateVehicles():
    while True:
        vehicle_type     = random.randint(0, 3)
        lane_number      = random.randint(1, 2)
        direction_number = random.choices([0, 1, 2, 3], weights=[25, 25, 25, 25])[0]
        Vehicle(
            lane_number,
            vehicleTypes[vehicle_type],
            direction_number,
            directionNumbers[direction_number],
        )
        time.sleep(random.uniform(1.5, 3.5))


# ── HUD drawing functions ─────────────────────────────────────────────────────

def drawVehicleCountPanel(screen, font, bold_font):
    px, py, pw, ph = 10, 620, 270, 170
    surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 180))
    screen.blit(surf, (px, py))
    pygame.draw.rect(screen, (255, 255, 255), (px, py, pw, ph), 2)
    screen.blit(bold_font.render("VEHICLES CROSSED", True, (255, 220, 0)), (px + 10, py + 8))
    pygame.draw.line(screen, (255, 255, 255), (px + 5, py + 32), (px + pw - 5, py + 32), 1)
    colors = {
        "car":   (100, 180, 255),
        "bus":   (255, 160, 60),
        "truck": (220, 80,  80),
        "bike":  (80,  210, 80),
    }
    for i, vt in enumerate(["car", "bus", "truck", "bike"]):
        txt = font.render("  " + vt.capitalize() + " : " + str(vehicleCrossedCount[vt]),
                          True, colors[vt])
        screen.blit(txt, (px + 10, py + 40 + i * 26))
    pygame.draw.line(screen, (255, 255, 255), (px + 5, py + 144), (px + pw - 5, py + 144), 1)
    screen.blit(bold_font.render("  TOTAL : " + str(totalCrossed), True, (255, 255, 255)),
                (px + 10, py + 148))


def drawDirectionCounts(screen, font, bold_font):
    positions  = {"right": (10, 450), "left": (1150, 450), "down": (560, 10), "up": (560, 768)}
    dir_labels = {"right": "-> Waiting", "left": "<- Waiting",
                  "down": "v Waiting",  "up": "^ Waiting"}
    for direction, pos in positions.items():
        count = getVehicleCount(direction)
        color = (100, 255, 100) if count == 0 else (255, 220, 0) if count <= 4 else (255, 80, 80)
        txt    = bold_font.render(dir_labels[direction] + ": " + str(count), True, color)
        tw, th = txt.get_size()
        bg     = pygame.Surface((tw + 10, th + 6), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        screen.blit(bg,  (pos[0] - 5, pos[1] - 3))
        screen.blit(txt, pos)


def drawRLHUD(screen, font, bold_font):
    m    = bridge.metrics
    px, py, pw, ph = 1095, 530, 295, 265

    surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
    surf.fill((0, 20, 40, 215))
    screen.blit(surf, (px, py))
    pygame.draw.rect(screen, (0, 200, 255), (px, py, pw, ph), 2)

    screen.blit(bold_font.render("RL AGENT", True, (0, 220, 255)), (px + 10, py + 8))
    pygame.draw.line(screen, (0, 200, 255), (px + 5, py + 28), (px + pw - 5, py + 28), 1)

    if m["was_explore"]:
        screen.blit(font.render("EXPLORING  (random)",  True, (255, 180, 0)), (px + 10, py + 34))
    else:
        screen.blit(font.render("EXPLOITING (Q-table)", True, (0, 255, 120)), (px + 10, py + 34))

    state = m["state"] or (0,) * 8
    screen.blit(font.render("State: " + str(state), True, (180, 180, 180)), (px + 10, py + 54))

    pygame.draw.line(screen, (0, 200, 255), (px + 5, py + 72), (px + pw - 5, py + 72), 1)
    screen.blit(font.render("Q-values + Wait:", True, (180, 180, 180)), (px + 10, py + 76))

    qv         = m["q_values"]
    waits      = m.get("wait_steps", [0, 0, 0, 0])
    best       = qv.index(max(qv))
    q_min      = min(qv)
    q_range    = max(qv) - q_min + 1e-9
    bar_labels = ["->R", "vD ", "<-L", "^U "]
    bar_colors = [(0, 200, 255), (255, 160, 60), (100, 255, 100), (200, 100, 255)]

    for i in range(4):
        bx      = px + 10
        by      = py + 94 + i * 36
        bar_len = int((qv[i] - q_min) / q_range * 90)
        col     = bar_colors[i]

        if i == best:
            pygame.draw.rect(screen, col,             (bx, by, max(bar_len, 4), 13))
            pygame.draw.rect(screen, (255, 255, 255), (bx, by, max(bar_len, 4), 13), 1)
        else:
            pygame.draw.rect(screen, (col[0]//3, col[1]//3, col[2]//3),
                             (bx, by, max(bar_len, 4), 13))

        marker    = " <" if i == best else ""
        txt_color = col if i == best else (140, 140, 140)
        screen.blit(font.render(
            bar_labels[i] + " Q:" + str(round(qv[i], 1)) + marker,
            True, txt_color), (bx + 98, by))

        wait_val = waits[i]
        wait_len = int(min(wait_val / 30.0, 1.0) * 90)
        wait_col = (255, 60, 60) if wait_val > 15 else (255, 200, 0) if wait_val > 8 else (60, 200, 60)
        pygame.draw.rect(screen, (40, 40, 40), (bx, by + 16, 90, 7))
        pygame.draw.rect(screen, wait_col,     (bx, by + 16, max(wait_len, 2), 7))
        screen.blit(font.render("wait:" + str(wait_val) + "s", True, wait_col), (bx + 98, by + 14))

    pygame.draw.line(screen, (0, 200, 255), (px + 5, py + 242), (px + pw - 5, py + 242), 1)
    screen.blit(font.render(
        "Reward:" + str(round(m["total_reward"], 0)) + "  Steps:" + str(m["steps"]),
        True, (200, 200, 200)), (px + 10, py + 246))


# ── main ─────────────────────────────────────────────────────────────────────

thread1 = threading.Thread(target=initialize, daemon=True)
thread1.start()

screen       = pygame.display.set_mode((1400, 800))
background   = pygame.image.load("images/intersection.png")
redSignal    = pygame.image.load("images/signals/red.png")
yellowSignal = pygame.image.load("images/signals/yellow.png")
greenSignal  = pygame.image.load("images/signals/green.png")
font         = pygame.font.Font(None, 26)
bold_font    = pygame.font.Font(None, 28)

pygame.display.set_caption("Traffic Intersection — RL Controlled")

thread2 = threading.Thread(target=generateVehicles, daemon=True)
thread2.start()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            bridge.agent.save("qtable.json")
            sys.exit()

    screen.blit(background, (0, 0))

    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 1:
                signals[i].signalText = signals[i].yellow
                screen.blit(yellowSignal, signalCoods[i])
            else:
                signals[i].signalText = signals[i].green
                screen.blit(greenSignal, signalCoods[i])
        else:
            signals[i].signalText = "---"
            screen.blit(redSignal, signalCoods[i])

    for i in range(noOfSignals):
        screen.blit(font.render(str(signals[i].signalText), True, (255, 255, 255), (0, 0, 0)),
                    signalTimerCoods[i])

    for vehicle in simulation:
        screen.blit(vehicle.image, [vehicle.x, vehicle.y])
        vehicle.move()

    drawVehicleCountPanel(screen, font, bold_font)
    drawDirectionCounts(screen, font, bold_font)
    drawRLHUD(screen, font, bold_font)

    pygame.display.update()