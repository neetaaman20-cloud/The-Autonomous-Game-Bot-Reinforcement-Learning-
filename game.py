import pygame
import random
import numpy as np
from collections import namedtuple
from enum import Enum

pygame.init()

FONT_LARGE  = pygame.font.SysFont("consolas", 28, bold=True)
FONT_MEDIUM = pygame.font.SysFont("consolas", 20)
FONT_SMALL  = pygame.font.SysFont("consolas", 15)

BG_DARK    = (10,  12,  20)
GRID_COLOR = (18,  22,  35)
SNAKE_HEAD = (0,  230, 180)
SNAKE_BODY = (0,  180, 130)
FOOD_COLOR = (255, 60,  80)
UI_ACCENT  = (80, 160, 255)
TEXT_COLOR = (200, 220, 255)
DIM_TEXT   = (80, 100, 140)
PANEL_BG   = (14,  18,  30)
BORDER_CLR = (30,  40,  65)

BLOCK   = 24
SPEED   = 60
W_CELLS = 26
H_CELLS = 20
W       = W_CELLS * BLOCK + 220
H       = H_CELLS * BLOCK + 80

Point = namedtuple("Point", "x y")

class Direction(Enum):
    RIGHT = 0
    DOWN  = 1
    LEFT  = 2
    UP    = 3

CLOCKWISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

class SnakeGameAI:
    def __init__(self, render=True):
        self.render_mode = render
        if render:
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("Autonomous RL Snake Bot")
            self.clock  = pygame.time.Clock()
        self.game_surface = pygame.Surface((W_CELLS * BLOCK, H_CELLS * BLOCK))
        self.reset()

    def reset(self):
        cx, cy = W_CELLS // 2, H_CELLS // 2
        self.dir   = Direction.RIGHT
        self.head  = Point(cx, cy)
        self.snake = [self.head, Point(cx - 1, cy), Point(cx - 2, cy)]
        self.score = 0
        self.food  = None
        self._place_food()
        self.frame_iteration = 0
        return self._get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, W_CELLS - 1)
            y = random.randint(0, H_CELLS - 1)
            p = Point(x, y)
            if p not in self.snake:
                self.food = p
                break

    def step(self, action):
        self.frame_iteration += 1
        if self.render_mode:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
        idx = CLOCKWISE.index(self.dir)
        if action == 0:
            new_dir = CLOCKWISE[idx]
        elif action == 1:
            new_dir = CLOCKWISE[(idx + 1) % 4]
        else:
            new_dir = CLOCKWISE[(idx - 1) % 4]
        self.dir  = new_dir
        self.head = self._next_head()
        self.snake.insert(0, self.head)
        reward, done = 0, False
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            done   = True
            reward = -10
            self.snake.pop()
            return self._get_state(), reward, done, self.score
        if self.head == self.food:
            self.score += 1
            reward      = 10
            self._place_food()
        else:
            self.snake.pop()
        if self.render_mode:
            self._draw()
        return self._get_state(), reward, done, self.score

    def _next_head(self):
        x, y = self.head
        if self.dir == Direction.RIGHT:
            x += 1
        elif self.dir == Direction.LEFT:
            x -= 1
        elif self.dir == Direction.DOWN:
            y += 1
        elif self.dir == Direction.UP:
            y -= 1
        return Point(x, y)

    def _is_collision(self, pt=None):
        pt = pt or self.head
        if not (0 <= pt.x < W_CELLS and 0 <= pt.y < H_CELLS):
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _get_state(self):
        head = self.head
        idx  = CLOCKWISE.index(self.dir)
        d_r  = CLOCKWISE[(idx + 1) % 4]
        d_l  = CLOCKWISE[(idx - 1) % 4]
        d_s  = self.dir
        def nxt(d):
            x, y = head
            if d == Direction.RIGHT:
                return Point(x + 1, y)
            if d == Direction.LEFT:
                return Point(x - 1, y)
            if d == Direction.DOWN:
                return Point(x, y + 1)
            return Point(x, y - 1)
        state = [
            self._is_collision(nxt(d_s)),
            self._is_collision(nxt(d_r)),
            self._is_collision(nxt(d_l)),
            self.dir == Direction.RIGHT,
            self.dir == Direction.DOWN,
            self.dir == Direction.LEFT,
            self.dir == Direction.UP,
            self.food.x > head.x,
            self.food.x < head.x,
            self.food.y > head.y,
            self.food.y < head.y,
        ]
        return np.array(state, dtype=np.float32)

    def _draw(self):
        self.screen.fill(BG_DARK)
        pygame.draw.rect(self.screen, PANEL_BG,   (0, 60, W_CELLS * BLOCK, H_CELLS * BLOCK))
        pygame.draw.rect(self.screen, BORDER_CLR, (0, 60, W_CELLS * BLOCK, H_CELLS * BLOCK), 1)
        gs = self.game_surface
        gs.fill(GRID_COLOR)
        for gx in range(0, W_CELLS * BLOCK, BLOCK):
            pygame.draw.line(gs, (22, 28, 45), (gx, 0), (gx, H_CELLS * BLOCK))
        for gy in range(0, H_CELLS * BLOCK, BLOCK):
            pygame.draw.line(gs, (22, 28, 45), (0, gy), (W_CELLS * BLOCK, gy))
        fx = self.food.x * BLOCK
        fy = self.food.y * BLOCK
        pygame.draw.rect(gs, FOOD_COLOR, (fx + 3, fy + 3, BLOCK - 6, BLOCK - 6), border_radius=6)
        for i, pt in enumerate(self.snake):
            bx = pt.x * BLOCK
            by = pt.y * BLOCK
            if i == 0:
                pygame.draw.rect(gs, SNAKE_HEAD, (bx + 1, by + 1, BLOCK - 2, BLOCK - 2), border_radius=6)
            else:
                t = i / max(len(self.snake), 1)
                c = tuple(int(SNAKE_BODY[j] + (SNAKE_HEAD[j] - SNAKE_BODY[j]) * (1 - t)) for j in range(3))
                pygame.draw.rect(gs, c, (bx + 2, by + 2, BLOCK - 4, BLOCK - 4), border_radius=4)
        self.screen.blit(gs, (0, 60))
        self._draw_panel(W_CELLS * BLOCK + 10)
        self._draw_header()
        pygame.display.flip()
        self.clock.tick(SPEED)

    def _draw_header(self):
        title = FONT_LARGE.render("AUTONOMOUS RL SNAKE", True, UI_ACCENT)
        self.screen.blit(title, (10, 15))
        pygame.draw.line(self.screen, BORDER_CLR, (0, 58), (W, 58), 1)

    def _draw_panel(self, px):
        pw = 200
        pygame.draw.rect(self.screen, PANEL_BG,   (px - 4, 60, pw, H_CELLS * BLOCK), border_radius=4)
        pygame.draw.rect(self.screen, BORDER_CLR, (px - 4, 60, pw, H_CELLS * BLOCK), 1, border_radius=4)
        def label(text, y, color=DIM_TEXT):
            s = FONT_SMALL.render(text, True, color)
            self.screen.blit(s, (px + 4, y))
        def value(text, y, color=TEXT_COLOR):
            s = FONT_MEDIUM.render(text, True, color)
            self.screen.blit(s, (px + 4, y))
        label("SCORE",     75)
        value(str(self.score), 92, SNAKE_HEAD)
        label("LENGTH",   130)
        value(str(len(self.snake)), 147)
        label("DIRECTION", 185)
        value(self.dir.name, 202, UI_ACCENT)
        label("FRAME",    240)
        value(str(self.frame_iteration), 257, DIM_TEXT)
        label("FOOD",     295)
        value(str(self.food.x) + "," + str(self.food.y), 312, FOOD_COLOR)
