import pylab
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
import pygame
from typing import List, Tuple

matplotlib.use("Agg")

def draw_plot(screen: pygame.Surface, x: list, y: list, x_label: str = 'Generation', y_label: str = 'Fitness') -> None:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.plot(x, y)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.tight_layout()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_argb()

    size = canvas.get_width_height()
    surf = pygame.image.fromstring(raw_data, size, "ARGB")
    screen.blit(surf, (0, 0))
    plt.close()
    
def draw_cities(screen: pygame.Surface, cities_locations: List[Tuple[int, int]], rgb_color: Tuple[int, int, int], node_radius: int) -> None:
    for city_location in cities_locations:
        pygame.draw.circle(screen, rgb_color, city_location, node_radius)

def draw_paths(screen: pygame.Surface, path: List[Tuple[int, int]], rgb_color: Tuple[int, int, int], width: int = 1):
    pygame.draw.lines(screen, rgb_color, True, path, width=width)

def draw_text(screen: pygame.Surface, text: str, color: pygame.Color) -> None:
    pygame.font.init()
    font_size = 15
    my_font = pygame.font.SysFont('Arial', font_size)
    text_surface = my_font.render(text, False, color)
    
    screen.blit(text_surface, (10, 10))
