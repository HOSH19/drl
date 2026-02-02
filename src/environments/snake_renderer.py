"""
Snake Game Renderer for visualization.
Uses matplotlib for rendering game state.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Tuple, Optional


class SnakeRenderer:
    """Renderer for Snake game using matplotlib."""
    
    def __init__(self, grid_size: int = 20, cell_size: int = 20):
        """
        Initialize renderer.
        
        Args:
            grid_size: Size of the game grid
            cell_size: Size of each cell in pixels
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fig = None
        self.ax = None
        self.initialized = False
    
    def _init_display(self):
        """Initialize matplotlib figure and axes."""
        if not self.initialized:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()  # Match array indexing
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_title('Snake Game', fontsize=16, fontweight='bold')
            self.ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
            self.initialized = True
    
    def render(
        self, 
        snake: List[Tuple[int, int]], 
        food: Optional[Tuple[int, int]], 
        score: int = 0,
        steps: int = 0
    ):
        """
        Render the current game state.
        
        Args:
            snake: List of (row, col) positions of snake body
            food: (row, col) position of food, or None
            score: Current score
            steps: Current step count
        """
        self._init_display()
        self.ax.clear()
        
        # Draw grid
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        
        # Draw background
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = patches.Rectangle(
                    (j, i), 1, 1,
                    linewidth=0.5,
                    edgecolor='lightgray',
                    facecolor='white',
                    alpha=0.3
                )
                self.ax.add_patch(rect)
        
        # Draw snake body
        for i, segment in enumerate(snake):
            row, col = segment
            if i == 0:
                # Head: darker green
                color = '#2E7D32'
                alpha = 1.0
            else:
                # Body: lighter green
                color = '#66BB6A'
                alpha = 0.8 - (i / len(snake)) * 0.3
            
            rect = patches.Rectangle(
                (col, row), 1, 1,
                linewidth=1,
                edgecolor='darkgreen',
                facecolor=color,
                alpha=alpha
            )
            self.ax.add_patch(rect)
        
        # Draw food
        if food:
            row, col = food
            circle = patches.Circle(
                (col + 0.5, row + 0.5),
                0.4,
                color='red',
                alpha=0.8
            )
            self.ax.add_patch(circle)
        
        # Add score and steps text
        info_text = f"Score: {score} | Steps: {steps} | Length: {len(snake)}"
        self.ax.text(
            0.5, -0.05,
            info_text,
            transform=self.ax.transAxes,
            ha='center',
            fontsize=12,
            fontweight='bold'
        )
        
        self.ax.set_title('Snake Game', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.pause(0.01)  # Small pause to update display
    
    def save_frame(self, filename: str, snake: List[Tuple[int, int]], 
                   food: Optional[Tuple[int, int]], score: int = 0, steps: int = 0):
        """Save current frame to file."""
        self.render(snake, food, score, steps)
        if self.fig:
            self.fig.savefig(filename, dpi=100, bbox_inches='tight')
    
    def close(self):
        """Close the renderer."""
        if self.fig:
            plt.close(self.fig)
            self.initialized = False
