import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import random

class ChemicalGradient:
    def __init__(self, width, height, sources=None):
        """Initialize chemical gradient field."""
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        
        # Create sources if not provided
        if sources is None:
            sources = [
                {"pos": (width * 0.8, height * 0.8), "strength": 10.0},  # Increased from 4.0
                {"pos": (width * 0.2, height * 0.2), "strength": 5.0}    # Added second source
            ]
        self.sources = sources
        
        # Apply sources to the grid
        self.update_grid()
        
    def update_grid(self):
        """Update chemical concentration grid based on sources."""
        self.grid = np.zeros((self.height, self.width))
        
        # For each point in the grid, calculate the concentration from all sources
        for y in range(self.height):
            for x in range(self.width):
                for source in self.sources:
                    sx, sy = source["pos"]
                    strength = source["strength"]
                    
                    # Calculate distance from the point to the source
                    distance = np.sqrt((x - sx)**2 + (y - sy)**2)
                    
                    # Concentration falls off with square of distance
                    if distance > 0:
                        self.grid[y, x] += strength / (1 + 0.005 * distance**2)  # Changed from 0.01 to 0.005
        
        # Normalize the grid
        max_val = np.max(self.grid)
        #if max_val > 0:
            # Cap at 1.0 instead of fully normalizing to preserve gradient shape
         #   self.grid = np.minimum(self.grid, 1.0)
    
    def get_concentration(self, x, y):
        """Get concentration at a specific point."""
        # Handle out of bounds with zero concentration
        if (x < 0 or x >= self.width or y < 0 or y >= self.height):
            return 0
        
        # Get concentration at the point
        ix, iy = int(x), int(y)
        return self.grid[iy, ix]
    
    def get_gradient(self, x, y, sample_distance=2.0):
        """Calculate gradient vector at the given point."""
        # Sample concentrations at adjacent points
        c_center = self.get_concentration(x, y)
        c_right = self.get_concentration(x + sample_distance, y)
        c_left = self.get_concentration(x - sample_distance, y)
        c_up = self.get_concentration(x, y - sample_distance)  # y decreases going up in the grid
        c_down = self.get_concentration(x, y + sample_distance)
        
        # Calculate gradient components
        dx = (c_right - c_left) / (2 * sample_distance)
        dy = (c_down - c_up) / (2 * sample_distance)
        
        return dx, dy


class Cell:
    def __init__(self, x, y, gradient, sensitivity=5.0, random_movement=0.1):
        """Initialize cell with position and movement parameters."""
        self.x = x
        self.y = y
        self.gradient = gradient
        self.sensitivity = sensitivity  # How strongly the cell responds to the gradient
        self.random_movement = random_movement  # Amount of random movement
        self.history = [(x, y)]  # Track cell's path
        self.velocity = (0, 0)  # Current velocity vector
        self.max_speed = 4.0  # Maximum speed
        self.size = 1.0  # Cell size
        
    def sense_and_move(self):
        """Sense chemical gradient and adjust movement accordingly."""
        # Get gradient at current position
        dx, dy = self.gradient.get_gradient(self.x, self.y)
        
        # Apply sensitivity to gradient response
        force_x = dx * self.sensitivity
        force_y = dy * self.sensitivity
        
        # Add random movement component
        force_x += (random.random() - 0.5) * self.random_movement
        force_y += (random.random() - 0.5) * self.random_movement
        
        # Update velocity (with simple momentum)
        self.velocity = (
            0.7 * self.velocity[0] + 0.3 * force_x,
            0.7 * self.velocity[1] + 0.3 * force_y
        )
        
        # Limit speed
        speed = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if speed > self.max_speed:
            self.velocity = (
                self.velocity[0] * self.max_speed / speed,
                self.velocity[1] * self.max_speed / speed
            )
        
        # Update position
        self.x += self.velocity[0]
        self.y += self.velocity[1]
        
        # Ensure the cell stays within bounds
        self.x = max(0, min(self.gradient.width - 1, self.x))
        self.y = max(0, min(self.gradient.height - 1, self.y))
        
        # Store position in history
        self.history.append((self.x, self.y))


class Simulation:
    def __init__(self, width=300, height=300, num_cells=5):
        """Initialize simulation with environment and cells."""
        self.width = width
        self.height = height
        
        # Create chemical gradient
        self.gradient = ChemicalGradient(width, height)
        
        # Create cells
        self.cells = []
        for _ in range(num_cells):
            x = random.uniform(width * 0.4, width * 0.6)
            y = random.uniform(height * 0.4, height * 0.6)
            sensitivity = random.uniform(2.0, 5.0)  # Increased from 0.3-0.8
            random_movement = random.uniform(0.1, 0.4)  # Decreased from 0.1-0.4
            self.cells.append(Cell(x, y, self.gradient, sensitivity, random_movement))
    
    def step(self):
        """Advance simulation by one time step."""
        for cell in self.cells:
            cell.sense_and_move()
    
    def run_simulation(self, steps=100):
        """Run simulation for a specified number of steps."""
        for _ in range(steps):
            self.step()
        
        # Return final state
        return {
            "gradient": self.gradient.grid,
            "cells": [(cell.x, cell.y) for cell in self.cells],
            "paths": [cell.history for cell in self.cells]
        }
    
    def visualize(self, fig_size=(10, 8)):
        """Create static visualization of current state."""
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Plot chemical gradient
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'lightblue', 'blue'], N=256)
        gradient_img = ax.imshow(self.gradient.grid, cmap=cmap, origin='lower', 
                                 extent=(0, self.width, 0, self.height))
        plt.colorbar(gradient_img, ax=ax, label='Chemical Concentration')
        
        # Plot cells and their paths
        for i, cell in enumerate(self.cells):
            # Plot path
            path = np.array(cell.history)
            ax.plot(path[:, 0], path[:, 1], '-', linewidth=1, alpha=0.7, 
                    color=plt.cm.tab10(i % 10))
            
            # Plot cell
            circle = patches.Circle((cell.x, cell.y), cell.size, 
                                   color=plt.cm.tab10(i % 10), 
                                   alpha=0.8)
            ax.add_patch(circle)
        
        # Plot chemical sources
        for source in self.gradient.sources:
            pos = source["pos"]
            strength = source["strength"]
            circle = patches.Circle(pos, 2 + strength * 2, 
                                   color='red', alpha=0.7)
            ax.add_patch(circle)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title('Cell Chemotaxis Simulation')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        
        return fig
    
    def animate(self, frames=200, interval=50):
        """Create animation of cells moving over time."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot chemical gradient in the background
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'lightblue', 'blue'], N=256)
        gradient_img = ax.imshow(self.gradient.grid, cmap=cmap, origin='lower', 
                                 extent=(0, self.width, 0, self.height))
        plt.colorbar(gradient_img, ax=ax, label='Chemical Concentration')
        
        # Initialize cell visualization
        cell_circles = []
        cell_paths = []
        
        for i, cell in enumerate(self.cells):
            # Empty line for the path
            line, = ax.plot([], [], '-', linewidth=1, alpha=0.7, 
                           color=plt.cm.tab10(i % 10))
            cell_paths.append(line)
            
            # Circle for the cell
            circle = patches.Circle((cell.x, cell.y), cell.size, 
                                   color=plt.cm.tab10(i % 10), 
                                   alpha=0.8)
            ax.add_patch(circle)
            cell_circles.append(circle)
        
        # Plot chemical sources
        for source in self.gradient.sources:
            pos = source["pos"]
            strength = source["strength"]
            circle = patches.Circle(pos, 2 + strength * 2, 
                                   color='red', alpha=0.7)
            ax.add_patch(circle)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title('Cell Chemotaxis Simulation')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        
        # Reset cell histories for animation
        for cell in self.cells:
            cell.history = [(cell.x, cell.y)]
        
        def init():
            """Initialize animation."""
            for line in cell_paths:
                line.set_data([], [])
            return cell_paths + cell_circles
        
        def animate(frame):
            """Update animation by one frame."""
            # Move cells
            self.step()
            
            # Update visualization
            for i, (cell, line, circle) in enumerate(zip(self.cells, cell_paths, cell_circles)):
                # Update path
                path = np.array(cell.history)
                line.set_data(path[:, 0], path[:, 1])
                
                # Update cell position
                circle.center = (cell.x, cell.y)
            
            return cell_paths + cell_circles
        
        anim = FuncAnimation(fig, animate, frames=frames, init_func=init, 
                             interval=interval, blit=True)
        
        return anim

# Example usage
if __name__ == "__main__":
    # Create simulation with more cells and larger size
    sim = Simulation(width=200, height=200, num_cells=10)
    
    # Increase chemical source strengths
    sim.gradient.sources = [
        {"pos": (sim.width * 0.8, sim.height * 0.8), "strength": 10.0},
    ]
    
    # Update the gradient with the new source configurations
    sim.gradient.update_grid()
    
    # Generate static visualization
    fig = sim.visualize()
    plt.savefig('chemotaxis_static.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Generate animation
    anim = sim.animate(frames=200, interval=50)
    
    # Save animation (requires ffmpeg to be installed)
    try:
        anim.save('chemotaxis_animation.mp4', writer='ffmpeg', fps=20, dpi=150)
    except Exception as e:
        print(f"Could not save animation: {e}")
        print("To save animations, install ffmpeg and make sure it's in your PATH")
    
    plt.show()  # Display the animation in a window