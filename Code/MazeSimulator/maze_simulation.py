import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import convolve
import matplotlib.animation as animation
from IPython.display import HTML
import random

class MazeSimulation:
    def __init__(self, maze_size=(50, 50), n_cells=10, 
                 diffusion_rate=0.1, decay_rate=0.001, 
                 consumption_rate=0.05, initial_concentration=1.0,
                 cell_sensitivity=0.1, random_movement_prob=0.1):
        """
        Initialize the simulation with maze and parameters.
        
        Parameters:
        -----------
        maze_size : tuple
            Size of the maze grid (height, width)
        n_cells : int
            Number of cells in the simulation
        diffusion_rate : float
            Rate of attractant diffusion
        decay_rate : float
            Natural decay rate of the attractant
        consumption_rate : float
            Rate at which cells consume the attractant
        initial_concentration : float
            Initial concentration of attractant at the source
        cell_sensitivity : float
            Cell sensitivity to concentration gradients
        random_movement_prob : float
            Probability of random movement instead of gradient following
        """
        self.height, self.width = maze_size
        self.n_cells = n_cells
        self.diffusion_rate = diffusion_rate
        self.decay_rate = decay_rate
        self.consumption_rate = consumption_rate
        self.initial_concentration = initial_concentration
        self.cell_sensitivity = cell_sensitivity
        self.random_movement_prob = random_movement_prob
        
        # Initialize maze with walls (0 = open space, 1 = wall)
        self.maze = np.zeros(maze_size)
        
        # Initialize attractant concentration (diffusible chemical)
        self.concentration = np.zeros(maze_size)
        
        # For diffusion calculation
        self.diffusion_kernel = np.array([[0.05, 0.2, 0.05],
                                         [0.2, 0, 0.2],
                                         [0.05, 0.2, 0.05]])
        
        # Cell positions (y, x) coordinates
        self.cells = []
        
        # Cell movement history for plotting trajectories
        self.cell_history = []
        
        # Initialize data for visualization
        self.fig = None
        self.ax = None
        self.im = None
        self.cell_plots = None
        
    def generate_simple_maze(self):
        """Generate a simple maze with walls"""
        # Start with all open
        self.maze = np.zeros((self.height, self.width))
        
        # Add boundary walls
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1
        
        # Add some internal walls to create a simple maze
        # Horizontal walls
        self.maze[10, 10:40] = 1
        self.maze[20, 10:30] = 1
        self.maze[30, 20:40] = 1
        
        # Vertical walls
        self.maze[10:30, 15] = 1
        self.maze[20:40, 25] = 1
        self.maze[10:20, 35] = 1
        
        # Ensure there are openings in the walls
        self.maze[10, 25] = 0  # Opening in first horizontal wall
        self.maze[20, 20] = 0  # Opening in second horizontal wall
        self.maze[30, 30] = 0  # Opening in third horizontal wall
        self.maze[20, 15] = 0  # Opening in first vertical wall
        self.maze[30, 25] = 0  # Opening in second vertical wall
        
        return self.maze
    
    def generate_complex_maze(self, complexity=0.75):
        """Generate a more complex maze using a randomized approach"""
        # Start with all walls
        self.maze = np.ones((self.height, self.width))
        
        # Create a path through the maze
        # Start with a grid of cells, each containing a wall
        # Carve passages through walls by selecting cells randomly
        
        # Define the grid size (must be odd for proper maze generation)
        grid_h, grid_w = (self.height - 1) // 2, (self.width - 1) // 2
        
        # Initialize the grid with all walls
        for i in range(grid_h):
            for j in range(grid_w):
                # Mark cells as open
                self.maze[2*i+1, 2*j+1] = 0
        
        # Define the directions for maze carving
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Start at a random cell
        stack = [(random.randint(0, grid_h-1), random.randint(0, grid_w-1))]
        visited = set([stack[0]])
        
        # Carve passages
        while stack:
            current = stack[-1]
            
            # Find unvisited neighbors
            neighbors = []
            for dy, dx in directions:
                ny, nx = current[0] + dy, current[1] + dx
                if 0 <= ny < grid_h and 0 <= nx < grid_w and (ny, nx) not in visited:
                    neighbors.append((ny, nx, dy, dx))
            
            if neighbors:
                # Choose a random neighbor
                ny, nx, dy, dx = random.choice(neighbors)
                
                # Remove the wall between current cell and chosen neighbor
                self.maze[2*current[0]+1 + dy, 2*current[1]+1 + dx] = 0
                
                # Mark neighbor as visited and add to stack
                visited.add((ny, nx))
                stack.append((ny, nx))
            else:
                # Backtrack
                stack.pop()
        
        # Ensure the maze is surrounded by walls
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1
        
        # Add some random openings based on complexity
        for _ in range(int((1-complexity) * grid_h * grid_w)):
            y = random.randint(1, self.height-2)
            x = random.randint(1, self.width-2)
            if self.maze[y, x] == 1:
                # Check if removing this wall doesn't create a 2x2 open space
                neighbors = [(y+dy, x+dx) for dy, dx in directions]
                if sum(1 for ny, nx in neighbors if 0 <= ny < self.height and 0 <= nx < self.width and self.maze[ny, nx] == 0) <= 2:
                    self.maze[y, x] = 0
        
        return self.maze
    
    def set_source_and_target(self):
        """Set the source and target positions in the maze"""
        # Find viable positions (not walls)
        viable_positions = np.where(self.maze == 0)
        viable_indices = [(y, x) for y, x in zip(viable_positions[0], viable_positions[1])]
        
        if not viable_indices:
            raise ValueError("No viable positions found in the maze")
        
        # Pick positions at opposite sides of the maze if possible
        distances = {}
        for i, pos1 in enumerate(viable_indices):
            for j, pos2 in enumerate(viable_indices[i+1:], i+1):
                dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                distances[(i, j)] = dist
        
        if distances:
            source_idx, target_idx = max(distances, key=distances.get)
            self.source_pos = viable_indices[source_idx]
            self.target_pos = viable_indices[target_idx]
        else:
            # Fallback to random positions
            indices = random.sample(range(len(viable_indices)), 2)
            self.source_pos = viable_indices[indices[0]]
            self.target_pos = viable_indices[indices[1]]
        
        # Set initial concentration at the target (food source)
        self.concentration[self.target_pos] = self.initial_concentration
        
        # Place cells near the source
        self.place_cells_near_source()
        
        return self.source_pos, self.target_pos
    
    def place_cells_near_source(self):
        """Place cells near the source position"""
        self.cells = []
        self.cell_history = [[] for _ in range(self.n_cells)]
        
        # Get positions around the source that are not walls
        positions = []
        y, x = self.source_pos
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and 
                    self.maze[ny, nx] == 0):
                    positions.append((ny, nx))
        
        # Place cells randomly near the source
        for _ in range(self.n_cells):
            if positions:
                pos = random.choice(positions)
                self.cells.append(list(pos))  # Use list for mutability
                self.cell_history[_].append(pos)
            else:
                # Fallback if no positions available
                self.cells.append(list(self.source_pos))
                self.cell_history[_].append(self.source_pos)
    
    def diffuse_attractant(self):
        """Simulate diffusion of the attractant"""
        # Apply diffusion
        diffused = convolve(self.concentration, self.diffusion_kernel, mode='constant', cval=0)
        self.concentration = (1 - self.diffusion_rate) * self.concentration + self.diffusion_rate * diffused
        
        # Apply decay
        self.concentration -= self.decay_rate * self.concentration
        
        # Ensure concentration stays at the target
        self.concentration[self.target_pos] = self.initial_concentration
        
        # Ensure no diffusion through walls
        self.concentration[self.maze == 1] = 0
    
    def move_cells(self):
        """Move cells based on concentration gradient"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for i, (y, x) in enumerate(self.cells):
            # Consume attractant at current position
            self.concentration[y, x] = max(0, self.concentration[y, x] - self.consumption_rate)
            
            # Decide whether to follow gradient or move randomly
            if random.random() < self.random_movement_prob:
                # Random movement
                valid_moves = []
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < self.height and 0 <= nx < self.width and 
                        self.maze[ny, nx] == 0):
                        valid_moves.append((ny, nx))
                
                if valid_moves:
                    self.cells[i] = list(random.choice(valid_moves))
            else:
                # Gradient-based movement
                best_pos = (y, x)
                best_conc = self.concentration[y, x]
                
                # Check concentration in surrounding cells
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < self.height and 0 <= nx < self.width and 
                        self.maze[ny, nx] == 0):
                        conc = self.concentration[ny, nx]
                        if conc > best_conc + self.cell_sensitivity:
                            best_conc = conc
                            best_pos = (ny, nx)
                
                self.cells[i] = list(best_pos)
            
            # Record cell position for trajectory
            self.cell_history[i].append(tuple(self.cells[i]))
    
    def setup_visualization(self):
        """Setup the visualization"""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Create a custom colormap for concentration (blue gradient)
        colors = [(1, 1, 1), (0, 0, 1)]  # White to blue
        cmap = LinearSegmentedColormap.from_list('concentration_cmap', colors, N=256)
        
        # Plot the maze walls
        wall_mask = self.maze == 1
        wall_data = np.zeros_like(self.maze, dtype=float)
        wall_data[wall_mask] = 1
        wall_cmap = LinearSegmentedColormap.from_list('wall_cmap', [(1, 1, 1), (0, 0, 0)], N=2)
        self.ax.imshow(wall_data, cmap=wall_cmap, interpolation='nearest', alpha=0.7)
        
        # Plot the concentration
        self.im = self.ax.imshow(self.concentration, cmap=cmap, interpolation='nearest', 
                                vmin=0, vmax=self.initial_concentration, alpha=0.5)
        
        # Mark source and target
        self.ax.plot(self.source_pos[1], self.source_pos[0], 'go', markersize=10, label='Start')
        self.ax.plot(self.target_pos[1], self.target_pos[0], 'ro', markersize=10, label='Target')
        
        # Plot cells
        cell_positions = np.array(self.cells)
        self.cell_plots, = self.ax.plot(cell_positions[:, 1], cell_positions[:, 0], 'yo', markersize=5, alpha=0.7, label='Cells')
        
        # Add legend and title
        self.ax.legend(loc='upper right')
        self.ax.set_title('Cells Navigating Maze via Attractant Gradients')
        
        return self.fig, self.ax
    
    def update_visualization(self):
        """Update the visualization with current state"""
        # Update concentration display
        self.im.set_data(self.concentration)
        
        # Update cell positions
        cell_positions = np.array(self.cells)
        if len(cell_positions) > 0:  # Check if there are cells
            self.cell_plots.set_data(cell_positions[:, 1], cell_positions[:, 0])
        
        # Return the updated artists
        return self.im, self.cell_plots
    
    def plot_trajectories(self):
        """Plot the trajectories of cells"""
        plt.figure(figsize=(10, 8))
        
        # Plot maze walls
        wall_mask = self.maze == 1
        wall_data = np.zeros_like(self.maze, dtype=float)
        wall_data[wall_mask] = 1
        wall_cmap = LinearSegmentedColormap.from_list('wall_cmap', [(1, 1, 1), (0, 0, 0)], N=2)
        plt.imshow(wall_data, cmap=wall_cmap, interpolation='nearest', alpha=0.7)
        
        # Plot source and target
        plt.plot(self.source_pos[1], self.source_pos[0], 'go', markersize=10, label='Start')
        plt.plot(self.target_pos[1], self.target_pos[0], 'ro', markersize=10, label='Target')
        
        # Plot trajectories with different colors
        colors = plt.cm.jet(np.linspace(0, 1, self.n_cells))
        for i, history in enumerate(self.cell_history):
            if history:
                path = np.array(history)
                plt.plot(path[:, 1], path[:, 0], '-', color=colors[i], linewidth=1, alpha=0.7)
        
        plt.title('Cell Trajectories')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def capture_frame(self):
        """Capture the current visualization state as a frame for animation"""
        # Create a new figure with fixed dimensions - this avoids canvas sizing issues
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create a custom colormap for concentration (blue gradient)
        colors = [(1, 1, 1), (0, 0, 1)]  # White to blue
        cmap = LinearSegmentedColormap.from_list('concentration_cmap', colors, N=256)
        
        # Plot the maze walls
        wall_mask = self.maze == 1
        wall_data = np.zeros_like(self.maze, dtype=float)
        wall_data[wall_mask] = 1
        wall_cmap = LinearSegmentedColormap.from_list('wall_cmap', [(1, 1, 1), (0, 0, 0)], N=2)
        ax.imshow(wall_data, cmap=wall_cmap, interpolation='nearest', alpha=0.7)
        
        # Plot the concentration
        ax.imshow(self.concentration, cmap=cmap, interpolation='nearest', 
                  vmin=0, vmax=self.initial_concentration, alpha=0.5)
        
        # Mark source and target
        ax.plot(self.source_pos[1], self.source_pos[0], 'go', markersize=10, label='Start')
        ax.plot(self.target_pos[1], self.target_pos[0], 'ro', markersize=10, label='Target')
        
        # Plot cells
        cell_positions = np.array(self.cells)
        ax.plot(cell_positions[:, 1], cell_positions[:, 0], 'yo', markersize=5, alpha=0.7, label='Cells')
        
        # Add title
        ax.set_title(f'Cells Navigating Maze')
        
        # Render the figure
        fig.canvas.draw()
        
        # Convert to image array - fixed dimensions, no reshaping issues
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Close the figure to free memory
        plt.close(fig)
        
        return img
    
    def run_simulation(self, steps=100, visualize=True):
        """Run the simulation for a specified number of steps"""
        if visualize:
            self.setup_visualization()
            frames = []
            
            for _ in range(steps):
                self.diffuse_attractant()
                self.move_cells()
                
                # Use our reliable capture_frame method to get frames
                frames.append(self.capture_frame())
                
                # Update the main figure we already have for display
                self.update_visualization()
                self.ax.set_title(f'Simulation Step {_+1}')
                self.fig.canvas.draw()
            
            # Return frames for potential animation creation
            return frames
        else:
            # Run simulation without visualization
            for _ in range(steps):
                self.diffuse_attractant()
                self.move_cells()
    
    def create_animation(self, frames, filename='maze_simulation.mp4', fps=5):
        """Create an animation from frames"""
        if not frames:
            print("No frames to animate")
            return
        
        # Check if any frames were captured successfully
        if len(frames) == 0:
            print("No valid frames were captured for animation")
            return
        
        # Get dimensions from the first valid frame
        height, width, _ = frames[0].shape
        
        # Create a figure with matching dimensions
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        im = ax.imshow(frames[0])
        ax.axis('off')  # Hide axes for cleaner look
        
        def animate(i):
            if i < len(frames):
                im.set_array(frames[i])
            return [im]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                      interval=1000/fps, blit=True)
        
        # Save animation
        try:
            anim.save(filename, writer='ffmpeg', fps=fps)
            print(f"Animation saved as {filename}")
        except Exception as e:
            print(f"Error saving animation: {e}")
        
        plt.close(fig)
        return anim

# Example usage
def run_example():
    # Create simulation
    sim = MazeSimulation(maze_size=(50, 50), n_cells=20, 
                         diffusion_rate=0.2, decay_rate=0.005,
                         consumption_rate=0.1, initial_concentration=1.0,
                         cell_sensitivity=0.05, random_movement_prob=0.1)
    
    # Generate maze and set source/target
    sim.generate_complex_maze(complexity=0.8)
    sim.set_source_and_target()
    
    # Run simulation
    frames = sim.run_simulation(steps=100, visualize=True)
    
    # Create animation (optional)
    # sim.create_animation(frames, 'maze_simulation.mp4', fps=10)
    
    # Plot cell trajectories
    sim.plot_trajectories()

if __name__ == "__main__":
    run_example()