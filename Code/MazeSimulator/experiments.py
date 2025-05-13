import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from IPython.display import HTML
import time

# Import our simulation class
from maze_simulation import MazeSimulation  # Assuming above code is saved as maze_simulation.py

def experiment_1_basic_maze():
    """
    Experiment 1: Simple maze with a single path from source to target
    This recreates the basic concept from the paper showing cells can navigate
    around corners following attractant gradients
    """
    print("Running Experiment 1: Basic Maze Navigation")
    
    # Create simulation with parameters suitable for demonstrating the basic concept
    sim = MazeSimulation(maze_size=(40, 40), n_cells=30, 
                         diffusion_rate=0.2, decay_rate=0.001,
                         consumption_rate=0.05, initial_concentration=1.0,
                         cell_sensitivity=0.05, random_movement_prob=0.1)
    
    # Create a simple L-shaped maze to demonstrate "seeing around corners"
    sim.maze = np.zeros((40, 40))
    
    # Add boundary walls
    sim.maze[0, :] = 1
    sim.maze[-1, :] = 1
    sim.maze[:, 0] = 1
    sim.maze[:, -1] = 1
    
    # Create L-shaped corridor
    sim.maze[5:35, 20] = 1  # Vertical wall
    sim.maze[20, 5:20] = 1  # Horizontal wall part
    
    # Set source and target manually to demonstrate "seeing around corners"
    sim.source_pos = (30, 10)  # Bottom left of the L
    sim.target_pos = (10, 30)  # Top right of the L
    sim.concentration = np.zeros((40, 40))
    sim.concentration[sim.target_pos] = sim.initial_concentration
    
    # Place cells near source
    sim.place_cells_near_source()
    
    # Run simulation
    print("Simulating cell movement...")
    frames = sim.run_simulation(steps=200, visualize=True)
    
    # Plot final state and trajectories
    plt.figure(figsize=(12, 5))
    
    # Plot final concentration
    plt.subplot(1, 2, 1)
    
    # Plot maze walls
    wall_mask = sim.maze == 1
    wall_data = np.zeros_like(sim.maze, dtype=float)
    wall_data[wall_mask] = 1
    wall_cmap = LinearSegmentedColormap.from_list('wall_cmap', [(1, 1, 1), (0, 0, 0)], N=2)
    plt.imshow(wall_data, cmap=wall_cmap, interpolation='nearest', alpha=0.7)
    
    # Plot concentration
    cmap = LinearSegmentedColormap.from_list('concentration_cmap', [(1, 1, 1), (0, 0, 1)], N=256)
    plt.imshow(sim.concentration, cmap=cmap, interpolation='nearest', 
               vmin=0, vmax=sim.initial_concentration, alpha=0.5)
    
    plt.plot(sim.source_pos[1], sim.source_pos[0], 'go', markersize=10, label='Start')
    plt.plot(sim.target_pos[1], sim.target_pos[0], 'ro', markersize=10, label='Target')
    
    cell_positions = np.array(sim.cells)
    plt.plot(cell_positions[:, 1], cell_positions[:, 0], 'yo', markersize=5, alpha=0.7, label='Cells')
    
    plt.title('Final Concentration')
    plt.legend()
    
    # Plot trajectories
    plt.subplot(1, 2, 2)
    plt.imshow(wall_data, cmap=wall_cmap, interpolation='nearest', alpha=0.7)
    plt.plot(sim.source_pos[1], sim.source_pos[0], 'go', markersize=10, label='Start')
    plt.plot(sim.target_pos[1], sim.target_pos[0], 'ro', markersize=10, label='Target')
    
    # Plot trajectories with different colors
    colors = plt.cm.jet(np.linspace(0, 1, sim.n_cells))
    for i, history in enumerate(sim.cell_history):
        if history:
            path = np.array(history)
            plt.plot(path[:, 1], path[:, 0], '-', color=colors[i], linewidth=1, alpha=0.7)
    
    plt.title('Cell Trajectories')
    plt.tight_layout()
    plt.savefig('experiment1_results.png', dpi=300)
    plt.show()
    
    return sim, frames

def experiment_2_t_maze():
    """
    Experiment 2: T-Maze from the paper
    Recreates the T-maze experiment showing how cells can make decisions
    at T-junctions based on attractant gradients
    """
    print("Running Experiment 2: T-Maze Decision Making")
    
    # Create simulation
    sim = MazeSimulation(maze_size=(50, 50), n_cells=30, 
                         diffusion_rate=0.2, decay_rate=0.001,
                         consumption_rate=0.03, initial_concentration=1.0,
                         cell_sensitivity=0.05, random_movement_prob=0.05)
    
    # Create a T-maze
    sim.maze = np.ones((50, 50))  # Start with all walls
    
    # Create T-shape corridor
    # Vertical corridor
    sim.maze[5:35, 24:27] = 0
    # Horizontal corridor
    sim.maze[34:37, 5:45] = 0
    
    # Set source at bottom of T
    sim.source_pos = (30, 25)
    
    # Set target at one end of the T
    sim.target_pos = (35, 40)
    
    # Initialize concentration
    sim.concentration = np.zeros((50, 50))
    sim.concentration[sim.target_pos] = sim.initial_concentration
    
    # Place cells near source
    sim.place_cells_near_source()
    
    # Run simulation
    print("Simulating cell movement...")
    frames = sim.run_simulation(steps=200, visualize=True)
    
    # Plot final state and trajectories
    plt.figure(figsize=(12, 5))
    
    # Plot final concentration
    plt.subplot(1, 2, 1)
    
    # Plot maze walls
    wall_mask = sim.maze == 1
    wall_data = np.zeros_like(sim.maze, dtype=float)
    wall_data[wall_mask] = 1
    wall_cmap = LinearSegmentedColormap.from_list('wall_cmap', [(1, 1, 1), (0, 0, 0)], N=2)
    plt.imshow(wall_data, cmap=wall_cmap, interpolation='nearest', alpha=0.7)
    
    # Plot concentration
    cmap = LinearSegmentedColormap.from_list('concentration_cmap', [(1, 1, 1), (0, 0, 1)], N=256)
    plt.imshow(sim.concentration, cmap=cmap, interpolation='nearest', 
               vmin=0, vmax=sim.initial_concentration, alpha=0.5)
    
    plt.plot(sim.source_pos[1], sim.source_pos[0], 'go', markersize=10, label='Start')
    plt.plot(sim.target_pos[1], sim.target_pos[0], 'ro', markersize=10, label='Target')
    
    cell_positions = np.array(sim.cells)
    plt.plot(cell_positions[:, 1], cell_positions[:, 0], 'yo', markersize=5, alpha=0.7, label='Cells')
    
    plt.title('Final Concentration')
    plt.legend()
    
    # Plot trajectories
    plt.subplot(1, 2, 2)
    plt.imshow(wall_data, cmap=wall_cmap, interpolation='nearest', alpha=0.7)
    plt.plot(sim.source_pos[1], sim.source_pos[0], 'go', markersize=10, label='Start')
    plt.plot(sim.target_pos[1], sim.target_pos[0], 'ro', markersize=10, label='Target')
    
    # Plot trajectories with different colors
    colors = plt.cm.jet(np.linspace(0, 1, sim.n_cells))
    for i, history in enumerate(sim.cell_history):
        if history:
            path = np.array(history)
            plt.plot(path[:, 1], path[:, 0], '-', color=colors[i], linewidth=1, alpha=0.7)
    
    plt.title('Cell Trajectories')
    plt.tight_layout()
    plt.savefig('experiment2_results.png', dpi=300)
    plt.show()
    
    return sim, frames

def experiment_3_complex_maze():
    """
    Experiment 3: Complex Maze Navigation
    Recreates the complex maze navigation scenario from the paper
    """
    print("Running Experiment 3: Complex Maze Navigation")
    
    # Create simulation
    sim = MazeSimulation(maze_size=(60, 60), n_cells=40, 
                         diffusion_rate=0.2, decay_rate=0.001,
                         consumption_rate=0.02, initial_concentration=1.0,
                         cell_sensitivity=0.05, random_movement_prob=0.05)
    
    # Generate complex maze
    sim.generate_complex_maze(complexity=0.8)
    
    # Set source and target
    sim.set_source_and_target()
    
    # Run simulation
    print("Simulating cell movement...")
    frames = sim.run_simulation(steps=300, visualize=True)
    
    # Plot final state and trajectories
    plt.figure(figsize=(12, 5))
    
    # Plot final concentration
    plt.subplot(1, 2, 1)
    
    # Plot maze walls
    wall_mask = sim.maze == 1
    wall_data = np.zeros_like(sim.maze, dtype=float)
    wall_data[wall_mask] = 1
    wall_cmap = LinearSegmentedColormap.from_list('wall_cmap', [(1, 1, 1), (0, 0, 0)], N=2)
    plt.imshow(wall_data, cmap=wall_cmap, interpolation='nearest', alpha=0.7)
    
    # Plot concentration
    cmap = LinearSegmentedColormap.from_list('concentration_cmap', [(1, 1, 1), (0, 0, 1)], N=256)
    plt.imshow(sim.concentration, cmap=cmap, interpolation='nearest', 
               vmin=0, vmax=sim.initial_concentration, alpha=0.5)
    
    plt.plot(sim.source_pos[1], sim.source_pos[0], 'go', markersize=10, label='Start')
    plt.plot(sim.target_pos[1], sim.target_pos[0], 'ro', markersize=10, label='Target')
    
    cell_positions = np.array(sim.cells)
    plt.plot(cell_positions[:, 1], cell_positions[:, 0], 'yo', markersize=5, alpha=0.7, label='Cells')
    
    plt.title('Final Concentration')
    plt.legend()
    
    # Plot trajectories
    plt.subplot(1, 2, 2)
    plt.imshow(wall_data, cmap=wall_cmap, interpolation='nearest', alpha=0.7)
    plt.plot(sim.source_pos[1], sim.source_pos[0], 'go', markersize=10, label='Start')
    plt.plot(sim.target_pos[1], sim.target_pos[0], 'ro', markersize=10, label='Target')
    
    # Plot trajectories with different colors
    colors = plt.cm.jet(np.linspace(0, 1, sim.n_cells))
    for i, history in enumerate(sim.cell_history):
        if history:
            path = np.array(history)
            plt.plot(path[:, 1], path[:, 0], '-', color=colors[i], linewidth=1, alpha=0.7)
    
    plt.title('Cell Trajectories')
    plt.tight_layout()
    plt.savefig('experiment3_results.png', dpi=300)
    plt.show()
    
    return sim, frames

def experiment_4_multiple_sources():
    """
    Experiment 4: Multiple Attractant Sources
    Tests how cells respond when multiple food sources are present
    """
    print("Running Experiment 4: Multiple Attractant Sources")
    
    # Create simulation
    sim = MazeSimulation(maze_size=(50, 50), n_cells=50, 
                         diffusion_rate=0.2, decay_rate=0.001,
                         consumption_rate=0.02, initial_concentration=1.0,
                         cell_sensitivity=0.05, random_movement_prob=0.05)
    
    # Create a simple maze with open space and a few obstacles
    sim.maze = np.zeros((50, 50))
    
    # Add boundary walls
    sim.maze[0, :] = 1
    sim.maze[-1, :] = 1
    sim.maze[:, 0] = 1
    sim.maze[:, -1] = 1
    
    # Add some internal walls
    sim.maze[10:20, 10:12] = 1
    sim.maze[30:40, 25:27] = 1
    sim.maze[15:17, 30:45] = 1
    sim.maze[25:40, 40:42] = 1
    
    # Set source position
    sim.source_pos = (25, 10)
    
    # Set multiple targets (we'll have to handle this manually since our class only supports one target)
    sim.target_pos = (10, 40)  # This will be our "official" target
    target_positions = [(10, 40), (40, 10), (40, 40)]  # Three targets
    
    # Initialize concentration with multiple sources
    sim.concentration = np.zeros((50, 50))
    for target_pos in target_positions:
        sim.concentration[target_pos] = sim.initial_concentration
    
    # Place cells near source
    sim.place_cells_near_source()
    
    # Update the simulation to maintain target concentrations
    original_diffuse = sim.diffuse_attractant
    
    def multi_source_diffuse():
        original_diffuse()
        # Ensure all targets maintain their concentration
        for target_pos in target_positions:
            sim.concentration[target_pos] = sim.initial_concentration
    
    # Replace diffuse method temporarily
    sim.diffuse_attractant = multi_source_diffuse
    
    # Run simulation
    print("Simulating cell movement...")
    frames = sim.run_simulation(steps=200, visualize=True)
    
    # Plot final state and trajectories
    plt.figure(figsize=(12, 5))
    
    # Plot final concentration
    plt.subplot(1, 2, 1)
    
    # Plot maze walls
    wall_mask = sim.maze == 1
    wall_data = np.zeros_like(sim.maze, dtype=float)
    wall_data[wall_mask] = 1
    wall_cmap = LinearSegmentedColormap.from_list('wall_cmap', [(1, 1, 1), (0, 0, 0)], N=2)
    plt.imshow(wall_data, cmap=wall_cmap, interpolation='nearest', alpha=0.7)
    
    # Plot concentration
    cmap = LinearSegmentedColormap.from_list('concentration_cmap', [(1, 1, 1), (0, 0, 1)], N=256)
    plt.imshow(sim.concentration, cmap=cmap, interpolation='nearest', 
               vmin=0, vmax=sim.initial_concentration, alpha=0.5)
    
    plt.plot(sim.source_pos[1], sim.source_pos[0], 'go', markersize=10, label='Start')
    
    # Plot all targets
    for i, target_pos in enumerate(target_positions):
        if i == 0:
            plt.plot(target_pos[1], target_pos[0], 'ro', markersize=10, label='Targets')
        else:
            plt.plot(target_pos[1], target_pos[0], 'ro', markersize=10)
    
    cell_positions = np.array(sim.cells)
    plt.plot(cell_positions[:, 1], cell_positions[:, 0], 'yo', markersize=5, alpha=0.7, label='Cells')
    
    plt.title('Final Concentration')
    plt.legend()
    
    # Plot trajectories
    plt.subplot(1, 2, 2)
    plt.imshow(wall_data, cmap=wall_cmap, interpolation='nearest', alpha=0.7)
    plt.plot(sim.source_pos[1], sim.source_pos[0], 'go', markersize=10, label='Start')
    
    # Plot all targets
    for i, target_pos in enumerate(target_positions):
        if i == 0:
            plt.plot(target_pos[1], target_pos[0], 'ro', markersize=10, label='Targets')
        else:
            plt.plot(target_pos[1], target_pos[0], 'ro', markersize=10)
    
    # Plot trajectories with different colors
    colors = plt.cm.jet(np.linspace(0, 1, sim.n_cells))
    for i, history in enumerate(sim.cell_history):
        if history:
            path = np.array(history)
            plt.plot(path[:, 1], path[:, 0], '-', color=colors[i], linewidth=1, alpha=0.7)
    
    plt.title('Cell Trajectories')
    plt.tight_layout()
    plt.savefig('experiment4_results.png', dpi=300)
    plt.show()
    
    # Restore original diffuse method
    sim.diffuse_attractant = original_diffuse
    
    return sim, frames

def experiment_5_parameter_analysis():
    """
    Experiment 5: Parameter Analysis
    Analyzes how changing different parameters affects the behavior of the system
    """
    print("Running Experiment 5: Parameter Analysis")
    
    # Parameters to analyze
    diffusion_rates = [0.1, 0.2, 0.3]
    consumption_rates = [0.01, 0.05, 0.1]
    
    # Create a simple maze for consistent testing
    maze_size = (40, 40)
    maze = np.zeros(maze_size)
    # Add boundary walls
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1
    # Add a central wall with gaps
    maze[10:30, 20] = 1
    maze[20, :20] = 1
    maze[20, 10] = 0  # Gap
    
    source_pos = (30, 10)
    target_pos = (10, 30)
    
    # Results storage
    success_counts = np.zeros((len(diffusion_rates), len(consumption_rates)))
    avg_time_to_target = np.zeros((len(diffusion_rates), len(consumption_rates)))
    
    # Run parameter analysis
    for i, diff_rate in enumerate(diffusion_rates):
        for j, cons_rate in enumerate(consumption_rates):
            print(f"Testing diffusion rate {diff_rate}, consumption rate {cons_rate}")
            
            # Create simulation with these parameters
            sim = MazeSimulation(maze_size=maze_size, n_cells=30, 
                                diffusion_rate=diff_rate, decay_rate=0.001,
                                consumption_rate=cons_rate, initial_concentration=1.0,
                                cell_sensitivity=0.05, random_movement_prob=0.05)
            
            sim.maze = maze.copy()
            sim.source_pos = source_pos
            sim.target_pos = target_pos
            sim.concentration = np.zeros(maze_size)
            sim.concentration[target_pos] = sim.initial_concentration
            
            # Place cells near source
            sim.place_cells_near_source()
            
            # Track cells that reach the target
            target_reached = [False] * sim.n_cells
            time_to_target = [0] * sim.n_cells
            
            # Run simulation for fixed steps
            total_steps = 200
            for step in range(total_steps):
                sim.diffuse_attractant()
                sim.move_cells()
                
                # Check if cells reached target
                for k, cell_pos in enumerate(sim.cells):
                    if not target_reached[k]:
                        # If cell is within 3 units of target, consider it reached
                        dist = np.sqrt((cell_pos[0] - target_pos[0])**2 + (cell_pos[1] - target_pos[1])**2)
                        if dist < 3:
                            target_reached[k] = True
                            time_to_target[k] = step + 1
            
            # Calculate metrics
            success_counts[i, j] = sum(target_reached) / sim.n_cells
            # Average time for cells that reached the target
            reached_times = [t for t, reached in zip(time_to_target, target_reached) if reached]
            avg_time_to_target[i, j] = np.mean(reached_times) if reached_times else float('inf')
    
    # Plot results
    plt.figure(figsize=(16, 6))
    
    # Plot success rate
    plt.subplot(1, 2, 1)
    plt.imshow(success_counts, interpolation='nearest', origin='lower',
              extent=[min(consumption_rates), max(consumption_rates),
                     min(diffusion_rates), max(diffusion_rates)])
    plt.colorbar(label='Success Rate')
    plt.xlabel('Consumption Rate')
    plt.ylabel('Diffusion Rate')
    plt.title('Fraction of Cells Reaching Target')
    
    # Plot average time to target
    plt.subplot(1, 2, 2)
    # Replace inf with NaN for better visualization
    avg_time_to_target[avg_time_to_target == float('inf')] = np.nan
    plt.imshow(avg_time_to_target, interpolation='nearest', origin='lower',
              extent=[min(consumption_rates), max(consumption_rates),
                     min(diffusion_rates), max(diffusion_rates)])
    plt.colorbar(label='Time Steps')
    plt.xlabel('Consumption Rate')
    plt.ylabel('Diffusion Rate')
    plt.title('Average Time to Reach Target')
    
    plt.tight_layout()
    plt.savefig('experiment5_results.png', dpi=300)
    plt.show()
    
    return success_counts, avg_time_to_target

def run_all_experiments():
    """Run all experiments and save results"""
    exp1_sim, exp1_frames = experiment_1_basic_maze()
    exp2_sim, exp2_frames = experiment_2_t_maze()
    exp3_sim, exp3_frames = experiment_3_complex_maze()
    exp4_sim, exp4_frames = experiment_4_multiple_sources()
    success_rates, avg_times = experiment_5_parameter_analysis()
    
    print("All experiments completed!")
    return {
        "exp1": (exp1_sim, exp1_frames),
        "exp2": (exp2_sim, exp2_frames),
        "exp3": (exp3_sim, exp3_frames),
        "exp4": (exp4_sim, exp4_frames),
        "exp5": (success_rates, avg_times)
    }

if __name__ == "__main__":
    run_all_experiments()