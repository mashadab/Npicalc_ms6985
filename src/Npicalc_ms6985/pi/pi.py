import numpy as np
from multiprocessing import Pool
from functools import reduce

def map_function(n_samples):
    """
    Map function: Generate random points and count those inside the unit circle.
    
    Args:
        n_samples: Number of random points to generate
        
    Returns:
        Tuple of (points_inside_circle, total_points)
    """
    # Generate random points in [0,1) x [0,1)
    x = np.random.random(n_samples)
    y = np.random.random(n_samples)
    
    # Calculate distances from origin
    distances_squared = x**2 + y**2
    
    # Count points inside the unit circle (distance < 1)
    inside_circle = np.sum(distances_squared < 1.0)
    
    return (inside_circle, n_samples)

def reduce_function(acc, result):
    """
    Reduce function: Combine results from map operations.
    
    Args:
        acc: Accumulator tuple (total_inside, total_points)
        result: Current result tuple (inside, points)
        
    Returns:
        Updated accumulator tuple
    """
    return (acc[0] + result[0], acc[1] + result[1])

def estimate_pi(total_samples, n_processes=4):
    """
    Estimate π using map-reduce Monte Carlo simulation.
    
    Args:
        total_samples: Total number of samples to use
        n_processes: Number of parallel processes
        
    Returns:
        Estimated value of π
    """
    # Divide work into chunks for parallel processing
    samples_per_process = total_samples // n_processes
    sample_chunks = [samples_per_process] * n_processes
    
    # Map phase: parallel execution
    with Pool(processes=n_processes) as pool:
        results = pool.map(map_function, sample_chunks)
    
    # Reduce phase: combine all results
    total_inside, total_points = reduce(reduce_function, results, (0, 0))
    
    # π ≈ 4 * (points_inside_circle / total_points)
    pi_estimate = 4.0 * total_inside / total_points
    
    return pi_estimate

def main():
    # Example usage
    n_samples = 10_000_000
    n_procs = 4
    
    print(f"Estimating π using {n_samples:,} samples with {n_procs} processes...")
    pi_est = estimate_pi(n_samples, n_procs)
    
    print(f"Estimated π: {pi_est}")
    print(f"Actual π:    {np.pi}")
    print(f"Error:       {abs(pi_est - np.pi)}")

