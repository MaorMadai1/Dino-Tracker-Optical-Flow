import numpy as np

# Load the trajectory file
trajectories = np.load(r'C:\Users\Maor Madai\OneDrive - Technion\Documents\Technion\10 - SemH\03 - ProjectA\DinoTrackerProject\output_folder\davis_480\3\dinov2_grid_trajectory\trajectories_0_of0.npy')

# Extract query points (frame 0)
query_points = trajectories[:, 0, :]

print('Query Points for Video 3 (davis_480)')
print('=' * 50)
print(f'Total number of query points: {query_points.shape[0]}')
print('\nPoint #  |   X   |   Y')
print('-' * 50)

for i, (x, y) in enumerate(query_points):
    print(f'Point {i:2d} | {x:5d} | {y:5d}')

print('\n' + '=' * 50)
print('\nAs NumPy array:')
print(query_points)
