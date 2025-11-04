import numpy as np
import os
import matplotlib.pyplot as plt

# Create folder to save images
output_dir = 'predicted_figures_long2'
os.makedirs(output_dir, exist_ok=True)

# List of chosen sequences for saving images
chosen_sequences = [0, 2, 5]

# Parameters
num_measurements = 10

# Initialize MAPE storage
mape_values = []

conductivity_stack = []

# Loop through all sequences for error evaluation
for seq in range(np.load('united_triangular_matrices.npy').shape[0]):  # Replace with actual sequence range
    # Load predicted and true resistivity data
    initial = np.load(f"output_image_initial_series_{seq}.npy")
    initial = np.expand_dims(np.array(initial), axis=0)
    im1to4 = np.load(f"output_image_1_4_series_long2_{seq}.npy")
    initial_data = 1 / np.load('united_triangular_matrices.npy')[seq, 0, :, :] * 1000
    initial_data = np.expand_dims(initial_data, axis=0)

    resistivity_diff = np.concatenate((initial_data, initial, im1to4), axis=0)
    resistivity = np.cumsum(resistivity_diff, axis=0)
    resistivity = np.nan_to_num(resistivity, nan=0)
    
    conductivity_stack.append(resistivity)

    true_resistivity_data = 1 / np.load('united_triangular_matrices.npy') * 1000
    true_resistivity_data = np.nan_to_num(true_resistivity_data, nan=0)
    true_resistivity_data = true_resistivity_data[seq, 0::num_measurements, :, :]

    # Initialize MAPE calculation
    mape_sum = 0
    valid_pixel_count = 0

    for t in range(resistivity.shape[0]):
        true_values = true_resistivity_data[t, :, :]
        predicted_values = resistivity[t, :, :]

        # Avoid division by zero
        mask = true_values != 0
        relative_diff = np.abs((predicted_values - true_values) / true_values)

        # Accumulate the valid relative differences
        mape_sum += np.sum(relative_diff[mask])
        valid_pixel_count += np.sum(mask)

        # Save images for visualization only for selected sequences
        if seq in chosen_sequences:
            diff_map = np.abs(predicted_values - true_values)

            plt.figure(figsize=(18, 6))

            plt.subplot(1, 3, 1)
            plt.imshow(predicted_values, aspect='auto', cmap='hot')
            plt.colorbar(label="Predicted conductivity")
            plt.title(f"Predicted conductivity at Time Step {t} (Seq {seq})")

            plt.subplot(1, 3, 2)
            plt.imshow(true_values, aspect='auto', cmap='hot')
            plt.colorbar(label="True conductivity")
            plt.title(f"True conductivity at Time Step {t} (Seq {seq})")

            plt.subplot(1, 3, 3)
            plt.imshow(diff_map, aspect='auto', cmap='coolwarm')
            plt.colorbar(label="Difference")
            plt.title(f"Difference at Time Step {t} (Seq {seq})")

            # Save the figure
            figure_filename = os.path.join(output_dir, f'measurement_locations_seq_{seq}_timestep_{t}.png')
            plt.savefig(figure_filename)
            plt.close()

    # Compute MAPE for the sequence
    mape = (mape_sum / valid_pixel_count) * 100
    mape_values.append((seq, mape))
    print(f"Sequence {seq}: MAPE = {mape}%")

# Save MAPE values to a file
mape_file = os.path.join(output_dir, 'mape_values.txt')
with open(mape_file, 'w') as f:
    for seq, mape in mape_values:
        f.write(f"Sequence {seq}: MAPE = {mape}%\n")

print(f"MAPE values saved to {mape_file}")

np.save("conductivity.npy", conductivity_stack)
