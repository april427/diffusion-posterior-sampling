import numpy as np
import torch
import sys
import os

# Add the current directory to the path so we can import our custom dataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data.aoa_amp_dataset import AoaAmpDataset, convert_existing_aoa_amp_to_dataset
    dataset_available = True
except ImportError as e:
    print(f"Warning: Could not import AoA-Amp dataset: {e}")
    dataset_available = False

# Original data generation (kept for reference)  
x_distance = 100 # meter
y_distance = 100 # meter

grid = 0.5 # meter
num_x = int(x_distance / grid)
num_y = int(y_distance / grid)
x = np.linspace(-x_distance/2, x_distance/2, num_x)
y = np.linspace(-y_distance/2, y_distance/2, num_y)
xv, yv = np.meshgrid(x, y)

aoa_amp = []

bs_loc = np.array([0,0])
for i in range(num_x):
    for j in range(num_y):
        aoa = np.arctan2(yv[i,j]-bs_loc[1], xv[i,j]-bs_loc[0]) # radian
        amp = ((xv[i,j]-bs_loc[0])**2 + (yv[i,j]-bs_loc[1])**2) * 4 * np.pi
        aoa_amp.append(np.exp(1j*aoa) / amp)

print("Generated AoA-Amp data:")
print(f"Grid size: {num_x} x {num_y}")
print(f"Total data points: {len(aoa_amp)}")
print(f"Data type: complex numbers")

if dataset_available:
    # Convert to dataset format for training
    print("\nConverting aoa_amp data to training format...")
    grid_shape = (num_x, num_y)
    training_sample = convert_existing_aoa_amp_to_dataset(aoa_amp, grid_shape, output_size=256)
    print(f"Training sample shape: {training_sample.shape}")
    print(f"Data type: {training_sample.dtype}")
    print(f"Real part range: [{training_sample[0].min():.6f}, {training_sample[0].max():.6f}]")
    print(f"Imaginary part range: [{training_sample[1].min():.6f}, {training_sample[1].max():.6f}]")

    # Example of creating the dataset for training
    print("\nTesting AoA-Amp dataset creation...")
    try:
        dataset = AoaAmpDataset(
            x_distance=x_distance,
            y_distance=y_distance,
            grid=2.0,  # Larger grid for 256x256 output
            num_samples=5,  # Small number for testing
            output_size=256
        )

        print(f"Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Sample dtype: {sample.dtype}")
        print(f"Sample range: [{sample.min():.6f}, {sample.max():.6f}]")

        # Save a sample for inspection
        torch.save(training_sample, 'sample_aoa_amp_data.pt')
        print("Sample data saved as 'sample_aoa_amp_data.pt'")

    except Exception as e:
        print(f"Error creating dataset: {e}")

print("\n" + "="*70)
print("TRAINING WITH PRE-TRAINED MODEL")
print("="*70)
print("Your pre-trained model: models/ffhq_10m.pt")
print("")
print("To start training with the pre-trained FFHQ model, run:")
print("")
print("# Option 1: Using the script (recommended)")
print("./scripts/train_aoa_amp.sh 0 ./checkpoints/aoa_amp")
print("")
print("# Option 2: With frozen encoder (faster, less parameters to train)")
print("./scripts/train_aoa_amp.sh 0 ./checkpoints/aoa_amp freeze")
print("")
print("# Option 3: Direct command")
print("python train_aoa_amp.py \\")
print("  --model_config configs/aoa_amp_model_config.yaml \\")
print("  --diffusion_config configs/diffusion_config.yaml \\")
print("  --train_config configs/aoa_amp_train_config.yaml \\")
print("  --gpu 0 \\")
print("  --checkpoint_dir ./checkpoints/aoa_amp \\")
print("  --pretrained_model models/ffhq_10m.pt")
print("")
print("The training will:")
print("✓ Load pre-trained FFHQ weights")
print("✓ Adapt input layer from 3 channels (RGB) to 2 channels (Real/Imag)")
print("✓ Fine-tune on your AoA-Amp data")
print("✓ Save checkpoints every 20 epochs")
print("✓ Use transfer learning for faster convergence")
print("="*70)


