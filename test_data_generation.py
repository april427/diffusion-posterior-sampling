import numpy as np

# Original data generation (kept for reference)  
x_distance = 100 # meter
y_distance = 100 # meter

grid = 0.1 # meter
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
print(f"Sample values: {aoa_amp[:3]}")

# Convert to 2D array
aoa_amp_array = np.array(aoa_amp).reshape(num_x, num_y)
print(f"Reshaped array: {aoa_amp_array.shape}")
print(f"Real part range: [{np.real(aoa_amp_array).min():.6f}, {np.real(aoa_amp_array).max():.6f}]")
print(f"Imaginary part range: [{np.imag(aoa_amp_array).min():.6f}, {np.imag(aoa_amp_array).max():.6f}]")

print("\n" + "="*80)
print("SETUP COMPLETE - READY FOR TRAINING WITH PRE-TRAINED MODEL")
print("="*80)
print("Your pre-trained model: models/ffhq_10m.pt")
print("")
print("NEXT STEPS:")
print("1. First, install the required dependencies:")
print("   pip install torch torchvision torchaudio")
print("   pip install -r requirements.txt")
print("")
print("2. Then start training with the pre-trained FFHQ model:")
print("")
print("   # Option A: Using the script (recommended)")
print("   ./scripts/train_aoa_amp.sh 0 ./checkpoints/aoa_amp")
print("")
print("   # Option B: With frozen encoder (faster, less parameters to train)")
print("   ./scripts/train_aoa_amp.sh 0 ./checkpoints/aoa_amp freeze")
print("")
print("   # Option C: Direct command")
print("   python3 train_aoa_amp.py \\")
print("     --model_config configs/aoa_amp_model_config.yaml \\")
print("     --diffusion_config configs/diffusion_config.yaml \\")
print("     --train_config configs/aoa_amp_train_config.yaml \\")
print("     --gpu 0 \\")
print("     --checkpoint_dir ./checkpoints/aoa_amp \\")
print("     --pretrained_model models/ffhq_10m.pt")
print("")
print("TRAINING FEATURES:")
print("✓ Loads pre-trained FFHQ diffusion model")
print("✓ Adapts input layer from 3 channels (RGB) to 2 channels (Real/Imag)")
print("✓ Fine-tunes on your AoA-Amp data")
print("✓ Uses transfer learning for faster convergence")
print("✓ Saves checkpoints every 20 epochs")
print("✓ Supports mixed precision training")
print("✓ Includes validation monitoring")
print("="*80)