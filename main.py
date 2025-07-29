import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("CUDA available:", tf.test.is_built_with_cuda())
print("GPU available:", tf.test.is_gpu_available())

# List all devices
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(f"  {device}")

from pipelines.train_pipeline import train_pipeline
train_pipeline()