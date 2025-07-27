from zenml import step
import tensorflow as tf

# Emotion labels from GoEmotions dataset
labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

@step
def check_result(result, comments_orig):
    # Ensure result is a Tensor
    if isinstance(result, list):
        result_tensor = tf.stack(result)
    else:
        result_tensor = result

    # Apply threshold to get binary predictions
    binary_predictions = tf.cast(result_tensor > 0.2, dtype=tf.int32).numpy()

    # Iterate over each prediction and comment
    for i, prediction in enumerate(binary_predictions):
        print(f"\nComment: {comments_orig[i]}")
        predicted_emotions = [labels[j] for j, val in enumerate(prediction) if val == 1]
        
        if predicted_emotions:
            print("Predicted Emotions:", ", ".join(predicted_emotions))
        else:
            print("Predicted Emotions: None")

