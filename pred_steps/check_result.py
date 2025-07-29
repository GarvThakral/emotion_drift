from zenml import step
import tensorflow as tf
import numpy as np
import pandas as pd

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

    print(binary_predictions)
    sumArr = [0]*28
    for x in binary_predictions:
        sumArr = np.add(sumArr,x) 

    label_arr = [[labels[index] for index,y in enumerate(x) if y == 1] for x in binary_predictions]
    print(label_arr)

    print(sumArr)

    # Create DataFrame
    week1_df = pd.DataFrame({
        "emotion": sumArr
    })

    # Save to CSV
    week1_df.to_csv("./data/week2_emotions.csv", index=False)

    # # Iterate over each prediction and comment
    # for i, prediction in enumerate(binary_predictions):
    #     print(f"\nComment: {comments_orig[i]}")
    #     predicted_emotions = [labels[j] for j, val in enumerate(prediction) if val == 1]
        
    #     if predicted_emotions:
    #         print("Predicted Emotions:", ", ".join(predicted_emotions))
    #     else:
    #         print("Predicted Emotions: None")
    return 1

