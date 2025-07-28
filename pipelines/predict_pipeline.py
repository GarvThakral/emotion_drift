from zenml import pipeline
from steps.preprocess_data_pred import preprocessing_data
from steps.fetch_comments import fetch_comments
from steps.predict import predict
from steps.check_result import check_result
from steps.check_drift import plot_drift
@pipeline
def predict_for_yt():
    video_url = "https://www.youtube.com/watch?v=IYDRe94pTOM"
    comments = fetch_comments(video_url)
    (processed_comments,comments_orig) = preprocessing_data(comments)
    (result,comments_orig )= predict("./saved_models/trained_model",processed_comments,comments_orig,5)
    weekNum = check_result(result,comments_orig)
    plot_drift(weekNum)
