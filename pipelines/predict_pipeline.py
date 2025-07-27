from zenml import pipeline
from steps.preprocess_data_pred import preprocessing_data
from steps.fetch_comments import fetch_comments
from steps.predict import predict
from steps.check_result import check_result
@pipeline
def predict_for_yt():
    video_url = "https://www.youtube.com/watch?v=IYDRe94pTOM"
    comments = fetch_comments(video_url)
    processed_comments = preprocessing_data(comments)
    result = predict("./saved_models/trained_model",processed_comments,5)
    check_result(result)