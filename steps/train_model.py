from zenml import step
from transformers import TFDistilBertForSequenceClassification
import transformers
@step
def train_model(model:transformers.models.distilbert.modeling_tf_distilbert.TFDistilBertForSequenceClassification,tf_ds):
    model.fit(
        x=tf_ds,
        batch_size=32,
        epochs=1
    ) # type: ignore
    return model