from django.db import models

class PredictionLog(models.Model):
    input_data = models.TextField(help_text="Input data for the prediction")
    predicted_output = models.TextField(help_text="The predicted result")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction {self.id} at {self.created_at}"
