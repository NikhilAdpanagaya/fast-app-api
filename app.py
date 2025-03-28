from typing import Union
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # Validators

from mangum import Mangum

fast_app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))
# The logistic regression model was used as it was determined to be the best fit based on evaluation.
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# The text type validation
class TextRequest(BaseModel):
    text: str


@fast_app.post("/predict/")
def predict_category(request: TextRequest):
    description = request.text.strip()
    if not description:
        raise HTTPException(status_code=204, detail="Text cannot be empty")
    input_data = vectorizer.transform([description])
    prediction = model.predict(input_data)[0]

    return {"prediction": prediction}


handler = Mangum(fast_app)
