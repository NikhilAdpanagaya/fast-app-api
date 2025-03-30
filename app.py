from typing import Union
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # Validators

from mangum import Mangum

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))
# The logistic regression model was used as it was determined to be the best fit based on evaluation.
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# The text type validation
class TextRequest(BaseModel):
    ticket_query: str

class TicketCategoryResponse(BaseModel):
    category: str


@app.post("/classify-ticket/", response_model=TicketCategoryResponse)
def predict_category(request: TextRequest):
    description = request.ticket_query.strip()
    if not description:
        raise HTTPException(status_code=204, detail="Text cannot be empty")
    input_data = vectorizer.transform([description])
    prediction = model.predict(input_data)[0]

    return TicketCategoryResponse(category=prediction)


handler = Mangum(app)
