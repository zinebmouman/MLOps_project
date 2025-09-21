from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    sepal_length: float = Field(..., ge=0)
    sepal_width: float = Field(..., ge=0)
    petal_length: float = Field(..., ge=0)
    petal_width: float = Field(..., ge=0)

    def as_list(self) -> List[float]:
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]

class PredictResponse(BaseModel):
    label: int
    proba: list
