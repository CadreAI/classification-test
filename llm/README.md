# Loan Application Classification using LLMs - Interview Task

**Overview:**  
You are given a tabular dataset representing processed mortgage application data. Your task is to implement a loan approval classifier using Large Language Models (LLMs). Each row represents a mortgage application with various features and a binary outcome indicating loan approval status. You'll need to create a system that leverages LLMs to predict whether a loan should be approved or denied.

**Data Description (mortgage_training_data.csv):**  
The CSV file contains the following columns (already cleaned):
- `application_id` (int): unique identifier
- `annual_income` (float)
- `loan_amount` (float)
- `debt_to_income_ratio` (float)
- `region` (string): categorical feature, one of {"West", "East", "North", "South"}
- `loan_status` (string): target variable, either "approved" or "denied"
- `credit_score` (int)
- `employment_length` (float): in years
- `property_type` (string): type of property

**What You Need to Do:**

1. **Data Loading and Preparation:**
   - Load the mortgage_training_data.csv into a DataFrame
   - Prepare the data for LLM consumption
   - Split into train/test sets for evaluation

2. **LLM Integration:**
   - Set up OpenAI API integration
   - Create structured output handling using Pydantic models
   - Implement proper error handling for API calls

3. **Prompt Engineering:**
   - Design an effective prompt structure
   - Include few-shot examples for better performance
   - Ensure consistent output formatting

4. **Model Implementation:**
   - Create a prediction pipeline
   - Handle batch processing for multiple applications
   - Implement proper async/await patterns for API calls

5. **Evaluation:**
   - Calculate standard metrics (accuracy, AUC)
   - Compare confidence scores with predictions
   - Analyze model explanations

6. **Deliverables:**
   - A Python script (`llm_classifier.py`) that:
     - Loads and prepares data
     - Implements LLM-based classification
     - Evaluates and prints metrics
     - Logs predictions and explanations

Here's a boilerplate to get you started:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from openai import OpenAI
import asyncio
from typing import List, Dict, Tuple, Literal
from pydantic import BaseModel
import time

class LoanDecision(BaseModel):
    """Output schema for loan decisions"""
    decision: Literal["approved", "denied"]
    confidence: float
    explanation: str

class LLMLoanClassifier:
    def __init__(self, api_key: str, model: str = "gpt-4-0125-preview"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def _create_prompt(self, loan_data: Dict) -> str:
        """Create a standardized prompt for the LLM"""
        # TODO: Implement prompt creation
        pass

    async def _get_prediction(self, loan_data: Dict) -> LoanDecision:
        """Get prediction from LLM for a single loan application"""
        # TODO: Implement LLM prediction
        pass

    async def predict_batch(self, df: pd.DataFrame, batch_size: int = 5) -> List[LoanDecision]:
        """Predict multiple loan applications in batches"""
        # TODO: Implement batch prediction
        pass

def evaluate_llm_predictions(y_true: pd.Series, predictions: List[LoanDecision]) -> Dict:
    """Evaluate LLM predictions"""
    # TODO: Implement evaluation metrics
    pass

async def main():
    # Load data
    df = pd.read_csv("mortgage_training_data.csv")
    
    # Prepare target and features
    y = df['loan_status'].map({'approved': 1, 'denied': 0})
    X = df.drop(columns=['application_id', 'loan_status'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # TODO: Initialize classifier and get predictions
    # TODO: Evaluate and print results

if __name__ == "__main__":
    asyncio.run(main())
```

**What We're Looking For:**
- Proper LLM integration and error handling
- Effective prompt engineering
- Understanding of async programming
- Proper evaluation methodology
- Clear code organization and documentation

**Time Limit:**
Expected completion time is 1 hour. Feel free to ask questions during the implementation.

**Evaluation Criteria:**
1. Code quality and organization
2. Prompt engineering effectiveness
3. Error handling and robustness
4. Understanding of LLM vs traditional ML evaluation
5. Implementation of proper async patterns