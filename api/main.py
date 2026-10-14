"""
AttritionIQ — Intelligent Employee Attrition Prediction System
FastAPI Production Backend REST Service
"""

import os
from pathlib import Path
from typing import List, Literal, Optional
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# ------------------------------------------------------------------------------
# Path Resolution & Global Model State
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "attrition_model_pipeline.joblib"))

# Global model cache to avoid reloading on every request
model_state = {
    "pipeline": None,
    "model_loaded": False,
    "model_type": None,
    "classes": None,
    "load_error": None
}


def load_model():
    """Safely loads the serialized Scikit-Learn production pipeline."""
    if not MODEL_PATH.exists():
        err_msg = f"Model pipeline artifact not found at {MODEL_PATH}"
        model_state["load_error"] = err_msg
        model_state["model_loaded"] = False
        return None
    try:
        pipeline = joblib.load(MODEL_PATH)
        model_state["pipeline"] = pipeline
        model_state["model_loaded"] = True
        clf = pipeline.named_steps.get("classifier", None)
        model_state["model_type"] = type(clf).__name__ if clf else "Unknown"
        model_state["classes"] = getattr(clf, "classes_", [0, 1]).tolist()
        model_state["load_error"] = None
        return pipeline
    except Exception as e:
        model_state["load_error"] = str(e)
        model_state["model_loaded"] = False
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load the model on application startup."""
    load_model()
    yield
    # Cleanup if needed
    model_state["pipeline"] = None


# ------------------------------------------------------------------------------
# FastAPI Application Initialization
# ------------------------------------------------------------------------------
app = FastAPI(
    title="AttritionIQ REST API",
    description="Production Machine Learning Decision-Support API for Employee Attrition Prediction.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit Community Cloud and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Pydantic Schemas matching the exact 30-feature dataset schema
# ------------------------------------------------------------------------------
class EmployeeProfile(BaseModel):
    """
    Complete schema for an employee profile, matching the 30 features
    expected by the AttritionIQ production pipeline.
    """
    model_config = ConfigDict(extra="forbid")

    # Numerical features (14)
    Age: int = Field(default=32, ge=18, le=75, description="Employee Age in years")
    DailyRate: float = Field(default=800.0, ge=0.0, description="Daily compensation rate")
    DistanceFromHome: float = Field(default=8.0, ge=0.0, le=100.0, description="Commuting distance in miles")
    HourlyRate: float = Field(default=65.0, ge=0.0, description="Hourly compensation rate")
    MonthlyIncome: float = Field(default=4500.0, ge=500.0, le=50000.0, description="Monthly base salary ($ USD)")
    MonthlyRate: float = Field(default=14000.0, ge=0.0, description="Monthly billing rate")
    NumCompaniesWorked: int = Field(default=2, ge=0, le=20, description="Number of previous employers")
    PercentSalaryHike: float = Field(default=14.0, ge=0.0, le=100.0, description="Percentage salary hike at last appraisal")
    TotalWorkingYears: int = Field(default=8, ge=0, le=60, description="Total cumulative working experience in years")
    TrainingTimesLastYear: int = Field(default=3, ge=0, le=20, description="Number of training programs attended last year")
    YearsAtCompany: int = Field(default=5, ge=0, le=50, description="Tenure at current company in years")
    YearsInCurrentRole: int = Field(default=2, ge=0, le=40, description="Tenure in current functional role in years")
    YearsSinceLastPromotion: int = Field(default=1, ge=0, le=40, description="Years elapsed since last promotion")
    YearsWithCurrManager: int = Field(default=2, ge=0, le=40, description="Tenure under current reporting manager in years")

    # Ordinal features (9)
    Education: int = Field(default=3, ge=1, le=5, description="Education level (1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor)")
    EnvironmentSatisfaction: int = Field(default=3, ge=1, le=4, description="Workplace environment satisfaction (1=Low, 4=Very High)")
    JobInvolvement: int = Field(default=3, ge=1, le=4, description="Job involvement rating (1=Low, 4=Very High)")
    JobLevel: int = Field(default=2, ge=1, le=5, description="Organizational job hierarchy level (1 to 5)")
    JobSatisfaction: int = Field(default=3, ge=1, le=4, description="Self-reported job satisfaction (1=Low, 4=Very High)")
    PerformanceRating: int = Field(default=3, ge=1, le=4, description="Latest performance rating (3=Excellent, 4=Outstanding)")
    RelationshipSatisfaction: int = Field(default=3, ge=1, le=4, description="Interpersonal relationship satisfaction (1=Low, 4=Very High)")
    StockOptionLevel: int = Field(default=1, ge=0, le=3, description="Equity stock option tier (0=None, 3=Max)")
    WorkLifeBalance: int = Field(default=3, ge=1, le=4, description="Work-life balance rating (1=Bad, 4=Best)")

    # Nominal Categorical features (7)
    BusinessTravel: Literal["Travel_Rarely", "Travel_Frequently", "Non-Travel"] = Field(
        default="Travel_Rarely", description="Frequency of business-related travel"
    )
    Department: Literal["Sales", "Research & Development", "Human Resources"] = Field(
        default="Sales", description="Organizational department"
    )
    EducationField: Literal["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"] = Field(
        default="Life Sciences", description="Primary academic field of study"
    )
    Gender: Literal["Male", "Female"] = Field(default="Male", description="Gender identity")
    JobRole: Literal[
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources"
    ] = Field(default="Sales Executive", description="Job title / role")
    MaritalStatus: Literal["Single", "Married", "Divorced"] = Field(
        default="Married", description="Marital status"
    )
    OverTime: Literal["Yes", "No"] = Field(
        default="No", description="Requirement for compulsory overtime work"
    )


class PredictionResponse(BaseModel):
    """Structured response schema returned by the /predict endpoint."""
    predicted_class: int = Field(description="Binary classification outcome (0 = Retained, 1 = Departed)")
    attrition_label: str = Field(description="Interpreted attrition status: 'Yes' (Depart) or 'No' (Retain)")
    attrition_probability: float = Field(description="Estimated departure probability P(Y=1|x) between 0.0 and 1.0")
    risk_category: str = Field(description="Risk categorization tier ('Low', 'Moderate', or 'Elevated')")
    decision_threshold: float = Field(description="Operating decision threshold used to classify risk")
    risk_factors: List[str] = Field(description="Specific empirical risk signals detected in the profile")
    recommended_action: str = Field(description="Recommended human-in-the-loop retention action")
    model_version: str = Field(default="RandomForest_Tuned_v2.0", description="Model architecture identifier")


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def evaluate_risk_factors(data: dict) -> List[str]:
    """Identifies top empirical risk factors present in the employee profile."""
    factors = []
    if data.get("OverTime") == "Yes":
        factors.append("Compulsory Overtime Active (Primary Turnover Risk)")
    if data.get("StockOptionLevel", 1) == 0:
        factors.append("Zero Stock Options / Equity (Lack of Retention Anchor)")
    if data.get("MonthlyIncome", 5000) < 3500:
        factors.append(f"Low Monthly Income (${data.get('MonthlyIncome'):,.0f} - Below Peer Median)")
    if data.get("JobSatisfaction", 3) <= 2:
        factors.append(f"Depressed Job Satisfaction Rating ({data.get('JobSatisfaction')}/4)")
    if data.get("WorkLifeBalance", 3) <= 2:
        factors.append(f"Low Work-Life Balance Rating ({data.get('WorkLifeBalance')}/4)")
    if data.get("DistanceFromHome", 5) > 15:
        factors.append(f"Long Commute Distance ({data.get('DistanceFromHome'):.0f} miles)")
    if data.get("MaritalStatus") == "Single":
        factors.append("Single Demographic Cohort (Elevated Baseline Mobility)")
    if data.get("YearsAtCompany", 5) <= 2:
        factors.append(f"Early Tenure Hazard Period ({data.get('YearsAtCompany')} years at company)")
    return factors


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/", tags=["General"])
def read_root():
    """Root endpoint providing service identity and documentation links."""
    return {
        "service": "AttritionIQ Employee Attrition Prediction API",
        "version": "2.0.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint reporting model status and runtime readiness."""
    if not model_state["model_loaded"]:
        # Attempt recovery load
        load_model()
    
    is_healthy = model_state["model_loaded"]
    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "healthy" if is_healthy else "degraded",
        "model_loaded": is_healthy,
        "model_type": model_state["model_type"],
        "model_path": str(MODEL_PATH),
        "classes": model_state["classes"],
        "error": model_state["load_error"]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_attrition(
    employee: EmployeeProfile,
    threshold: float = Query(
        default=0.50,
        ge=0.10,
        le=0.90,
        description="Decision sensitivity threshold τ (Default: 0.50, High-Recall: 0.42)"
    )
):
    """
    Validates an employee profile and computes real-time voluntary attrition probability
    using the production Scikit-Learn pipeline.
    """
    pipeline = model_state.get("pipeline")
    if pipeline is None:
        pipeline = load_model()
        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Production model pipeline is not available. Details: {model_state['load_error']}"
            )

    # Convert Pydantic model to a 1-row Pandas DataFrame with exact column names
    data_dict = employee.model_dump()

    # Enforce logical consistency: total working years cannot be less than tenure
    if data_dict["YearsAtCompany"] > data_dict["TotalWorkingYears"]:
        data_dict["TotalWorkingYears"] = data_dict["YearsAtCompany"]

    input_df = pd.DataFrame([data_dict])

    try:
        # Predict class probabilities: classes_ are [0, 1] -> col 1 is Attrition = Yes
        probabilities = pipeline.predict_proba(input_df)[0]
        prob_attrition = float(probabilities[1])

        # Apply calibrated operational threshold
        predicted_class = int(1 if prob_attrition >= threshold else 0)
        attrition_label = "Yes" if predicted_class == 1 else "No"

        # Categorize risk tier
        if prob_attrition < 0.30:
            risk_category = "Low"
        elif prob_attrition < 0.60:
            risk_category = "Moderate"
        else:
            risk_category = "Elevated"

        # Determine risk signals and recommended action
        risk_factors = evaluate_risk_factors(data_dict)
        if predicted_class == 1:
            recommended_action = (
                "Initiate proactive retention playbook: schedule confidential stay interview, "
                "review compensation competitiveness against external benchmarks, and explore "
                "workload / overtime rebalancing."
            )
        else:
            recommended_action = (
                "Maintain standard talent stewardship: continue regular quarterly check-ins "
                "and ongoing professional development opportunities."
            )

        return PredictionResponse(
            predicted_class=predicted_class,
            attrition_label=attrition_label,
            attrition_probability=round(prob_attrition, 4),
            risk_category=risk_category,
            decision_threshold=threshold,
            risk_factors=risk_factors,
            recommended_action=recommended_action,
            model_version="RandomForest_Tuned_v2.0"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction inference failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
