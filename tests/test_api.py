import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app, EmployeeProfile

client = TestClient(app)


def test_root_endpoint():
    """Verify that the GET / endpoint returns 200 and service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "AttritionIQ Employee Attrition Prediction API"
    assert data["status"] == "online"
    assert "/docs" in data["documentation"]


def test_health_endpoint():
    """Verify that the GET /health endpoint returns 200 and reports model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_type"] == "LogisticRegression"
    assert data["classes"] == [0, 1]


def test_predict_endpoint_valid_at_risk():
    """Verify that POST /predict correctly identifies an elevated risk employee."""
    payload = {
        "Age": 28,
        "DailyRate": 800.0,
        "DistanceFromHome": 20.0,
        "HourlyRate": 65.0,
        "MonthlyIncome": 2500.0,
        "MonthlyRate": 14000.0,
        "NumCompaniesWorked": 3,
        "PercentSalaryHike": 11.0,
        "TotalWorkingYears": 4,
        "TrainingTimesLastYear": 2,
        "YearsAtCompany": 2,
        "YearsInCurrentRole": 1,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 1,
        "Education": 3,
        "EnvironmentSatisfaction": 1,
        "JobInvolvement": 1,
        "JobLevel": 1,
        "JobSatisfaction": 1,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 2,
        "StockOptionLevel": 0,
        "WorkLifeBalance": 1,
        "BusinessTravel": "Travel_Frequently",
        "Department": "Sales",
        "EducationField": "Life Sciences",
        "Gender": "Male",
        "JobRole": "Sales Representative",
        "MaritalStatus": "Single",
        "OverTime": "Yes"
    }
    response = client.post("/predict?threshold=0.50", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == 1
    assert data["attrition_label"] == "Yes"
    assert data["attrition_probability"] >= 0.50
    assert data["risk_category"] == "Elevated"
    assert len(data["risk_factors"]) > 0
    assert "stay interview" in data["recommended_action"].lower()


def test_predict_endpoint_valid_retained():
    """Verify that POST /predict correctly identifies a low-risk stable employee."""
    payload = {
        "Age": 48,
        "DailyRate": 900.0,
        "DistanceFromHome": 3.0,
        "HourlyRate": 75.0,
        "MonthlyIncome": 16000.0,
        "MonthlyRate": 18000.0,
        "NumCompaniesWorked": 1,
        "PercentSalaryHike": 15.0,
        "TotalWorkingYears": 22,
        "TrainingTimesLastYear": 3,
        "YearsAtCompany": 15,
        "YearsInCurrentRole": 8,
        "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 8,
        "Education": 4,
        "EnvironmentSatisfaction": 4,
        "JobInvolvement": 4,
        "JobLevel": 4,
        "JobSatisfaction": 4,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 4,
        "StockOptionLevel": 2,
        "WorkLifeBalance": 3,
        "BusinessTravel": "Travel_Rarely",
        "Department": "Research & Development",
        "EducationField": "Medical",
        "Gender": "Female",
        "JobRole": "Manager",
        "MaritalStatus": "Married",
        "OverTime": "No"
    }
    response = client.post("/predict?threshold=0.50", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == 0
    assert data["attrition_label"] == "No"
    assert data["attrition_probability"] < 0.30
    assert data["risk_category"] == "Low"


def test_predict_endpoint_invalid_category():
    """Verify that invalid categorical inputs return 422 Unprocessable Entity."""
    payload = {
        "Age": 30,
        "Department": "NonExistentDepartment",  # Invalid
        "OverTime": "Maybe"                    # Invalid
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_invalid_numerical_bounds():
    """Verify that out-of-bounds numerical inputs return 422 Unprocessable Entity."""
    payload = {
        "Age": 10,  # Below minimum 18
        "Department": "Sales",
        "OverTime": "No"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_vs_direct_pipeline_consistency():
    """Verify that predictions from API match direct pipeline predictions identically."""
    pipeline = joblib.load("models/attrition_model_pipeline.joblib")
    
    test_profile = EmployeeProfile(
        Age=35,
        MonthlyIncome=6000.0,
        OverTime="No",
        StockOptionLevel=1,
        JobSatisfaction=3,
        WorkLifeBalance=3,
        Department="Research & Development",
        JobRole="Research Scientist",
        MaritalStatus="Married",
        YearsAtCompany=6,
        TotalWorkingYears=10
    )
    
    # 1. API prediction
    api_res = client.post("/predict?threshold=0.50", json=test_profile.model_dump())
    assert api_res.status_code == 200
    api_data = api_res.json()
    
    # 2. Direct pipeline prediction
    df = pd.DataFrame([test_profile.model_dump()])
    direct_proba = float(pipeline.predict_proba(df)[0][1])
    
    assert round(api_data["attrition_probability"], 4) == round(direct_proba, 4)
    print("\nAPI probability:", api_data["attrition_probability"], "== Direct pipeline:", round(direct_proba, 4))


if __name__ == "__main__":
    test_root_endpoint()
    test_health_endpoint()
    test_predict_endpoint_valid_at_risk()
    test_predict_endpoint_valid_retained()
    test_predict_endpoint_invalid_category()
    test_predict_endpoint_invalid_numerical_bounds()
    test_api_vs_direct_pipeline_consistency()
    print("\nALL 6 UNIT & CONSISTENCY TESTS PASSED CLEANLY!")
