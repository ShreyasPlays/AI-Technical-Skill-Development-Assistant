"""
Live smoke-test and demonstration client for the FastAPI Inference Service (Phase 2).

Usage:
    python test_service.py [optional-base-url]
"""

import json
import sys
from typing import Dict
import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def run_smoke_test(base_url: str = DEFAULT_BASE_URL):
    print("=" * 70)
    print(" AI Technical Skill Development Assistant - API Smoke Test")
    print(f" Target Service: {base_url}")
    print("=" * 70)

    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        # 1. Root & Health Check
        print("\n[1/5] Checking Service Health...")
        health_resp = client.get("/health")
        print(f"Status Code: {health_resp.status_code}")
        print(json.dumps(health_resp.json(), indent=2))
        assert health_resp.status_code == 200

        # 2. Model Info & Registry
        print("\n[2/5] Querying Model Registry Lineage...")
        info_resp = client.get("/model/info")
        print(f"Status Code: {info_resp.status_code}")
        print(json.dumps(info_resp.json(), indent=2))
        assert info_resp.status_code == 200

        # 3. Predict Pass Scenario (21st-Century Skills)
        print("\n[3/5] Testing 21st-Century Skill Profile (High Proficiency -> Predict Pass)...")
        strong_profile = {
            "ai_ml": 5,
            "data_science": 5,
            "software_development": 5,
            "programming_skills": 5,
            "cyber_security": 4,
            "distributed_computing_systems": 4,
            "database_fundamentals": 4,
            "software_engineering": 4,
            "troubleshooting_skills": 4,
            "technical_communication": 4,
        }
        predict_payload = {
            "student_id": "DEMO_STUDENT_001",
            "skills": strong_profile,
            "target_role": "AI ML Specialist",
        }
        pred_resp = client.post("/predict", json=predict_payload)
        print(f"Status Code: {pred_resp.status_code}")
        print(json.dumps(pred_resp.json(), indent=2))
        assert pred_resp.status_code == 200

        # 4. Predict Fail Scenario (Low Proficiency -> Predict Fail)
        print("\n[4/5] Testing 21st-Century Skill Profile (Low Proficiency -> Predict Fail)...")
        weak_profile = {
            "ai_ml": 1,
            "data_science": 1,
            "software_development": 1,
            "programming_skills": 1,
            "cyber_security": 1,
            "distributed_computing_systems": 1,
            "database_fundamentals": 1,
            "software_engineering": 1,
            "troubleshooting_skills": 1,
            "technical_communication": 1,
        }
        pred_weak_payload = {
            "student_id": "DEMO_STUDENT_002",
            "skills": weak_profile,
        }
        weak_resp = client.post("/predict", json=pred_weak_payload)
        print(f"Status Code: {weak_resp.status_code}")
        print(json.dumps(weak_resp.json(), indent=2))
        assert weak_resp.status_code == 200

        # 5. Roadmap Generation
        print("\n[5/5] Requesting Targeted Learning Roadmap...")
        roadmap_payload = {
            "student_id": "DEMO_STUDENT_002",
            "skills": weak_profile,
            "target_role": "Software Developer",
        }
        roadmap_resp = client.post("/roadmap", json=roadmap_payload)
        print(f"Status Code: {roadmap_resp.status_code}")
        res_data = roadmap_resp.json()
        print(f"Target Role: {res_data.get('target_role')}")
        print(f"Average Skill Gap: {res_data.get('average_skill_gap')}")
        print(f"Readiness: {res_data.get('role_readiness_status')}")
        print("Sprints Generated:", len(res_data.get("recommended_learning_plan", {}).get("sprints", [])))
        assert roadmap_resp.status_code == 200

    print("\n" + "=" * 70)
    print(" ALL 5 SMOKE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    try:
        run_smoke_test(url)
    except Exception as e:
        print(f"\n[ERROR] Smoke test could not reach service at {url}: {e}")
        print("Make sure the API server is running (`run_api.bat` or `python -m uvicorn src.api.app:app`).")

