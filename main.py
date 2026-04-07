"""
OpenEnv DataCleaning Environment — FastAPI Application
Exposes: POST /reset, POST /step, GET /state, GET /tasks, GET /health, GET /validate
"""

from __future__ import annotations

import os
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from env.environment import DataCleaningEnv, TASK_META
from env.models import DataAction, DataObservation, StepResult, EnvironmentState

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenEnv DataCleaning Environment",
    description="Real-world data cleaning & validation environment for AI agents",
    version="1.0.0",
)

_env: DataCleaningEnv = DataCleaningEnv(task_id="task_easy")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "environment": "data-cleaning-env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    tasks = []
    for tid, meta in TASK_META.items():
        tasks.append({"task_id": tid, **meta})
    return {"tasks": tasks}


@app.post("/reset", response_model=DataObservation)
def reset(task_id: str = "task_easy", seed: int = 42):
    global _env
    if task_id not in TASK_META:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: {task_id}. Valid: {list(TASK_META)}"
        )
    _env = DataCleaningEnv(task_id=task_id, seed=seed)
    obs = _env.reset()
    return obs


@app.post("/step", response_model=StepResult)
def step(action: DataAction):
    try:
        result = _env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EnvironmentState)
def state():
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/validate")
def validate():
    """OpenEnv validation endpoint — returns spec compliance info."""
    with open("openenv.yaml", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    return {
        "valid": True,
        "name": spec.get("name"),
        "version": spec.get("version"),
        "tasks": [t["id"] for t in spec.get("tasks", [])],
        "endpoints": ["POST /reset", "POST /step", "GET /state", "GET /tasks", "GET /health", "GET /validate"],
        "observation_type": "DataObservation",
        "action_type": "DataAction",
        "reward_type": "DataReward",
        "spec_compliant": True,
    }


@app.get("/openenv.yaml")
def openenv_yaml():
    with open("openenv.yaml", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, media_type="text/plain")


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
