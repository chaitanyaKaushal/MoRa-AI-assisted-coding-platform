from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from stages import stage_3_formalize, stage_4_oracle, stage_5_reference, stage_6_execute_user
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InitRequest(BaseModel):
    problem_text: str

class ExecuteRequest(BaseModel):
    code: str
    test_suite: list

@app.post("/generate-problem")
def generate_problem(req: InitRequest):
    try:
        # 1. Formalize
        spec = stage_3_formalize(req.problem_text)
        
        # 2. Generate Inputs
        oracle_inputs = stage_4_oracle(spec)
        
        # 3. Generate Solution & Suite
        test_suite, ref_code = stage_5_reference(spec, oracle_inputs)
        
        return {
            "spec": spec,
            "test_suite": test_suite,
            "reference_code": ref_code
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute")
def execute_code(req: ExecuteRequest):
    results = stage_6_execute_user(req.code, req.test_suite)
    return {"results": results}