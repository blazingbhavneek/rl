from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    target_function: str
    task_description: str
    detailed_steps: list[str]
    required_functions: list[str]
    preconditions: list[str] = Field(default_factory=list)
    cleanup_steps: list[str] = Field(default_factory=list)
    expected_behavior: str = ""
    io_placeholder: str = Field(
        default=(
            "/* IO test case placeholder — single process, 2 files\n"
            "   Will be filled in when runtime IO harness is ready. */"
        )
    )
    test_case_commented: str = Field(
        default="",
        description="Commented-out test case skeleton for future IO verification",
    )


class TaskDesignerOutput(BaseModel):
    spec: TaskSpec
    detailed_prompt: str
    vague_prompt: str


class MultiTaskOutput(BaseModel):
    function_name: str
    tasks: list[TaskDesignerOutput]


class SFTPair(BaseModel):
    function_name: str
    difficulty: str  # b0, b1, b2, b3, b4, b4_vague
    prompt: str
    code: str
    compiled: bool
    compiler_output: str = ""
    vague_prompt: Optional[str] = None  # b4 only


class VerifyResult(BaseModel):
    success: bool
    compiler_output: str
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
