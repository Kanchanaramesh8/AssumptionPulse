from datetime import date, datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    VALID = "VALID"          # Assumption holds true / Decision is safe
    AT_RISK = "AT_RISK"      # Downstream decision impacted by weak/broken parent
    INVALID = "INVALID"      # Assumption broken or expired
    ARCHIVED = "ARCHIVED"    # Replaced by an alternative decision


class Assumption(BaseModel):
    id: str = Field(..., description="Unique identifier, e.g., A1")
    statement: str = Field(..., min_length=5, description="The underlying premise or fact")
    category: str = Field(default="Technical", description="Technical, Financial, Legal, or Operational")
    created_at: date = Field(default_factory=date.today)
    valid_until: Optional[date] = Field(default=None, description="Expiry date for temporal validation")
    scope: str = Field(default="Global", description="Applicable boundary, e.g., '< 10k users/month'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Probability that assumption still holds")
    status: NodeStatus = Field(default=NodeStatus.VALID)


class Decision(BaseModel):
    id: str = Field(..., description="Unique identifier, e.g., D1")
    title: str = Field(..., min_length=3, description="Short name of the decision")
    description: str = Field(..., description="Detailed rationale of what was decided")
    created_at: date = Field(default_factory=date.today)
    criticality: float = Field(default=0.8, ge=0.1, le=1.0, description="Business/system impact if this fails")
    status: NodeStatus = Field(default=NodeStatus.VALID)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Calculated dynamically by ADIE")


class DependencyEdge(BaseModel):
    source_id: str = Field(..., description="Parent Assumption or Decision ID")
    target_id: str = Field(..., description="Dependent Decision ID")
    weight: float = Field(default=1.0, ge=0.1, le=1.0, description="Strength of dependency (0.1 to 1.0)")
    rationale: str = Field(default="", description="Why target depends on source")


class ProjectGraphData(BaseModel):
    project_name: str
    assumptions: list[Assumption] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    dependencies: list[DependencyEdge] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.now)