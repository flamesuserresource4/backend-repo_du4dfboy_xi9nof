"""
Database Schemas for LooksMax (healthy, habit-based appearance improvement)

Each Pydantic model name maps to a MongoDB collection using the lowercased
class name (handled by the helper functions that accept explicit collection names).
"""

from pydantic import BaseModel, Field
from typing import Optional, List

class Userprofile(BaseModel):
    """
    Collection: "userprofile"
    Stores lightweight preferences to tailor recommendations.
    """
    name: Optional[str] = Field(None, description="Display name (optional)")
    email: Optional[str] = Field(None, description="Email (optional, no auth)")
    skin_type: Optional[str] = Field(
        None, description="Skin type such as normal, oily, dry, combination, sensitive"
    )
    hair_type: Optional[str] = Field(
        None, description="Hair type such as straight, wavy, curly, coily"
    )
    style_vibe: Optional[str] = Field(
        None, description="Style preference such as classic, streetwear, minimal, preppy"
    )

class Routine(BaseModel):
    """
    Collection: "routine"
    A named routine consisting of simple steps a user can follow.
    """
    title: str = Field(..., description="Routine name, e.g., Morning Skincare")
    steps: List[str] = Field(default_factory=list, description="Ordered steps")
    category: str = Field(..., description="skin | hair | style | fitness | sleep | confidence")
    owner_email: Optional[str] = Field(None, description="Optional owner email for grouping")

class Tip(BaseModel):
    """
    Collection: "tip"
    Simple, positive, evidence-based suggestions.
    """
    category: str = Field(..., description="skin | hair | style | fitness | sleep | confidence")
    title: str
    body: str
    tags: List[str] = Field(default_factory=list)
