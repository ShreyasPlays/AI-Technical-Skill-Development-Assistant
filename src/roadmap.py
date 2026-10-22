"""
Roadmap and Learning Recommendation Engine (Phase 1 Extension).

Translates student skill gap diagnostics into actionable, multi-modal learning
roadmaps with:
  1. Priority skill bottlenecks identified for the online assessment gate or target roles.
  2. Structured multi-week sprints (Weeks 1 to 4/8) with targeted weekly milestones.
  3. Curated resources (free courses, interactive practice, project ideas).
  4. Projected assessment pass probability improvement after roadmap completion.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

from src.data_loader import SKILL_COLUMNS, ROLE_REQUIREMENTS, CORE_ASSESSMENT_WEIGHTS


DEFAULT_RESOURCES_PATH = Path(__file__).resolve().parent.parent / "configs" / "learning_resources.yaml"


class RoadmapGenerator:
    """
    Generates personalized learning roadmaps based on student skill gaps.
    """

    def __init__(self, resources_path: Optional[Path] = None):
        path = resources_path or DEFAULT_RESOURCES_PATH
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.resources = yaml.safe_load(f) or {}
        else:
            self.resources = {}

    def get_skill_resource(self, skill_key: str) -> Dict[str, Any]:
        """Fetch resources and topics for a specific technical skill."""
        return self.resources.get(skill_key, {
            "title": skill_key.replace("_", " ").title(),
            "core_topics": [],
            "resources": [],
            "practice": [],
            "project": "Practice hands-on coding and build a capstone project.",
        })

    def generate_assessment_roadmap(
        self,
        current_skills: Dict[str, int],
        target_score_level: int = 4,
    ) -> Dict[str, Any]:
        """
        Generate a focused roadmap to clear the online assessment screening.

        Focuses on the 6 core assessment technical skills weighted in placement screening:
          - programming_skills (weight 0.22)
          - database_fundamentals (weight 0.18)
          - software_development (weight 0.18)
          - troubleshooting_skills (weight 0.15)
          - software_engineering (weight 0.15)
          - networking (weight 0.12)
        """
        gaps = []
        for skill, weight in CORE_ASSESSMENT_WEIGHTS.items():
            curr = current_skills.get(skill, 0)
            target = target_score_level
            gap = max(0, target - curr)
            if gap > 0:
                weighted_impact = gap * weight
                gaps.append({
                    "skill": skill,
                    "title": self.get_skill_resource(skill).get("title", skill),
                    "current": curr,
                    "target": target,
                    "gap": gap,
                    "weight": weight,
                    "weighted_impact": round(weighted_impact, 3),
                    "resource_info": self.get_skill_resource(skill),
                })

        # Sort descending by weighted impact (biggest bottleneck first)
        gaps.sort(key=lambda x: x["weighted_impact"], reverse=True)

        # Build 4-week sprint milestones
        sprints = self._build_sprints(gaps, duration_weeks=4)

        return {
            "type": "assessment_gate_roadmap",
            "target_score_level": target_score_level,
            "total_gap_points": sum(g["gap"] for g in gaps),
            "priority_areas": gaps,
            "sprints": sprints,
        }

    def generate_role_roadmap(
        self,
        current_skills: Dict[str, int],
        target_role: str,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive role-specific roadmap bridging gaps to industry requirements.
        """
        if target_role not in ROLE_REQUIREMENTS:
            raise ValueError(f"Unknown target role: '{target_role}'.")

        reqs = ROLE_REQUIREMENTS[target_role]
        gaps = []

        for skill, target in reqs.items():
            curr = current_skills.get(skill, 0)
            gap = max(0, target - curr)
            if gap > 0:
                gaps.append({
                    "skill": skill,
                    "title": self.get_skill_resource(skill).get("title", skill),
                    "current": curr,
                    "target": target,
                    "gap": gap,
                    "resource_info": self.get_skill_resource(skill),
                })

        # Sort descending by gap magnitude
        gaps.sort(key=lambda x: x["gap"], reverse=True)

        weeks = 6 if len(gaps) > 4 else 4
        sprints = self._build_sprints(gaps, duration_weeks=weeks)

        return {
            "type": "role_roadmap",
            "target_role": target_role,
            "total_gaps": len(gaps),
            "total_gap_points": sum(g["gap"] for g in gaps),
            "priority_areas": gaps,
            "sprints": sprints,
        }

    def _build_sprints(
        self,
        gaps: List[Dict[str, Any]],
        duration_weeks: int = 4,
    ) -> List[Dict[str, Any]]:
        """Distribute prioritized skill gaps across structured weekly sprints."""
        if not gaps:
            return [{
                "week": 1,
                "title": "All Requirements Met",
                "focus": "Ready for Assessment & Mock Interviews",
                "milestones": ["Take full-length mock assessment tests", "Refine resume & portfolio"],
                "resources": [],
            }]

        sprints = []
        # Assign primary focus per week
        for week_idx in range(1, duration_weeks + 1):
            assigned_gap = gaps[(week_idx - 1) % len(gaps)]
            res = assigned_gap.get("resource_info", {})
            topics = res.get("core_topics", [])
            practice = res.get("practice", [])
            project = res.get("project", "Build a small portfolio demonstration project.")

            # Pick a subset of topics and practice for this week
            sprints.append({
                "week": week_idx,
                "skill": assigned_gap["skill"],
                "skill_title": assigned_gap["title"],
                "focus_area": f"Bridge {assigned_gap['title']} (Level {assigned_gap['current']} → {assigned_gap['target']})",
                "key_topics": topics[:3] if topics else ["Fundamentals & Applied Concepts"],
                "practice_targets": practice[:2] if practice else ["Targeted problem sets"],
                "recommended_resources": res.get("resources", [])[:2],
                "milestone_deliverable": f"Week {week_idx} Milestone: Complete practice questions & {project[:70]}...",
            })

        return sprints
