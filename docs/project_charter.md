# Biohub Cell Tracking Project Charter

---

# 1. Project Vision

Build a modular, production-quality AI framework for the Biohub Cell Tracking Challenge.

The project must prioritize correctness, readability, reproducibility and long-term maintainability over rapid prototyping.

---

# 2. Team Roles

## Lead Architect

Responsible for:

- overall architecture
- sprint planning
- competition analysis
- algorithm selection
- design decisions
- code review
- architecture review
- quality control
- roadmap planning

No implementation decisions may override the Lead Architect.

---

## Implementation Engineer

Responsible for:

- implementing specifications
- writing code
- fixing bugs
- writing unit tests
- refactoring when requested

Implementation Engineer MUST NOT

- redesign architecture
- invent algorithms
- change project structure
- change sprint scope
- analyze the competition
- make architectural decisions

---

# 3. Development Workflow

Every sprint must follow this order.

1. Research
2. Architecture Design
3. Specification
4. Implementation
5. Review
6. Testing
7. Git Commit
8. GitHub Push

No step may be skipped.

---

# 4. Architecture Principles

- Modular design
- Single responsibility
- Explicit interfaces
- Reproducible experiments
- Deterministic pipelines
- Clear separation of data, models and training
- Documentation before implementation

---

# 5. Coding Standards

- Python 3.11
- Type hints
- Docstrings
- Unit tests
- Logging
- Clean code
- PEP8 compliance

---

# 6. Git Rules

One sprint equals one commit.

Every commit must have a meaningful message.

Every completed sprint must be pushed to GitHub.

No datasets may be committed.

Temporary files must remain ignored.

---

# 7. Documentation Rules

Every architecture decision must be documented inside the docs/ directory before implementation.

Documentation is considered the source of truth.

---

# 8. Sprint Policy

A sprint has exactly one objective.

No feature expansion during a sprint.

No implementation outside the agreed specification.

---

# 9. Long-term Roadmap

Phase 1
Foundation

Completed

- Dataset Reader
- GEFF Reader
- Dataset Interface
- Torch Dataset
- Transform Pipeline
- DataLoader

Phase 2

Model Development

- Detection
- Tracking
- Division
- Submission

Phase 3

Optimization

- Performance
- Metrics
- Hyperparameter tuning

Phase 4

Competition Submission

- Final pipeline
- Validation
- Submission

---

# 10. Final Principle

The Lead Architect owns the design.

The Implementation Engineer owns the implementation.

The architecture always has priority over implementation speed.
