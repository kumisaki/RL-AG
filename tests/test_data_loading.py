"""Integration-style tests covering TAGAPT data ingestion."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import PolicyRepository, TechniqueRepository, TechniqueInstanceLibrary  # type: ignore  # noqa: E402
from data.models import EntityAlphabet, TechniqueInstanceSet  # type: ignore  # noqa: E402


DATA_ROOT = ROOT / "data"


def test_policy_repository_loads_valid_entities() -> None:
    repo = PolicyRepository(DATA_ROOT)
    policies = list(repo.iter_policies())
    assert policies, "expected at least one policy definition"
    alphabet = set(EntityAlphabet)
    for policy in policies:
        for triple in policy.triples:
            assert triple.subject in alphabet
            assert triple.obj in alphabet


def test_technique_repository_aligns_with_policies() -> None:
    policy_repo = PolicyRepository(DATA_ROOT)
    policy_keys = {policy.key for policy in policy_repo.iter_policies()}
    tech_repo = TechniqueRepository(DATA_ROOT)
    mappings = list(tech_repo.iter_mappings())
    assert mappings, "expected at least one technique mapping"
    for mapping in mappings:
        assert mapping.key in policy_keys
        assert mapping.technique_ids
        for technique in mapping.technique_ids:
            assert technique, "technique identifiers should be non-empty"


def test_instance_library_covers_known_techniques() -> None:
    tech_repo = TechniqueRepository(DATA_ROOT)
    instance_lib = TechniqueInstanceLibrary(DATA_ROOT)
    first_mapping = next(tech_repo.iter_mappings())
    technique_id = first_mapping.technique_ids[0]
    instances = instance_lib.instances_for(technique_id)
    assert isinstance(instances, TechniqueInstanceSet)
    # ensure that at least one entity type has concrete samples to draw from
    assert any(instances.get(entity_type) for entity_type in EntityAlphabet)
