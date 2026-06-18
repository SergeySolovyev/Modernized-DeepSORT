"""Standalone body re-identification system (Additional 9-10 task).

A persistent identity database (iid space) running alongside the DeepSORT tracker
(tid space): kNN search + gallery management + time-window vote + conflict resolution.
"""
from .config import ReidConfig  # noqa: F401
from .identity_db import IdentityDatabase, Identity  # noqa: F401
from .resolver import IdentityResolver  # noqa: F401
from .conflicts import ConflictManager  # noqa: F401
from .pipeline import BodyReidRunner  # noqa: F401
