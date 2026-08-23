"""Advanced query construction for policy retrieval."""

import re


class QueryExpander:
    """Generate policy-aware query variants without requiring an LLM."""

    POLICY_INTENTS = [
        "own damage to the insured vehicle accidental loss or damage Section I",
        "indemnity for loss or damage to the vehicle insured vehicle repair claim",
        "conditions exclusions limitations applicable to loss or damage to the insured vehicle",
        "depreciation excess deductible repair estimate claim settlement vehicle damage",
    ]

    TERMS = {
        "damage": ["damage", "loss", "repair", "accidental damage"],
        "coverage": ["coverage", "covered", "indemnity", "claim eligibility"],
        "depreciation": ["depreciation", "deduction", "wear and tear"],
        "exclusion": ["exclusion", "excluded", "limitation", "not covered"],
    }

    def expand(self, query: str) -> list[str]:
        base = query.strip()
        if not base:
            return []

        variants = [base]
        lower = base.lower()

        # Vehicle-component queries must be broadened to the policy's actual
        # own-damage/loss-and-damage language. A policy normally does not list
        # every component (e.g. headlamp, bumper) separately.
        if any(term in lower for term in ("damage", "head_lamp", "bumper", "door", "glass", "scratch", "dent")):
            variants.extend(f"{base} {intent}" for intent in self.POLICY_INTENTS)

        for key, terms in self.TERMS.items():
            if key in lower or any(term in lower for term in terms):
                variants.append(f"{base} {' '.join(terms[:3])}")

        return list(dict.fromkeys(v for v in variants if v))


class QueryNormalizer:
    """Normalize whitespace and punctuation before retrieval."""

    def normalize(self, query: str) -> str:
        query = query.replace("_", " ")
        return re.sub(r"\s+", " ", query).strip()
