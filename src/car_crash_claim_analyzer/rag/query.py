"""Advanced query construction for policy retrieval."""

import re


class QueryExpander:
    """Generate policy-aware query variants without requiring an LLM."""

    POLICY_INTENTS = [
        "Section I loss or damage to the insured vehicle own damage",
        "Section I indemnify the insured against loss of or damage to the vehicle",
        "own damage accidental loss or damage to the insured vehicle indemnity",
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

        if any(term in lower for term in (
            "damage", "head_lamp", "bumper", "door", "glass", "scratch", "dent",
            "unclassified_damage", "unclassified", "unknown",
        )):
            variants.extend(f"{base} {intent}" for intent in self.POLICY_INTENTS)

        # Always retrieve the governing own-damage provisions so a component
        # name cannot crowd out the policy's primary coverage section.
        variants.extend(self.POLICY_INTENTS[:2])

        for key, terms in self.TERMS.items():
            if key in lower or any(term in lower for term in terms):
                variants.append(f"{base} {' '.join(terms[:3])}")

        return list(dict.fromkeys(v for v in variants if v))


class QueryNormalizer:
    """Normalize whitespace and punctuation before retrieval."""

    def normalize(self, query: str) -> str:
        query = query.replace("_", " ")
        return re.sub(r"\s+", " ", query).strip()
