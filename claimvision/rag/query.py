"""Advanced query construction for policy retrieval."""

import re


class QueryExpander:
    """Generate deterministic query variants without requiring an LLM."""

    TERMS = {
        "damage": ["damage", "loss", "repair", "accidental damage"],
        "coverage": ["coverage", "covered", "indemnity", "claim eligibility"],
        "depreciation": ["depreciation", "deduction", "wear and tear"],
        "exclusion": ["exclusion", "excluded", "limitation", "not covered"],
    }

    def expand(self, query: str) -> list[str]:
        variants = [query.strip()]
        lower = query.lower()
        for key, terms in self.TERMS.items():
            if key in lower or any(term in lower for term in terms):
                variants.append(f"{query} {' '.join(terms[:3])}")
        return list(dict.fromkeys(v for v in variants if v))


class QueryNormalizer:
    """Normalize whitespace and remove noisy punctuation before retrieval."""

    def normalize(self, query: str) -> str:
        return re.sub(r"\s+", " ", query).strip()
