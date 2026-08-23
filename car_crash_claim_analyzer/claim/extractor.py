"""Structured claim-field extraction from OCR text."""

import re

from car_crash_claim_analyzer.schemas import ClaimInformation


class ClaimInformationExtractor:
    """Extract common insurance claim fields using explainable patterns."""

    PATTERNS = {
        "claim_id": [r"claim\s*(?:id|number|no\.?)\s*[:#-]?\s*([A-Z0-9/-]+)"],
        "policy_number": [r"policy\s*(?:number|no\.?|#)\s*[:#-]?\s*([A-Z0-9/-]+)"],
        "vehicle_registration": [
            r"(?:registration|regn|vehicle\s*no\.?)\s*[:#-]?\s*([A-Z0-9 -]{5,15})"
        ],
        "incident_date": [
            r"(?:incident|accident|loss)\s*date\s*[:#-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        ],
    }

    def extract(self, text: str) -> ClaimInformation:
        normalized = " ".join(text.split())
        values: dict[str, str] = {}

        for field, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, normalized, flags=re.IGNORECASE)
                if match:
                    values[field] = match.group(1).strip()
                    break

        claimant = re.search(
            r"(?:claimant|insured|policyholder)\s*(?:name)?\s*[:#-]?\s*([A-Za-z .'-]{2,80})",
            normalized,
            flags=re.IGNORECASE,
        )
        if claimant:
            values["claimant_name"] = claimant.group(1).strip()

        description = re.search(
            r"(?:incident|accident)\s*(?:description|details)\s*[:#-]?\s*(.+?)(?:\s+(?:claim|policy|vehicle)\s*(?:id|number|no\.?))",
            normalized,
            flags=re.IGNORECASE,
        )

        return ClaimInformation(
            claim_id=values.get("claim_id"),
            policy_number=values.get("policy_number"),
            claimant_name=values.get("claimant_name"),
            vehicle_registration=values.get("vehicle_registration"),
            incident_date=values.get("incident_date"),
            incident_description=description.group(1).strip() if description else None,
            raw_text=text,
            metadata={"extraction_method": "regex-baseline"},
        )
