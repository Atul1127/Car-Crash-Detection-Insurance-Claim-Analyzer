"""Structured claim-field extraction from OCR text."""

import re

from car_crash_claim_analyzer.schemas import ClaimInformation


class ClaimInformationExtractor:
    """Extract common insurance claim fields using tolerant, explainable patterns."""

    FIELD_PATTERNS = {
        "claim_id": [
            r"(?:claim\s*(?:id|number|no\.?|#)|claim\s*reference)\s*[:#-]?\s*([A-Z0-9][A-Z0-9/_-]{2,30})",
        ],
        "policy_number": [
            r"(?:policy\s*(?:number|no\.?|#)|policy\s*id)\s*[:#-]?\s*([A-Z0-9][A-Z0-9/_-]{3,40})",
        ],
        "vehicle_registration": [
            r"(?:vehicle\s*(?:registration|regn|number|no\.?)|registration\s*(?:number|no\.?)|regn)\s*[:#-]?\s*([A-Z0-9 -]{5,18})",
        ],
        "incident_date": [
            r"(?:incident|accident|loss|date\s*of\s*(?:loss|accident|incident))\s*date\s*[:#-]?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
            r"(?:date\s*of\s*(?:loss|accident|incident)|incident\s*date|accident\s*date|loss\s*date)\s*[:#-]?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
        ],
    }

    @staticmethod
    def _clean(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip(" .,:;|\t\n")

    def extract(self, text: str) -> ClaimInformation:
        normalized = " ".join(text.split())
        values: dict[str, str] = {}

        for field, patterns in self.FIELD_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, normalized, flags=re.IGNORECASE)
                if match:
                    values[field] = self._clean(match.group(1))
                    break

        claimant = re.search(
            r"(?:claimant|insured|policyholder|insured\s*name)\s*(?:name)?\s*[:#-]?\s*([A-Za-z][A-Za-z .'-]{1,79}?)(?=\s+(?:policy|claim|vehicle|registration|date|address)\b|$)",
            normalized,
            flags=re.IGNORECASE,
        )
        if claimant:
            values["claimant_name"] = self._clean(claimant.group(1))

        description = re.search(
            r"(?:incident|accident)\s*(?:description|details)\s*[:#-]?\s*(.+?)(?=\s+(?:claim|policy|vehicle)\s*(?:id|number|no\.?)\b|$)",
            normalized,
            flags=re.IGNORECASE,
        )

        return ClaimInformation(
            claim_id=values.get("claim_id"),
            policy_number=values.get("policy_number"),
            claimant_name=values.get("claimant_name"),
            vehicle_registration=values.get("vehicle_registration"),
            incident_date=values.get("incident_date"),
            incident_description=self._clean(description.group(1)) if description else None,
            raw_text=text,
            metadata={"extraction_method": "regex-tolerant"},
        )
