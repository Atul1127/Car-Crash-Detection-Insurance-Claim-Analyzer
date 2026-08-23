"""Validation rules for extracted claim information."""

import re

from car_crash_claim_analyzer.schemas import ClaimInformation


class ClaimValidator:
    """Flag suspicious or incomplete extracted fields without silently changing them."""

    def validate(self, claim: ClaimInformation) -> list[str]:
        warnings: list[str] = []

        if not claim.policy_number:
            warnings.append("Policy number was not extracted.")
        if not claim.claim_id:
            warnings.append("Claim ID was not extracted.")
        if claim.vehicle_registration and not re.fullmatch(
            r"[A-Z0-9 -]{5,15}", claim.vehicle_registration, flags=re.IGNORECASE
        ):
            warnings.append("Vehicle registration format should be manually verified.")
        if claim.incident_date and not re.fullmatch(
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", claim.incident_date
        ):
            warnings.append("Incident date format should be manually verified.")

        return warnings
