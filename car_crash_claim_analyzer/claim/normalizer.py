"""Normalization helpers for claim information."""

import re

from car_crash_claim_analyzer.schemas import ClaimInformation


class ClaimNormalizer:
    """Normalize extracted values while preserving the original OCR text."""

    def normalize(self, claim: ClaimInformation) -> ClaimInformation:
        if claim.policy_number:
            claim.policy_number = re.sub(r"\s+", "", claim.policy_number).upper()
        if claim.claim_id:
            claim.claim_id = re.sub(r"\s+", "", claim.claim_id).upper()
        if claim.vehicle_registration:
            claim.vehicle_registration = re.sub(r"\s+", " ", claim.vehicle_registration).strip().upper()
        if claim.claimant_name:
            claim.claimant_name = " ".join(claim.claimant_name.split()).title()
        if claim.incident_date:
            claim.incident_date = claim.incident_date.replace(".", "/").replace("-", "/")
        return claim
