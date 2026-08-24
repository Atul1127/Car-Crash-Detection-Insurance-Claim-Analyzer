from car_crash_claim_analyzer.claim.extractor import ClaimInformationExtractor


def test_extract_common_claim_fields_from_form_text():
    text = """
    Policy No: PC-2026/ABC123
    Claim Number: CLM-00981
    Insured Name: Rahul Kumar
    Vehicle Registration: DL 01 AB 1234
    Accident Date: 18/08/2026
    """

    claim = ClaimInformationExtractor().extract(text)

    assert claim.policy_number == "PC-2026/ABC123"
    assert claim.claim_id == "CLM-00981"
    assert claim.claimant_name == "Rahul Kumar"
    assert claim.vehicle_registration == "DL 01 AB 1234"
    assert claim.incident_date == "18/08/2026"


def test_extract_common_date_variants():
    claim = ClaimInformationExtractor().extract("Date of Loss: 18-08-2026")
    assert claim.incident_date == "18-08-2026"
