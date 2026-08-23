def test_core_imports():
    from car_crash_claim_analyzer.pipeline import CarCrashClaimAnalyzerPipeline
    from car_crash_claim_analyzer.schemas import ClaimInformation
    from car_crash_claim_analyzer.vision.detector import DamageDetector
    from car_crash_claim_analyzer.rag.pipeline import PolicyRAGPipeline
    from car_crash_claim_analyzer.decision.pipeline import ClaimDecisionPipeline
    from car_crash_claim_analyzer.claim.pipeline import ClaimDocumentPipeline

    assert CarCrashClaimAnalyzerPipeline
    assert ClaimInformation
    assert DamageDetector
    assert PolicyRAGPipeline
    assert ClaimDecisionPipeline
    assert ClaimDocumentPipeline
