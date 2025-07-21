"""
Integration test for the hybrid confidence fusion system.
"""

from src.data_models import TextBlock
from src.confidence_fusion import analyze_document_with_fusion, get_high_confidence_headings

def test_confidence_fusion_integration():
    """Test the confidence fusion system with realistic data."""
    
    # Create sample text blocks representing a document
    text_blocks = [
        # Title
        TextBlock("Research Paper on AI Systems", 1, 16.0, "Arial", True, (100, 50, 400, 80), 18.0),
        
        # Abstract
        TextBlock("Abstract", 1, 14.0, "Arial", True, (100, 120, 200, 140), 16.0),
        TextBlock("This paper presents a comprehensive analysis...", 1, 12.0, "Arial", False, (100, 150, 500, 170), 14.0),
        
        # H1 headings
        TextBlock("1. Introduction", 1, 14.0, "Arial", True, (100, 200, 280, 220), 16.0),
        TextBlock("Artificial intelligence systems have become...", 1, 12.0, "Arial", False, (100, 230, 500, 250), 14.0),
        
        TextBlock("2. Methodology", 2, 14.0, "Arial", True, (100, 100, 300, 120), 16.0),
        TextBlock("Our research methodology consists of...", 2, 12.0, "Arial", False, (100, 130, 500, 150), 14.0),
        
        # H2 headings
        TextBlock("2.1 Data Collection", 2, 13.0, "Arial", True, (120, 180, 320, 200), 15.0),
        TextBlock("We collected data from multiple sources...", 2, 12.0, "Arial", False, (120, 210, 500, 230), 14.0),
        
        TextBlock("2.2 Analysis Framework", 2, 13.0, "Arial", True, (120, 260, 350, 280), 15.0),
        TextBlock("The analysis framework includes...", 2, 12.0, "Arial", False, (120, 290, 500, 310), 14.0),
        
        # H3 heading
        TextBlock("2.2.1 Statistical Methods", 2, 12.5, "Arial", True, (140, 340, 380, 360), 14.5),
        TextBlock("Statistical analysis was performed using...", 2, 12.0, "Arial", False, (140, 370, 500, 390), 14.0),
        
        # H1 heading
        TextBlock("3. Results", 3, 14.0, "Arial", True, (100, 100, 250, 120), 16.0),
        TextBlock("The results demonstrate significant improvements...", 3, 12.0, "Arial", False, (100, 130, 500, 150), 14.0),
        
        # References
        TextBlock("References", 4, 14.0, "Arial", True, (100, 100, 250, 120), 16.0),
        TextBlock("[1] Smith, J. et al. (2023). AI Systems...", 4, 12.0, "Arial", False, (100, 130, 500, 150), 14.0),
    ]
    
    print("Testing confidence fusion system...")
    
    # Test full fusion analysis
    try:
        result = analyze_document_with_fusion(text_blocks)
        
        print(f"✓ Document type detected: {result.document_type.value}")
        print(f"✓ Found {len(result.candidates)} heading candidates")
        print(f"✓ Processing time: {result.performance_metrics['total_processing_time']:.2f}s")
        print(f"✓ Within performance target: {result.performance_metrics['within_target']}")
        
        # Check that we have reasonable results
        assert len(result.candidates) > 0, "Should find some heading candidates"
        assert result.document_type is not None, "Should detect document type"
        
        # Test high-confidence headings
        high_conf_headings = get_high_confidence_headings(text_blocks, confidence_threshold=0.7)
        print(f"✓ Found {len(high_conf_headings)} high-confidence headings")
        
        # Display some results
        print("\nTop heading candidates:")
        for i, candidate in enumerate(sorted(result.candidates, key=lambda c: c.confidence_score, reverse=True)[:5]):
            print(f"  {i+1}. '{candidate.text}' (confidence: {candidate.confidence_score:.3f})")
        
        print("\nFusion system integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_confidence_fusion_integration()