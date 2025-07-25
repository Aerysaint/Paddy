"""
Unit tests for the parser module - PDF text extraction and block formation.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from parser import (
    extract_words_from_pdf, extract_blocks, merge_adjacent_words,
    compute_block_features, compute_page_line_height, detect_bold_italic,
    detect_number_prefix, clean_duplicated_characters, merge_numbered_sections,
    _is_number_prefix, _should_merge_with_next
)


class TestExtractWordsFromPdf:
    """Test PDF word extraction functionality."""
    
    @patch('parser.pdfplumber.open')
    def test_extract_words_basic(self, mock_open):
        """Test basic word extraction from PDF."""
        # Mock PDF structure
        mock_page = Mock()
        mock_page.extract_words.return_value = [
            {
                'text': 'Hello', 'x0': 10, 'top': 100, 'x1': 30, 'bottom': 110,
                'fontname': 'Arial', 'size': 12.0, 'adv': 20.0
            },
            {
                'text': 'World', 'x0': 35, 'top': 100, 'x1': 55, 'bottom': 110,
                'fontname': 'Arial', 'size': 12.0, 'adv': 20.0
            }
        ]
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        words = extract_words_from_pdf('test.pdf')
        
        assert len(words) == 2
        assert words[0]['text'] == 'Hello'
        assert words[0]['page'] == 1  # 1-based page numbering
        assert words[0]['x0'] == 10
        assert words[0]['y0'] == 100  # top -> y0
        assert words[0]['y1'] == 110  # bottom -> y1
        assert words[0]['fontname'] == 'Arial'
        assert words[0]['size'] == 12.0
        
        assert words[1]['text'] == 'World'
        assert words[1]['page'] == 1
    
    @patch('parser.pdfplumber.open')
    def test_extract_words_multiple_pages(self, mock_open):
        """Test word extraction from multiple pages."""
        # Mock first page
        mock_page1 = Mock()
        mock_page1.extract_words.return_value = [
            {'text': 'Page1', 'x0': 10, 'top': 100, 'x1': 30, 'bottom': 110,
             'fontname': 'Arial', 'size': 12.0, 'adv': 20.0}
        ]
        
        # Mock second page
        mock_page2 = Mock()
        mock_page2.extract_words.return_value = [
            {'text': 'Page2', 'x0': 10, 'top': 100, 'x1': 30, 'bottom': 110,
             'fontname': 'Arial', 'size': 12.0, 'adv': 20.0}
        ]
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        words = extract_words_from_pdf('test.pdf')
        
        assert len(words) == 2
        assert words[0]['text'] == 'Page1'
        assert words[0]['page'] == 1
        assert words[1]['text'] == 'Page2'
        assert words[1]['page'] == 2
    
    @patch('parser.pdfplumber.open')
    def test_extract_words_missing_attributes(self, mock_open):
        """Test handling of missing font attributes."""
        mock_page = Mock()
        mock_page.extract_words.return_value = [
            {
                'text': 'Test', 'x0': 10, 'top': 100, 'x1': 30, 'bottom': 110
                # Missing fontname, size, adv
            }
        ]
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        words = extract_words_from_pdf('test.pdf')
        
        assert len(words) == 1
        assert words[0]['fontname'] == ''  # Default empty string
        assert words[0]['size'] == 0.0  # Default 0.0
        assert words[0]['adv'] == 0.0  # Default 0.0
    
    @patch('parser.pdfplumber.open')
    def test_extract_words_error_handling(self, mock_open):
        """Test error handling during PDF extraction."""
        mock_open.side_effect = Exception("PDF read error")
        
        words = extract_words_from_pdf('nonexistent.pdf')
        
        assert words == []  # Should return empty list on error


class TestMergeAdjacentWords:
    """Test word merging functionality."""
    
    def test_merge_identical_fonts(self):
        """Test merging words with identical font properties."""
        words = [
            {
                'text': 'Hello', 'page': 1, 'x0': 10, 'y0': 100, 'x1': 30, 'y1': 110,
                'fontname': 'Arial', 'size': 12.0, 'adv': 20.0
            },
            {
                'text': 'World', 'page': 1, 'x0': 35, 'y0': 100, 'x1': 55, 'y1': 110,
                'fontname': 'Arial', 'size': 12.0, 'adv': 20.0
            }
        ]
        
        blocks = merge_adjacent_words(words)
        
        assert len(blocks) == 1
        assert blocks[0]['text'] == 'Hello World'
        assert blocks[0]['x0'] == 10
        assert blocks[0]['x1'] == 55
        assert blocks[0]['fontname'] == 'Arial'
    
    def test_merge_different_fonts_no_merge(self):
        """Test that words with different fonts don't merge."""
        words = [
            {
                'text': 'Hello', 'page': 1, 'x0': 10, 'y0': 100, 'x1': 30, 'y1': 110,
                'fontname': 'Arial', 'size': 12.0, 'adv': 20.0
            },
            {
                'text': 'World', 'page': 1, 'x0': 31, 'y0': 100, 'x1': 51, 'y1': 110,
                'fontname': 'Times', 'size': 12.0, 'adv': 20.0
            }
        ]
        
        blocks = merge_adjacent_words(words)
        
        assert len(blocks) == 2
        assert blocks[0]['text'] == 'Hello'
        assert blocks[1]['text'] == 'World'
    
    def test_merge_vertical_misalignment(self):
        """Test that vertically misaligned words don't merge."""
        words = [
            {
                'text': 'Hello', 'page': 1, 'x0': 10, 'y0': 100, 'x1': 30, 'y1': 110,
                'fontname': 'Arial', 'size': 12.0, 'adv': 20.0
            },
            {
                'text': 'World', 'page': 1, 'x0': 31, 'y0': 105, 'x1': 51, 'y1': 115,  # y0 diff = 5 > 1.5
                'fontname': 'Arial', 'size': 12.0, 'adv': 20.0
            }
        ]
        
        blocks = merge_adjacent_words(words)
        
        assert len(blocks) == 2  # Should not merge due to vertical misalignment
    
    def test_merge_large_horizontal_gap(self):
        """Test that words with large horizontal gaps don't merge."""
        words = [
            {
                'text': 'Hello', 'page': 1, 'x0': 10, 'y0': 100, 'x1': 30, 'y1': 110,
                'fontname': 'Arial', 'size': 12.0, 'adv': 20.0
            },
            {
                'text': 'World', 'page': 1, 'x0': 100, 'y0': 100, 'x1': 120, 'y1': 110,  # Large gap
                'fontname': 'Arial', 'size': 12.0, 'adv': 20.0
            }
        ]
        
        blocks = merge_adjacent_words(words)
        
        assert len(blocks) == 2  # Should not merge due to large gap
    
    def test_merge_numbered_sections(self):
        """Test merging of numbered sections with different fonts."""
        words = [
            {
                'text': '1.', 'page': 1, 'x0': 10, 'y0': 100, 'x1': 15, 'y1': 110,
                'fontname': 'Arial-Bold', 'size': 12.0, 'adv': 5.0
            },
            {
                'text': 'Introduction', 'page': 1, 'x0': 20, 'y0': 100, 'x1': 80, 'y1': 110,
                'fontname': 'Arial', 'size': 12.0, 'adv': 60.0
            }
        ]
        
        blocks = merge_adjacent_words(words)
        
        assert len(blocks) == 1
        assert blocks[0]['text'] == '1. Introduction'
        assert blocks[0]['fontname'] == 'Arial'  # Should use text font, not number font
    
    def test_merge_empty_input(self):
        """Test merging with empty input."""
        blocks = merge_adjacent_words([])
        assert blocks == []
    
    def test_merge_single_word(self):
        """Test merging with single word."""
        words = [
            {
                'text': 'Single', 'page': 1, 'x0': 10, 'y0': 100, 'x1': 30, 'y1': 110,
                'fontname': 'Arial', 'size': 12.0, 'adv': 20.0
            }
        ]
        
        blocks = merge_adjacent_words(words)
        
        assert len(blocks) == 1
        assert blocks[0]['text'] == 'Single'


class TestComputeBlockFeatures:
    """Test block feature computation."""
    
    def test_compute_basic_features(self):
        """Test computation of basic block features."""
        block = {
            'text': 'Test Heading:', 'page': 1, 'x0': 10, 'y0': 100, 'x1': 90, 'y1': 110,
            'fontname': 'Arial-Bold', 'size': 14.0, 'adv': 80.0
        }
        
        enhanced = compute_block_features(block, None, 12.0)
        
        assert enhanced['bold'] == True  # Detected from Arial-Bold
        assert enhanced['italic'] == False
        assert abs(enhanced['uppercase_ratio'] - 2/11) < 0.001  # 'T' and 'H' out of 11 letters
        assert enhanced['num_prefix'] == False
        assert enhanced['ends_with_colon'] == True
        assert enhanced['x_center'] == 50.0  # (10 + 90) / 2
        assert enhanced['indent'] == 10.0  # x0
        assert enhanced['gap_above'] == 0.0  # No previous block
        assert enhanced['page_avg_line_height'] == 12.0
    
    def test_compute_with_previous_block(self):
        """Test gap calculation with previous block."""
        prev_block = {
            'y1': 90  # Bottom at y=90
        }
        
        current_block = {
            'text': 'Current', 'page': 1, 'x0': 10, 'y0': 100, 'x1': 50, 'y1': 110,
            'fontname': 'Arial', 'size': 12.0, 'adv': 40.0
        }
        
        enhanced = compute_block_features(current_block, prev_block, 12.0)
        
        assert enhanced['gap_above'] == 10.0  # 100 - 90
    
    def test_compute_uppercase_ratio(self):
        """Test uppercase ratio calculation."""
        # All uppercase
        block1 = {'text': 'UPPERCASE', 'fontname': 'Arial', 'size': 12.0, 'x0': 10, 'x1': 50}
        enhanced1 = compute_block_features(block1, None, 12.0)
        assert enhanced1['uppercase_ratio'] == 1.0
        
        # Mixed case
        block2 = {'text': 'MiXeD CaSe', 'fontname': 'Arial', 'size': 12.0, 'x0': 10, 'x1': 50}
        enhanced2 = compute_block_features(block2, None, 12.0)
        # 'M', 'X', 'D', 'C', 'S' = 5 uppercase out of 9 letters
        assert abs(enhanced2['uppercase_ratio'] - 5/9) < 0.001
        
        # No letters
        block3 = {'text': '123 !@#', 'fontname': 'Arial', 'size': 12.0, 'x0': 10, 'x1': 50}
        enhanced3 = compute_block_features(block3, None, 12.0)
        assert enhanced3['uppercase_ratio'] == 0.0
    
    def test_number_prefix_detection(self):
        """Test number prefix detection."""
        test_cases = [
            ('1. Introduction', True),
            ('1.2.3. Subsection', True),
            ('A. Appendix', True),
            ('I. Roman', True),
            ('(1) Parentheses', True),
            ('Regular text', False),
            ('Not 1. a number', False)
        ]
        
        for text, expected in test_cases:
            block = {'text': text, 'fontname': 'Arial', 'size': 12.0, 'x0': 10, 'x1': 50}
            enhanced = compute_block_features(block, None, 12.0)
            assert enhanced['num_prefix'] == expected, f"Failed for text: '{text}'"


class TestDetectBoldItalic:
    """Test bold and italic detection from font names."""
    
    def test_detect_bold_fonts(self):
        """Test detection of bold fonts."""
        bold_fonts = [
            'Arial-Bold', 'Times-Bold', 'Helvetica-Black',
            'Arial-Heavy', 'Times-Thick', 'Font-Demi', 'Font-Semi'
        ]
        
        for font in bold_fonts:
            is_bold, is_italic = detect_bold_italic(font)
            assert is_bold == True, f"Failed to detect bold in: {font}"
    
    def test_detect_italic_fonts(self):
        """Test detection of italic fonts."""
        italic_fonts = [
            'Arial-Italic', 'Times-Oblique', 'Helvetica-Slant'
        ]
        
        for font in italic_fonts:
            is_bold, is_italic = detect_bold_italic(font)
            assert is_italic == True, f"Failed to detect italic in: {font}"
    
    def test_detect_bold_italic_combination(self):
        """Test detection of bold-italic combinations."""
        is_bold, is_italic = detect_bold_italic('Arial-BoldItalic')
        assert is_bold == True
        assert is_italic == True
    
    def test_detect_regular_fonts(self):
        """Test that regular fonts are not detected as bold/italic."""
        regular_fonts = ['Arial', 'Times-Roman', 'Helvetica', 'Calibri']
        
        for font in regular_fonts:
            is_bold, is_italic = detect_bold_italic(font)
            assert is_bold == False, f"Incorrectly detected bold in: {font}"
            assert is_italic == False, f"Incorrectly detected italic in: {font}"


class TestDetectNumberPrefix:
    """Test number prefix detection patterns."""
    
    def test_arabic_numerals(self):
        """Test Arabic numeral patterns."""
        test_cases = [
            ('1. Chapter', True),
            ('1.2. Section', True),
            ('1.2.3. Subsection', True),
            ('10.5.2. Deep nesting', True),
            ('1 No dot', False),
            ('Chapter 1.', False)  # Number not at start
        ]
        
        for text, expected in test_cases:
            result = detect_number_prefix(text)
            assert result == expected, f"Failed for: '{text}'"
    
    def test_roman_numerals(self):
        """Test Roman numeral patterns."""
        test_cases = [
            ('I. Introduction', True),
            ('II. Chapter Two', True),
            ('III. Chapter Three', True),
            ('IV. Chapter Four', True),
            ('V. Chapter Five', True),
            ('X. Chapter Ten', True),
            ('I Introduction', False),  # No dot
        ]
        
        for text, expected in test_cases:
            result = detect_number_prefix(text)
            assert result == expected, f"Failed for: '{text}'"
    
    def test_letter_patterns(self):
        """Test letter-based numbering."""
        test_cases = [
            ('A. Appendix', True),
            ('B. Bibliography', True),
            ('a. lowercase', True),
            ('z. last letter', True),
            ('AA. Not supported', False),  # Multiple letters not in basic pattern
        ]
        
        for text, expected in test_cases:
            result = detect_number_prefix(text)
            assert result == expected, f"Failed for: '{text}'"
    
    def test_parentheses_patterns(self):
        """Test parentheses-based numbering."""
        test_cases = [
            ('(1) First item', True),
            ('(2) Second item', True),
            ('(a) Letter item', True),
            ('(A) Capital letter', True),
            ('1) Missing open paren', False),
            ('(1 Missing close paren', False)
        ]
        
        for text, expected in test_cases:
            result = detect_number_prefix(text)
            assert result == expected, f"Failed for: '{text}'"


class TestComputePageLineHeight:
    """Test page line height calculation."""
    
    def test_compute_normal_blocks(self):
        """Test line height calculation with normal blocks."""
        blocks = [
            {'y0': 100, 'y1': 112},  # Height = 12
            {'y0': 120, 'y1': 132},  # Height = 12
            {'y0': 140, 'y1': 154},  # Height = 14
        ]
        
        avg_height = compute_page_line_height(blocks)
        expected = (12 + 12 + 14) / 3
        assert abs(avg_height - expected) < 0.001
    
    def test_compute_empty_blocks(self):
        """Test line height with empty block list."""
        avg_height = compute_page_line_height([])
        assert avg_height == 12.0  # Default
    
    def test_compute_zero_height_blocks(self):
        """Test handling of zero-height blocks."""
        blocks = [
            {'y0': 100, 'y1': 100},  # Height = 0 (should be ignored)
            {'y0': 120, 'y1': 132},  # Height = 12
        ]
        
        avg_height = compute_page_line_height(blocks)
        assert avg_height == 12.0  # Only valid height used
    
    def test_compute_all_zero_heights(self):
        """Test when all blocks have zero height."""
        blocks = [
            {'y0': 100, 'y1': 100},
            {'y0': 120, 'y1': 120},
        ]
        
        avg_height = compute_page_line_height(blocks)
        assert avg_height == 12.0  # Default when no valid heights


class TestCleanDuplicatedCharacters:
    """Test character duplication cleanup."""
    
    def test_clean_repeated_letters(self):
        """Test cleaning of repeated letters."""
        # The function reduces 4+ repetitions to 1
        assert clean_duplicated_characters('Hellllllo') == 'Helo'  # 5 l's -> 1 l
        assert clean_duplicated_characters('Goooood') == 'God'     # 4 o's -> 1 o
        assert clean_duplicated_characters('Tesssst') == 'Test'    # 4 s's -> 1 s
    
    def test_clean_repeated_punctuation(self):
        """Test cleaning of repeated punctuation."""
        assert clean_duplicated_characters('Hello:::') == 'Hello:'
        assert clean_duplicated_characters('Wait...') == 'Wait.'
        assert clean_duplicated_characters('What???') == 'What?'
    
    def test_preserve_legitimate_repetition(self):
        """Test that legitimate repetition is preserved."""
        # Should not affect legitimate double letters
        assert clean_duplicated_characters('Hello') == 'Hello'
        assert clean_duplicated_characters('Mississippi') == 'Mississippi'
        assert clean_duplicated_characters('Bookkeeper') == 'Bookkeeper'
    
    def test_clean_empty_string(self):
        """Test cleaning empty string."""
        assert clean_duplicated_characters('') == ''
        assert clean_duplicated_characters(None) == None


class TestMergeNumberedSections:
    """Test numbered section merging functionality."""
    
    def test_is_number_prefix(self):
        """Test number prefix identification."""
        test_cases = [
            ({'text': '1.'}, True),
            ({'text': '2.'}, True),
            ({'text': 'A.'}, True),
            ({'text': 'I.'}, True),
            ({'text': 'Hello'}, False),
            ({'text': '1.2'}, False),  # No trailing dot
        ]
        
        for block, expected in test_cases:
            result = _is_number_prefix(block)
            assert result == expected, f"Failed for: {block['text']}"
    
    def test_should_merge_with_next(self):
        """Test merge decision logic."""
        current = {
            'page': 1, 'y0': 100, 'x1': 20, 'size': 12.0, 'text': '1.'
        }
        
        # Should merge - same page, line, reasonable gap, uppercase start
        next_good = {
            'page': 1, 'y0': 100, 'x0': 25, 'size': 12.0, 'text': 'Introduction'
        }
        assert _should_merge_with_next(current, next_good) == True
        
        # Should not merge - different page
        next_diff_page = {
            'page': 2, 'y0': 100, 'x0': 25, 'size': 12.0, 'text': 'Introduction'
        }
        assert _should_merge_with_next(current, next_diff_page) == False
        
        # Should not merge - lowercase start
        next_lowercase = {
            'page': 1, 'y0': 100, 'x0': 25, 'size': 12.0, 'text': 'introduction'
        }
        assert _should_merge_with_next(current, next_lowercase) == False
    
    def test_merge_numbered_sections_integration(self):
        """Test complete numbered section merging."""
        blocks = [
            {
                'text': '1.', 'page': 1, 'x0': 10, 'y0': 100, 'x1': 15, 'y1': 110,
                'fontname': 'Arial-Bold', 'size': 12.0, 'adv': 5.0
            },
            {
                'text': 'Introduction', 'page': 1, 'x0': 20, 'y0': 100, 'x1': 80, 'y1': 110,
                'fontname': 'Arial', 'size': 12.0, 'adv': 60.0
            },
            {
                'text': 'Regular text', 'page': 1, 'x0': 10, 'y0': 120, 'x1': 70, 'y1': 130,
                'fontname': 'Arial', 'size': 12.0, 'adv': 60.0
            }
        ]
        
        merged = merge_numbered_sections(blocks)
        
        assert len(merged) == 2
        assert merged[0]['text'] == '1. Introduction'
        assert merged[0]['fontname'] == 'Arial'  # Uses text font
        assert merged[1]['text'] == 'Regular text'


class TestExtractBlocks:
    """Test the main extract_blocks function integration."""
    
    @patch('parser.extract_words_from_pdf')
    def test_extract_blocks_integration(self, mock_extract_words):
        """Test complete block extraction pipeline."""
        # Mock word extraction
        mock_extract_words.return_value = [
            {
                'text': 'Title', 'page': 1, 'x0': 10, 'y0': 50, 'x1': 40, 'y1': 65,
                'fontname': 'Arial-Bold', 'size': 16.0, 'adv': 30.0
            },
            {
                'text': 'Chapter', 'page': 1, 'x0': 10, 'y0': 80, 'x1': 50, 'y1': 95,
                'fontname': 'Arial', 'size': 14.0, 'adv': 40.0
            },
            {
                'text': 'Body', 'page': 2, 'x0': 10, 'y0': 50, 'x1': 35, 'y1': 62,
                'fontname': 'Arial', 'size': 12.0, 'adv': 25.0
            }
        ]
        
        blocks = extract_blocks('test.pdf')
        
        assert len(blocks) == 3
        
        # Check that all features are computed
        for block in blocks:
            assert 'bold' in block
            assert 'italic' in block
            assert 'uppercase_ratio' in block
            assert 'num_prefix' in block
            assert 'ends_with_colon' in block
            assert 'x_center' in block
            assert 'indent' in block
            assert 'gap_above' in block
            assert 'page_avg_line_height' in block
        
        # Check page-specific processing
        assert blocks[0]['page'] == 1
        assert blocks[1]['page'] == 1
        assert blocks[2]['page'] == 2
        
        # Check gap calculation within page
        assert blocks[0]['gap_above'] == 0.0  # First block on page
        assert blocks[1]['gap_above'] == 15.0  # 80 - 65
        assert blocks[2]['gap_above'] == 0.0  # First block on page 2
    
    @patch('parser.extract_words_from_pdf')
    def test_extract_blocks_empty_words(self, mock_extract_words):
        """Test handling when no words are extracted."""
        mock_extract_words.return_value = []
        
        blocks = extract_blocks('empty.pdf')
        
        assert blocks == []


if __name__ == '__main__':
    pytest.main([__file__])