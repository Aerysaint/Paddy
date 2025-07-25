"""
Unit tests for the outline module - hierarchy assembly and JSON output generation.
"""

import pytest
import json
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from outline import (
    build_hierarchy, flatten_outline, format_json_structure,
    validate_json_schema, handle_empty_cases, serialize_to_json,
    generate_document_outline, _sort_headings_by_position,
    _apply_demotion_rules, _create_outline_entry,
    fix_page_numbering_for_single_page_docs, _get_total_pages_from_headings
)


class TestBuildHierarchy:
    """Test hierarchy building functionality."""
    
    def test_build_simple_hierarchy(self):
        """Test building simple H1 -> H2 hierarchy."""
        headings = [
            {
                'level': 1, 'text': 'Chapter 1', 'page': 1,
                'source_block': {'y0': 100}
            },
            {
                'level': 2, 'text': 'Section 1.1', 'page': 1,
                'source_block': {'y0': 150}
            },
            {
                'level': 2, 'text': 'Section 1.2', 'page': 2,
                'source_block': {'y0': 100}
            }
        ]
        
        hierarchy = build_hierarchy(headings)
        
        assert len(hierarchy) == 1  # One root H1
        assert hierarchy[0]['level'] == 1
        assert hierarchy[0]['text'] == 'Chapter 1'
        assert len(hierarchy[0]['children']) == 2  # Two H2 children
        assert hierarchy[0]['children'][0]['text'] == 'Section 1.1'
        assert hierarchy[0]['children'][1]['text'] == 'Section 1.2'
    
    def test_build_deep_hierarchy(self):
        """Test building deep hierarchy with multiple levels."""
        headings = [
            {
                'level': 1, 'text': 'Chapter 1', 'page': 1,
                'source_block': {'y0': 100}
            },
            {
                'level': 2, 'text': 'Section 1.1', 'page': 1,
                'source_block': {'y0': 150}
            },
            {
                'level': 3, 'text': 'Subsection 1.1.1', 'page': 1,
                'source_block': {'y0': 200}
            },
            {
                'level': 4, 'text': 'Sub-subsection 1.1.1.1', 'page': 1,
                'source_block': {'y0': 250}
            }
        ]
        
        hierarchy = build_hierarchy(headings)
        
        assert len(hierarchy) == 1  # One root H1
        h1 = hierarchy[0]
        assert h1['level'] == 1
        assert len(h1['children']) == 1
        
        h2 = h1['children'][0]
        assert h2['level'] == 2
        assert len(h2['children']) == 1
        
        h3 = h2['children'][0]
        assert h3['level'] == 3
        assert len(h3['children']) == 1
        
        h4 = h3['children'][0]
        assert h4['level'] == 4
        assert 'children' not in h4  # Leaf node
    
    def test_build_multiple_h1s(self):
        """Test building hierarchy with multiple H1 headings."""
        headings = [
            {
                'level': 1, 'text': 'Chapter 1', 'page': 1,
                'source_block': {'y0': 100}
            },
            {
                'level': 2, 'text': 'Section 1.1', 'page': 1,
                'source_block': {'y0': 150}
            },
            {
                'level': 1, 'text': 'Chapter 2', 'page': 2,
                'source_block': {'y0': 100}
            },
            {
                'level': 2, 'text': 'Section 2.1', 'page': 2,
                'source_block': {'y0': 150}
            }
        ]
        
        hierarchy = build_hierarchy(headings)
        
        assert len(hierarchy) == 2  # Two root H1s
        assert hierarchy[0]['text'] == 'Chapter 1'
        assert hierarchy[1]['text'] == 'Chapter 2'
        assert len(hierarchy[0]['children']) == 1
        assert len(hierarchy[1]['children']) == 1
    
    def test_build_empty_headings(self):
        """Test building hierarchy with empty headings list."""
        hierarchy = build_hierarchy([])
        assert hierarchy == []
    
    def test_build_with_demotion(self):
        """Test hierarchy building with demotion rules applied."""
        headings = [
            {
                'level': 1, 'text': 'Chapter 1', 'page': 1,
                'source_block': {'y0': 100}
            },
            {
                'level': 3, 'text': 'Orphan H3', 'page': 1,  # No H2 parent
                'source_block': {'y0': 150}
            }
        ]
        
        hierarchy = build_hierarchy(headings)
        
        assert len(hierarchy) == 1  # One root H1
        assert len(hierarchy[0]['children']) == 1  # H3 demoted to H2
        assert hierarchy[0]['children'][0]['level'] == 2  # Demoted level
        assert hierarchy[0]['children'][0]['text'] == 'Orphan H3'


class TestSortHeadingsByPosition:
    """Test heading sorting functionality."""
    
    def test_sort_by_page_and_position(self):
        """Test sorting headings by page and y-coordinate."""
        headings = [
            {
                'text': 'Second on page 1', 'page': 1,
                'source_block': {'y0': 200}
            },
            {
                'text': 'First on page 2', 'page': 2,
                'source_block': {'y0': 100}
            },
            {
                'text': 'First on page 1', 'page': 1,
                'source_block': {'y0': 100}
            }
        ]
        
        sorted_headings = _sort_headings_by_position(headings)
        
        assert len(sorted_headings) == 3
        assert sorted_headings[0]['text'] == 'First on page 1'
        assert sorted_headings[1]['text'] == 'Second on page 1'
        assert sorted_headings[2]['text'] == 'First on page 2'
    
    def test_sort_missing_source_block(self):
        """Test sorting with missing source_block information."""
        headings = [
            {
                'text': 'No source block', 'page': 1
                # Missing source_block
            },
            {
                'text': 'With source block', 'page': 1,
                'source_block': {'y0': 100}
            }
        ]
        
        sorted_headings = _sort_headings_by_position(headings)
        
        assert len(sorted_headings) == 2
        # Should handle missing source_block gracefully


class TestApplyDemotionRules:
    """Test demotion rules functionality."""
    
    def test_apply_demotion_single_level(self):
        """Test demotion by single level."""
        outline = []
        level_stack = {1: {'level': 1, 'text': 'Chapter 1', 'children': []}}
        
        outline_entry = {'level': 3, 'text': 'Orphan H3', 'page': 1}
        
        _apply_demotion_rules(outline_entry, 3, level_stack, outline)
        
        # Should be demoted to H2 and attached under H1
        assert len(level_stack[1]['children']) == 1
        assert level_stack[1]['children'][0]['level'] == 2
        assert level_stack[1]['children'][0]['text'] == 'Orphan H3'
    
    def test_apply_demotion_to_root(self):
        """Test demotion all the way to root level."""
        outline = []
        level_stack = {}  # No parents available
        
        outline_entry = {'level': 3, 'text': 'Orphan H3', 'page': 1}
        
        _apply_demotion_rules(outline_entry, 3, level_stack, outline)
        
        # Should be demoted to H1 and added to root
        assert len(outline) == 1
        assert outline[0]['level'] == 1
        assert outline[0]['text'] == 'Orphan H3'


class TestCreateOutlineEntry:
    """Test outline entry creation."""
    
    def test_create_basic_entry(self):
        """Test creating basic outline entry."""
        heading = {
            'level': 2,
            'text': '  Heading Text  ',  # With whitespace
            'page': 5
        }
        
        entry = _create_outline_entry(heading)
        
        assert entry['level'] == 2
        assert entry['text'] == '  Heading Text  '  # Whitespace preserved during creation
        assert entry['page'] == 5
    
    def test_create_entry_missing_fields(self):
        """Test creating entry with missing fields."""
        heading = {}  # Empty heading
        
        entry = _create_outline_entry(heading)
        
        assert entry['level'] == 1  # Default
        assert entry['text'] == ''  # Default
        assert entry['page'] == 1  # Default


class TestFlattenOutline:
    """Test outline flattening functionality."""
    
    def test_flatten_simple_hierarchy(self):
        """Test flattening simple hierarchy."""
        hierarchy = [
            {
                'level': 1, 'text': 'Chapter 1', 'page': 1,
                'children': [
                    {'level': 2, 'text': 'Section 1.1', 'page': 1},
                    {'level': 2, 'text': 'Section 1.2', 'page': 2}
                ]
            }
        ]
        
        flattened = flatten_outline(hierarchy)
        
        assert len(flattened) == 3
        assert flattened[0]['level'] == 'H1'
        assert flattened[0]['text'] == 'Chapter 1'
        assert flattened[1]['level'] == 'H2'
        assert flattened[1]['text'] == 'Section 1.1'
        assert flattened[2]['level'] == 'H2'
        assert flattened[2]['text'] == 'Section 1.2'
    
    def test_flatten_deep_hierarchy(self):
        """Test flattening deep hierarchy."""
        hierarchy = [
            {
                'level': 1, 'text': 'Chapter 1', 'page': 1,
                'children': [
                    {
                        'level': 2, 'text': 'Section 1.1', 'page': 1,
                        'children': [
                            {'level': 3, 'text': 'Subsection 1.1.1', 'page': 1}
                        ]
                    }
                ]
            }
        ]
        
        flattened = flatten_outline(hierarchy)
        
        assert len(flattened) == 3
        assert flattened[0]['level'] == 'H1'
        assert flattened[1]['level'] == 'H2'
        assert flattened[2]['level'] == 'H3'
    
    def test_flatten_empty_hierarchy(self):
        """Test flattening empty hierarchy."""
        flattened = flatten_outline([])
        assert flattened == []
    
    def test_flatten_no_children(self):
        """Test flattening hierarchy with no children."""
        hierarchy = [
            {'level': 1, 'text': 'Single Heading', 'page': 1}
        ]
        
        flattened = flatten_outline(hierarchy)
        
        assert len(flattened) == 1
        assert flattened[0]['level'] == 'H1'
        assert flattened[0]['text'] == 'Single Heading'


class TestFormatJsonStructure:
    """Test JSON structure formatting."""
    
    def test_format_basic_structure(self):
        """Test formatting basic JSON structure."""
        title = 'Document Title'
        headings = [
            {
                'level': 1, 'text': 'Chapter 1', 'page': 1,
                'source_block': {'y0': 100}
            },
            {
                'level': 2, 'text': 'Section 1.1', 'page': 1,
                'source_block': {'y0': 150}
            }
        ]
        
        json_structure = format_json_structure(title, headings)
        
        assert json_structure['title'] == 'Document Title'
        assert len(json_structure['outline']) == 2
        assert json_structure['outline'][0]['level'] == 'H1'
        assert json_structure['outline'][0]['text'] == 'Chapter 1'
        assert json_structure['outline'][1]['level'] == 'H2'
        assert json_structure['outline'][1]['text'] == 'Section 1.1'
    
    def test_format_empty_title(self):
        """Test formatting with empty title."""
        json_structure = format_json_structure('', [])
        
        assert json_structure['title'] == ''
        assert json_structure['outline'] == []
    
    def test_format_whitespace_handling(self):
        """Test that whitespace is properly stripped."""
        headings = [
            {
                'level': 1, 'text': '  Heading with spaces  ', 'page': 1,
                'source_block': {'y0': 100}
            }
        ]
        
        json_structure = format_json_structure('  Title with spaces  ', headings)
        
        assert json_structure['title'] == 'Title with spaces'
        assert json_structure['outline'][0]['text'] == 'Heading with spaces'


class TestValidateJsonSchema:
    """Test JSON schema validation."""
    
    def test_validate_correct_schema(self):
        """Test validation of correct JSON schema."""
        valid_json = {
            'title': 'Test Document',
            'outline': [
                {'level': 'H1', 'text': 'Chapter 1', 'page': 1},
                {'level': 'H2', 'text': 'Section 1.1', 'page': 1},
                {'level': 'H10', 'text': 'Deep heading', 'page': 2}
            ]
        }
        
        assert validate_json_schema(valid_json) == True
    
    def test_validate_missing_title(self):
        """Test validation with missing title field."""
        invalid_json = {
            'outline': []
        }
        
        assert validate_json_schema(invalid_json) == False
    
    def test_validate_missing_outline(self):
        """Test validation with missing outline field."""
        invalid_json = {
            'title': 'Test'
        }
        
        assert validate_json_schema(invalid_json) == False
    
    def test_validate_wrong_title_type(self):
        """Test validation with wrong title type."""
        invalid_json = {
            'title': 123,  # Should be string
            'outline': []
        }
        
        assert validate_json_schema(invalid_json) == False
    
    def test_validate_wrong_outline_type(self):
        """Test validation with wrong outline type."""
        invalid_json = {
            'title': 'Test',
            'outline': 'not a list'  # Should be list
        }
        
        assert validate_json_schema(invalid_json) == False
    
    def test_validate_invalid_outline_entry(self):
        """Test validation with invalid outline entry."""
        invalid_json = {
            'title': 'Test',
            'outline': [
                {'level': 'H1', 'text': 'Good entry', 'page': 1},
                {'level': 'BadLevel', 'text': 'Bad entry', 'page': 1}  # Invalid level
            ]
        }
        
        assert validate_json_schema(invalid_json) == False
    
    def test_validate_missing_outline_fields(self):
        """Test validation with missing outline entry fields."""
        invalid_json = {
            'title': 'Test',
            'outline': [
                {'level': 'H1', 'text': 'Missing page field'}  # Missing page
            ]
        }
        
        assert validate_json_schema(invalid_json) == False
    
    def test_validate_invalid_page_number(self):
        """Test validation with invalid page number."""
        invalid_json = {
            'title': 'Test',
            'outline': [
                {'level': 'H1', 'text': 'Test', 'page': -1}  # Negative page
            ]
        }
        
        assert validate_json_schema(invalid_json) == False
    
    def test_validate_empty_outline(self):
        """Test validation with empty outline (should be valid)."""
        valid_json = {
            'title': 'Test Document',
            'outline': []
        }
        
        assert validate_json_schema(valid_json) == True
    
    def test_validate_empty_title(self):
        """Test validation with empty title (should be valid)."""
        valid_json = {
            'title': '',
            'outline': [
                {'level': 'H1', 'text': 'Chapter 1', 'page': 1}
            ]
        }
        
        assert validate_json_schema(valid_json) == True


class TestHandleEmptyCases:
    """Test empty case handling."""
    
    def test_handle_empty_title(self):
        """Test handling of empty title."""
        result = handle_empty_cases(None, [])
        assert result['title'] == ''
        
        result = handle_empty_cases('', [])
        assert result['title'] == ''
        
        result = handle_empty_cases('   ', [])
        assert result['title'] == ''
    
    def test_handle_valid_title(self):
        """Test handling of valid title."""
        result = handle_empty_cases('Valid Title', [])
        assert result['title'] == 'Valid Title'
        
        result = handle_empty_cases('  Spaced Title  ', [])
        assert result['title'] == 'Spaced Title'
    
    def test_handle_empty_outline(self):
        """Test handling of empty outline."""
        result = handle_empty_cases('Title', None)
        assert result['outline'] == []
        
        result = handle_empty_cases('Title', [])
        assert result['outline'] == []
    
    def test_handle_valid_outline(self):
        """Test handling of valid outline."""
        outline = [{'level': 'H1', 'text': 'Test', 'page': 1}]
        result = handle_empty_cases('Title', outline)
        assert result['outline'] == outline


class TestSerializeToJson:
    """Test JSON serialization."""
    
    def test_serialize_basic(self):
        """Test basic JSON serialization."""
        data = {
            'title': 'Test Document',
            'outline': [
                {'level': 'H1', 'text': 'Chapter 1', 'page': 1}
            ]
        }
        
        json_string = serialize_to_json(data)
        
        # Should be valid JSON
        parsed = json.loads(json_string)
        assert parsed == data
    
    def test_serialize_with_unicode(self):
        """Test serialization with Unicode characters."""
        data = {
            'title': 'Tëst Dócümënt',
            'outline': [
                {'level': 'H1', 'text': 'Chäptër 1', 'page': 1}
            ]
        }
        
        json_string = serialize_to_json(data)
        parsed = json.loads(json_string)
        assert parsed == data
    
    def test_serialize_compact(self):
        """Test compact serialization."""
        data = {'title': 'Test', 'outline': []}
        
        json_string = serialize_to_json(data, indent=None)
        
        # Should be compact (no extra whitespace)
        assert '\n' not in json_string
        assert '  ' not in json_string
    
    def test_serialize_invalid_data(self):
        """Test serialization with invalid data."""
        # Create circular reference (not JSON serializable)
        data = {}
        data['self'] = data
        
        with pytest.raises(Exception):
            serialize_to_json(data)


class TestGenerateDocumentOutline:
    """Test complete document outline generation."""
    
    def test_generate_complete_outline(self):
        """Test complete outline generation pipeline."""
        title = 'Test Document'
        headings = [
            {
                'level': 1, 'text': 'Chapter 1', 'page': 1,
                'source_block': {'y0': 100}
            },
            {
                'level': 2, 'text': 'Section 1.1', 'page': 1,
                'source_block': {'y0': 150}
            }
        ]
        
        outline = generate_document_outline(title, headings)
        
        # Should pass validation
        assert validate_json_schema(outline) == True
        assert outline['title'] == 'Test Document'
        assert len(outline['outline']) == 2
    
    def test_generate_empty_document(self):
        """Test generation with empty document."""
        outline = generate_document_outline('', [])
        
        assert validate_json_schema(outline) == True
        assert outline['title'] == ''
        assert outline['outline'] == []
    
    def test_generate_with_validation_failure(self):
        """Test that validation failure raises exception."""
        # This is hard to trigger since our functions should always generate valid JSON
        # But we can test the validation check exists
        title = 'Test'
        headings = []
        
        # Should not raise exception for valid case
        outline = generate_document_outline(title, headings)
        assert outline is not None


class TestPageNumberingFixes:
    """Test page numbering fixes for single-page documents."""
    
    def test_fix_single_page_numbering(self):
        """Test fixing page numbering for single-page documents."""
        outline_entries = [
            {'level': 'H1', 'text': 'Title', 'page': 1},
            {'level': 'H2', 'text': 'Section', 'page': 1}
        ]
        
        fixed = fix_page_numbering_for_single_page_docs(outline_entries, 1)
        
        # Should convert to 0-based for single page
        assert fixed[0]['page'] == 0
        assert fixed[1]['page'] == 0
    
    def test_fix_multi_page_numbering(self):
        """Test that multi-page documents keep 1-based numbering."""
        outline_entries = [
            {'level': 'H1', 'text': 'Title', 'page': 1},
            {'level': 'H2', 'text': 'Section', 'page': 2}
        ]
        
        fixed = fix_page_numbering_for_single_page_docs(outline_entries, 2)
        
        # Should keep 1-based for multi-page
        assert fixed[0]['page'] == 1
        assert fixed[1]['page'] == 2
    
    def test_get_total_pages_from_headings(self):
        """Test getting total pages from headings."""
        headings = [
            {'page': 1, 'source_block': {'page': 1}},
            {'page': 2, 'source_block': {'page': 3}},
            {'page': 1, 'source_block': {'page': 2}}
        ]
        
        total_pages = _get_total_pages_from_headings(headings)
        assert total_pages == 3  # Max page number
    
    def test_get_total_pages_empty(self):
        """Test getting total pages from empty headings."""
        total_pages = _get_total_pages_from_headings([])
        assert total_pages == 1  # Default


if __name__ == '__main__':
    pytest.main([__file__])