from typing import List, Dict, Tuple
from collections import Counter
import statistics
import argparse
from extractor import PDFTextExtractor, TextBlock


class FontSizeAnalyzer:
    def __init__(self, blocks: List[TextBlock]):
        self.blocks = blocks
    
    def get_mode_font_size(self) -> float:
        """Calculate the mode (most frequent) font size."""
        if not self.blocks:
            return 0.0
        
        font_sizes = [block.font_size for block in self.blocks]
        
        # Round to 1 decimal place to handle floating point precision issues
        rounded_sizes = [round(size, 1) for size in font_sizes]
        
        # Get the most common font size
        size_counter = Counter(rounded_sizes)
        mode_size = size_counter.most_common(1)[0][0]
        
        return mode_size
    
    def get_average_font_size(self) -> float:
        """Calculate the average font size."""
        if not self.blocks:
            return 0.0
        
        font_sizes = [block.font_size for block in self.blocks]
        return statistics.mean(font_sizes)
    
    def get_font_size_frequency(self) -> Dict[float, int]:
        """Get frequency count of each font size."""
        if not self.blocks:
            return {}
        
        font_sizes = [round(block.font_size, 1) for block in self.blocks]
        return dict(Counter(font_sizes))
    
    def aggregate_blocks_by_font_size(self) -> Dict[float, List[TextBlock]]:
        """Group blocks by their font size."""
        size_groups = {}
        
        for block in self.blocks:
            rounded_size = round(block.font_size, 1)
            
            if rounded_size not in size_groups:
                size_groups[rounded_size] = []
            
            size_groups[rounded_size].append(block)
        
        return size_groups
    
    def get_font_size_statistics(self) -> Dict:
        """Get comprehensive font size statistics."""
        if not self.blocks:
            return {
                'mode': 0.0,
                'average': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'frequency': {},
                'unique_sizes': 0,
                'total_blocks': 0
            }
        
        font_sizes = [block.font_size for block in self.blocks]
        frequency = self.get_font_size_frequency()
        
        return {
            'mode': self.get_mode_font_size(),
            'average': self.get_average_font_size(),
            'median': statistics.median(font_sizes),
            'min': min(font_sizes),
            'max': max(font_sizes),
            'frequency': frequency,
            'unique_sizes': len(frequency),
            'total_blocks': len(self.blocks)
        }
    
    def print_font_size_analysis(self):
        """Print a detailed analysis of font sizes."""
        stats = self.get_font_size_statistics()
        
        print("=== FONT SIZE ANALYSIS ===")
        print(f"Total blocks: {stats['total_blocks']}")
        print(f"Unique font sizes: {stats['unique_sizes']}")
        print()
        
        print("=== BASIC STATISTICS ===")
        print(f"Mode (most frequent): {stats['mode']:.1f}pt")
        print(f"Average: {stats['average']:.2f}pt")
        print(f"Median: {stats['median']:.2f}pt")
        print(f"Min: {stats['min']:.1f}pt")
        print(f"Max: {stats['max']:.1f}pt")
        print()
        
        print("=== FREQUENCY DISTRIBUTION ===")
        # Sort by frequency (descending) then by font size
        sorted_freq = sorted(stats['frequency'].items(), 
                           key=lambda x: (-x[1], x[0]))
        
        for font_size, count in sorted_freq:
            percentage = (count / stats['total_blocks']) * 100
            print(f"{font_size:6.1f}pt: {count:3d} blocks ({percentage:5.1f}%)")
        
        print()
    
    def print_blocks_by_font_size(self, max_text_length: int = 50):
        """Print blocks grouped by font size."""
        size_groups = self.aggregate_blocks_by_font_size()
        
        print("=== BLOCKS GROUPED BY FONT SIZE ===")
        
        # Sort by font size
        for font_size in sorted(size_groups.keys()):
            blocks = size_groups[font_size]
            print(f"\n--- Font Size: {font_size:.1f}pt ({len(blocks)} blocks) ---")
            
            for i, block in enumerate(blocks, 1):
                # Truncate text if too long
                text = block.text.replace('\n', ' ').strip()
                if len(text) > max_text_length:
                    text = text[:max_text_length] + "..."
                
                table_indicator = " [TABLE]" if block.is_in_table else ""
                style_indicators = []
                if block.is_bold:
                    style_indicators.append("BOLD")
                if block.is_italic:
                    style_indicators.append("ITALIC")
                if block.is_underlined:
                    style_indicators.append("UNDERLINED")
                
                style_str = f" [{', '.join(style_indicators)}]" if style_indicators else ""
                
                print(f"  {i:2d}. {text}{table_indicator}{style_str}")


def main():
    parser = argparse.ArgumentParser(description="Analyze font size statistics from PDF")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--detailed", "-d", action="store_true", 
                       help="Show detailed block listings by font size")
    parser.add_argument("--max-text", "-m", type=int, default=50,
                       help="Maximum text length to display (default: 50)")
    
    args = parser.parse_args()
    
    # Extract text blocks from PDF
    extractor = PDFTextExtractor()
    blocks = extractor.extract_text_blocks(args.pdf_path)
    
    if not blocks:
        print("No text blocks found in the PDF.")
        return
    
    # Analyze font sizes
    analyzer = FontSizeAnalyzer(blocks)
    
    # Print basic analysis
    analyzer.print_font_size_analysis()
    
    # Print detailed block listings if requested
    if args.detailed:
        analyzer.print_blocks_by_font_size(args.max_text)
    
    # Return the analyzer for programmatic use
    return analyzer


if __name__ == "__main__":
    main()