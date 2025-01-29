import xml.etree.ElementTree as ET

from .utils.dom import Element
from .series import Series
from .sparkline import Sparkline
from .theming import svg_theme_toggle
from .token_bb import TokenBB


class Sparky:
    def __init__(self, chars_per_line: int = 80):
        self.chars_per_line = chars_per_line
        self.font_size = 14
        self.line_height = self.font_size
        self.line_gap = self.line_height
        self.char_width = 8.4    # Width of each character in SVG units
        self.sparkline_height = 20
        self.margin = 10         # Margin around visualization
        
        # Register SVG namespace for proper rendering
        ET.register_namespace("", "http://www.w3.org/2000/svg")

    def _wrap_tokens(self, spans: list[TokenBB]) -> list[tuple[int, int]]:
        """Split tokens into lines based on total width, returning (start,end) indices."""
        lines = []
        line_start = 0
        current_width = 0
        
        for i, span in enumerate(spans):
            if current_width + span.width > self.chars_per_line * self.char_width:
                if line_start < i:  # Don't create empty lines
                    lines.append((line_start, i))
                line_start = i
                current_width = span.width
            else:
                current_width += span.width
        
        if line_start < len(spans):
            lines.append((line_start, len(spans)))
        
        return lines

    def _add_legend(self, svg: ET.Element, x: float, y: float, series: list[Series]) -> float:
        """Add legend to the SVG."""
        legend = Element(svg, "g", transform=f"translate({x}, {y})")
        font = dict(font_family="system-ui", font_size=10, fill='var(--col-text)')
        line_height = 15

        _y = y
        for i, s in enumerate(series):
            Element(legend, "line", 
                x1=0, y1=_y,
                x2=20, y2=_y,
                stroke=f'var(--col-series-{i+1})',
                stroke_width=1,
                stroke_dasharray=s.dasharray,
                shape_rendering="crispEdges",
            )
            Element(legend, "text", text=s.label, x=25, y=_y+4, **font)
            _y += line_height

        return _y

    def _get_token_spans(self, tokens: list[str]) -> list[TokenBB]:
        """Calculate token bounding boxes in relative coordinates."""
        spans = []
        for token in tokens:
            width = len(token) * self.char_width
            first_char = self.char_width/2
            middle = width/2
            last_char = width - self.char_width/2
            spans.append(TokenBB(width, first_char, middle, last_char))
        return spans
    
    def _add_text_line(self, parent: ET.Element, tokens: list[str], window: tuple[int, int], x: float, y: float):
        """Add a line of text with centered tokens."""
        start, end = window
        line_tokens = tokens[start:end]
    
        if x != 0.0 or y != 0.0:
            parent = Element(parent, "g", transform=f"translate({x}, {y})")
    
        # Add main text element with centered alignment
        baseline = self.font_size * -0.2  # Still need this offset for text positioning
        text_elem = Element(parent, "text",
            font_family="Courier",
            font_size=self.font_size,
            y=baseline,
            text_anchor="middle",
            fill='var(--col-text)',
        )
        
        # Track cumulative x position as we place tokens
        pos = 0
        for token in line_tokens:
            width = len(token) * self.char_width
            mid = pos + width/2
            Element(text_elem, "tspan", x=mid, text=token)
            pos += width
    
    def visualize(self, tokens: str | list[str], series: list[Series]) -> None:
        """Generate and display an SVG visualization of text metrics."""
        if isinstance(tokens, str):
            # Assume character-level tokens
            tokens = list(tokens)

        # Split tokens into lines and calculate dimensions
        spans = self._get_token_spans(tokens)
        lines = self._wrap_tokens(spans)
        text_width = self.chars_per_line * self.char_width
        legend_space = 100
        toggle_space = 20 + self.margin
        width = text_width + 2*self.margin + legend_space + toggle_space
        full_line_height = self.line_height + self.sparkline_height + self.line_gap
        height = max(2, len(lines)) * full_line_height + 2*self.margin

        sparkline = Sparkline()
        for i, s in enumerate(series):
            sparkline.add_series(s.values, f'var(--col-series-{i+1})', dasharray=s.dasharray)

        # Create SVG root
        svg = Element(None, "svg",
            xmlns="http://www.w3.org/2000/svg",
            viewBox=f"0 0 {width} {height}"
        )
        Element(svg, 'style', text="text { white-space: pre; }")
        Element(svg, "rect", width=width, height=height, fill='var(--bg-color)')
        
        # Process each line
        for i, (start, end) in enumerate(lines):
            y_offset = i*full_line_height + self.margin
            baseline = y_offset + self.font_size + 1
    
            self._add_text_line(svg, tokens, (start, end), self.margin, baseline)

            # Add metric lines
            sparkline.render(
                parent=svg,
                spans=spans,
                window=(start, end),  # The current line's range
                x=self.margin,
                y=baseline+1,
                h=self.sparkline_height)
        
        # Add legend
        self._add_legend(svg, width - 90 - toggle_space, self.margin, series)

        # Add light/dark support
        svg_theme_toggle(
            svg,
            toggle_pos=(width-20, 20),
            theme_vars={
                'col-series-1': ('#ef4444', '#ff7878'),
                'col-series-2': '#3b82f6',
                'col-series-3': ('#22c55e', '#45e881'),
                'col-series-4': ('#f97316', '#ffa261'),
                'col-series-5': ('#a855f7', '#d9b1ff'),
                'col-text': ('#666666', '#dddddd'),
                'col-baseline': ('#cccccc', '#666666'),
                'blend-mode': ('multiply', 'screen'),
            })

        return ET.tostring(svg, encoding='unicode')
