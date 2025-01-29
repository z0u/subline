from typing import Optional
import xml.etree.ElementTree as ET
import numpy as np
from scipy import stats
import torch

from .series import Series
from .utils.dom import Element


class KDEPlot:
    """A multi-series KDE plot handler."""
    
    def __init__(self):
        self.series: list[Series] = []
        self.palette = [
            '#3b82f6',  # blue
            '#ef4444',  # red
            '#22c55e',  # green
            '#f97316',  # orange
            '#a855f7',  # purple
        ]
        self.fill_opacity = 0.05
        self.stroke_width = 1.0
    
    def add_series(self, values: torch.Tensor, color: Optional[str] = None):
        """Add a new data series to the plot."""
        self.series.append(Series(values, color))
        return self
    
    def render(self, parent: ET.Element, x=0.0, y=0.0, w=60.0, h=30.0):
        """Render all series into the plot at the specified position."""
        # Create a transformation group if we're not at 0,0
        if x != 0 or y != 0:
            parent = Element(parent, "g", transform=f"translate({x}, {y})")
            
        # Render the series
        palette = iter(self.palette)
        for series in self.series:
            color = series.color or next(palette)
            x_vals, y_vals = self._compute(series.values, w, h)
            self._render_series(parent, x_vals, y_vals, w, h, color)
    
        # Draw the x-axis
        Element(parent, "line",
            x1=0, y1=h, x2=w, y2=h,
            stroke="var(--col-baseline)",
            shape_rendering="crispEdges",
            style="mix-blend-mode: var(--blend-mode);")
            
        return self
    
    def _compute(self, values, width: int, height: int):
        """Compute KDE for the series."""
        valid_values = values[torch.isfinite(values)].numpy()
        if len(valid_values) < 2:
            x = np.array([0, 1])
            y = np.array([0, 0])
            return x, y

        kde = stats.gaussian_kde(valid_values)
        x = np.linspace(valid_values.min(), valid_values.max(), width)
        y = kde(x)
        y = height * (y / y.max())
        return x, y
    
    def _render_series(self, parent: ET.Element, xs: torch.Tensor, ys: torch.Tensor, w: float, h: float, color: str):
        """Render a single KDE series."""
        # Create path data for the line
        points = []
        x_scale = w / (xs[-1] - xs[0])
        for x,y in zip(xs,ys):
            px = (x - xs[0]) * x_scale
            py = h - y
            points.append(f"{px:4g},{py:4g}")
        line_data = 'M' + 'L'.join(points)

        # Create filled shape by extending to base
        points.append(f"{w:4g},{h:4g}")
        points.append(f"{0},{h:4g}")
        filled_shape_data = 'M' + 'L'.join(points)

        # Add the filled area and line for this series
        Element(parent, "path",
            d=filled_shape_data,
            fill=color,
            opacity=self.fill_opacity,
            style="mix-blend-mode: var(--blend-mode);")
               
        Element(parent, "path",
            d=line_data,
            stroke=color,
            stroke_width=self.stroke_width,
            fill="none",
            style="mix-blend-mode: var(--blend-mode);")
