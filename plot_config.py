# Author: Kaifeng ZHU
# This file is used to configure the plotly theme for the energy optimization research.
# Scientific Plotly theme for energy optimization research (Inter font)

from __future__ import annotations

from typing import Dict, List, Optional
import plotly.io as pio


# ----------------------------
# Color palette (color-blind friendly, print-friendly)
# ----------------------------
ENERGY_COLORS: Dict[str, str] = {
    "battery": "#1f4e79",   # Deep blue
    "pv": "#f39c12",        # Solar orange
    "grid": "#7f8c8d",      # Neutral gray
    "cost": "#8b0000",      # Dark red
    "soc": "#008080",       # Teal
    "soh": "#6a0dad",       # Purple
    "black": "#000000",
}


# Optional multi-series palette (when you need many distinct lines)
SCIENTIFIC_SEQ: List[str] = [
    "#1f4e79",
    "#f39c12",
    "#008080",
    "#6a0dad",
    "#8b0000",
    "#7f8c8d",
    "#2c3e50",
    "#16a085",
]


def apply_scientific_theme(
    template_name: str = "scientific_inter",
    base_font_size: int = 14,
    title_font_size: int = 20,
    axis_title_font_size: int = 16,
    tick_font_size: int = 14,
    legend_font_size: int = 13,
    grid_rgba: str = "rgba(200,200,200,0.35)",
    line_width: float = 2.5,
) -> None:
    """
    Register and set a professional, clean Plotly template for scientific figures.
    Usage:
        from plotly_scientific_template import apply_scientific_theme, ENERGY_COLORS
        apply_scientific_theme()
    """

    pio.templates[template_name] = dict(
        layout=dict(
            # Fonts
            font=dict(
                family="Inter, Helvetica, Arial",
                size=base_font_size,
                color="black",
            ),
            title=dict(
                font=dict(size=title_font_size),
                x=0.5,
                xanchor="center",
            ),

            # Axes styling
            xaxis=dict(
                title_font=dict(size=axis_title_font_size),
                tickfont=dict(size=tick_font_size),
                showgrid=True,
                gridcolor=grid_rgba,
                zeroline=False,
                showline=True,
                linecolor="black",
                mirror=True,
                ticks="outside",
                ticklen=6,
            ),
            yaxis=dict(
                title_font=dict(size=axis_title_font_size),
                tickfont=dict(size=tick_font_size),
                showgrid=True,
                gridcolor=grid_rgba,
                zeroline=False,
                showline=True,
                linecolor="black",
                mirror=True,
                ticks="outside",
                ticklen=6,
            ),

            # Legend styling
            legend=dict(
                font=dict(size=legend_font_size),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),

            # Background and margins
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=80, r=40, t=80, b=60),

            # Default discrete colors
            colorway=SCIENTIFIC_SEQ,
        )
    )

    # Set as default template
    pio.templates.default = template_name

    # Store a few defaults for convenience (can be used by your plot functions)
    # Note: Plotly doesn't have a global "default line width" in template that covers all traces
    # consistently, so we expose line_width for your own wrappers.
    globals()["DEFAULT_LINE_WIDTH"] = line_width


def style_trace_line(color: str, width: Optional[float] = None, dash: str = "solid") -> Dict:
    """
    Convenience helper: standard line style for traces.
    """
    lw = width if width is not None else globals().get("DEFAULT_LINE_WIDTH", 2.5)
    return dict(color=color, width=lw, dash=dash)


def style_marker(color: str, size: int = 6, symbol: str = "circle", line_width: int = 0) -> Dict:
    """
    Convenience helper: standard marker style for traces.
    """
    return dict(color=color, size=size, symbol=symbol, line=dict(width=line_width))


def finalize_figure(
    fig,
    title: Optional[str] = None,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    y2_title: Optional[str] = None,
    show_legend: bool = True,
    legend_top: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """
    Apply consistent layout tweaks to a figure created anywhere.
    """

    layout_updates = dict(showlegend=show_legend)
    if title is not None:
        layout_updates["title"] = dict(text=title)

    if width is not None:
        layout_updates["width"] = width
    if height is not None:
        layout_updates["height"] = height

    fig.update_layout(**layout_updates)

    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)

    if y2_title is not None:
        fig.update_layout(
            yaxis2=dict(
                title=y2_title,
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor="black",
                mirror=True,
            )
        )

    if legend_top:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            )
        )
