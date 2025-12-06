"""Interactive visualization of introspection experiment results using Plotly."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from introspection import analysis


def create_layer_effect_plot(  # pyright: ignore[reportUnknownVariableType]
    df: pd.DataFrame,
) -> go.Figure:
    """Create a plot showing score vs layer percentage across model scales."""
    agg = analysis.aggregate_scores(df)  # pyright: ignore[reportUnknownVariableType]

    # Filter to just the main grader prompt (first one if multiple exist)
    grader_prompts = agg["grader_prompt"].unique()  # pyright: ignore[reportUnknownMemberType]
    if len(grader_prompts) > 1:
        # Use the first grader prompt by default
        agg = agg[agg["grader_prompt"] == grader_prompts[0]]  # pyright: ignore[reportUnknownVariableType]

    # Separate intervention and control conditions
    intervention = agg[agg["condition"] == "intervention"]  # pyright: ignore[reportUnknownVariableType]
    control = agg[agg["condition"] == "control"]  # pyright: ignore[reportUnknownVariableType]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Intervention (Steered)", "Control (Baseline)"],
        shared_yaxes=True,
    )

    # Define a nice color palette for model scales
    color_map = {
        "8B": "#636EFA",
        "14B": "#EF553B",
        "32B": "#00CC96",
        "235B": "#AB63FA",
    }

    for model_scale in sorted(intervention["model_scale"].unique()):  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
        model_data = intervention[intervention["model_scale"] == model_scale]  # pyright: ignore[reportUnknownVariableType]
        color = color_map.get(model_scale, "#666666")
        fig.add_trace(
            go.Scatter(
                x=model_data["layer_percentage"],  # pyright: ignore[reportUnknownArgumentType]
                y=model_data["mean_score"],  # pyright: ignore[reportUnknownArgumentType]
                mode="lines+markers",
                name=f"{model_scale}",
                line={"color": color},
                marker={"size": 8},
                legendgroup=model_scale,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    for model_scale in sorted(control["model_scale"].unique()):  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
        model_data = control[control["model_scale"] == model_scale]  # pyright: ignore[reportUnknownVariableType]
        color = color_map.get(model_scale, "#666666")
        fig.add_trace(
            go.Scatter(
                x=model_data["layer_percentage"],  # pyright: ignore[reportUnknownArgumentType]
                y=model_data["mean_score"],  # pyright: ignore[reportUnknownArgumentType]
                mode="lines+markers",
                name=f"{model_scale}",
                line={"color": color, "dash": "dash"},
                marker={"size": 8},
                legendgroup=model_scale,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title="Effect of Steering by Layer Position",
        height=500,
        legend_title="Model Scale",
    )
    fig.update_xaxes(title_text="Layer Position (%)", row=1, col=1)
    fig.update_xaxes(title_text="Layer Position (%)", row=1, col=2)
    fig.update_yaxes(title_text="Mean Score", row=1, col=1)

    return fig


def create_intervention_vs_control_plot(df: pd.DataFrame) -> go.Figure:
    """Create a plot comparing intervention vs control scores."""
    agg: pd.DataFrame = analysis.aggregate_scores(df)

    # Pivot to get intervention and control side by side
    pivot: pd.DataFrame = agg.pivot_table(  # pyright: ignore[reportUnknownMemberType]
        index=["layer_percentage", "model_scale"],
        columns="condition",
        values="mean_score",
        aggfunc="mean",
    ).reset_index()

    fig = px.scatter(
        pivot,
        x="control",
        y="intervention",
        color="model_scale",
        hover_data=["layer_percentage"],
        labels={
            "control": "Control Score",
            "intervention": "Intervention Score",
            "model_scale": "Model Scale",
        },
        title="Intervention vs Control Performance",
    )

    # Add diagonal reference line
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line={"color": "gray", "dash": "dash"},
    )

    fig.update_layout(height=500)
    return fig


def create_heatmap_plot(df: pd.DataFrame) -> go.Figure:
    """Create a heatmap of scores across layer and model scale."""
    agg: pd.DataFrame = analysis.aggregate_scores(df)

    # Filter to intervention condition only
    intervention: pd.DataFrame = agg[agg["condition"] == "intervention"]

    # Pivot for heatmap
    pivot: pd.DataFrame = intervention.pivot_table(  # pyright: ignore[reportUnknownMemberType]
        index="model_scale",
        columns="layer_percentage",
        values="mean_score",
        aggfunc="mean",
    )

    # Sort model scales by size
    scale_order = ["8B", "14B", "32B", "235B"]
    existing_scales = [s for s in scale_order if s in pivot.index]
    pivot = pivot.reindex(existing_scales)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"{int(x)}%" for x in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            colorbar={"title": "Mean Score"},
        )
    )

    fig.update_layout(
        title="Score Heatmap: Model Scale × Layer Position (Intervention)",
        xaxis_title="Layer Position",
        yaxis_title="Model Scale",
        height=400,
    )

    return fig


def create_concept_analysis_plot(df: pd.DataFrame) -> go.Figure:
    """Create a plot showing per-concept performance."""
    # Group by concept and condition
    concept_scores: pd.DataFrame = (
        df.groupby(["concept", "condition"])["score"].mean().reset_index()  # pyright: ignore[reportUnknownMemberType]
    )

    # Pivot for easier plotting
    pivot = concept_scores.pivot(
        index="concept", columns="condition", values="score"
    ).reset_index()

    # Sort by intervention score
    if "intervention" in pivot.columns:
        pivot = pivot.sort_values("intervention", ascending=True)

    fig = go.Figure()

    if "intervention" in pivot.columns:
        fig.add_trace(
            go.Bar(
                y=pivot["concept"],
                x=pivot["intervention"],
                name="Intervention",
                orientation="h",
                marker_color="#636EFA",
            )
        )

    if "control" in pivot.columns:
        fig.add_trace(
            go.Bar(
                y=pivot["concept"],
                x=pivot["control"],
                name="Control",
                orientation="h",
                marker_color="#EF553B",
                opacity=0.7,
            )
        )

    fig.update_layout(
        title="Score by Concept",
        xaxis_title="Mean Score",
        yaxis_title="Concept",
        barmode="overlay",
        height=max(400, len(pivot) * 20),
        legend={"yanchor": "bottom", "y": 0.01, "xanchor": "right", "x": 0.99},
    )

    return fig


def create_full_dashboard(df: pd.DataFrame, output_path: Path) -> None:
    """Create a full HTML dashboard with all plots."""
    # Create individual plots
    layer_plot = create_layer_effect_plot(df)
    comparison_plot = create_intervention_vs_control_plot(df)
    heatmap_plot = create_heatmap_plot(df)
    concept_plot = create_concept_analysis_plot(df)

    # Calculate stats
    intervention_score = df[df["condition"] == "intervention"]["score"].mean()
    control_score = df[df["condition"] == "control"]["score"].mean()

    # Create combined HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Introspection Experiment Results</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }}
        .plot-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat {{
            display: inline-block;
            margin: 10px 20px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #636EFA;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>🔬 Introspection Experiment Results</h1>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <div class="stat">
            <div class="stat-value">{len(df)}</div>
            <div class="stat-label">Total Samples</div>
        </div>
        <div class="stat">
            <div class="stat-value">{df["model_scale"].nunique()}</div>
            <div class="stat-label">Model Scales</div>
        </div>
        <div class="stat">
            <div class="stat-value">{df["concept"].nunique()}</div>
            <div class="stat-label">Concepts</div>
        </div>
        <div class="stat">
            <div class="stat-value">{df["layer_percentage"].nunique()}</div>
            <div class="stat-label">Layer Positions</div>
        </div>
        <div class="stat">
            <div class="stat-value">{intervention_score:.2%}</div>
            <div class="stat-label">Intervention Avg Score</div>
        </div>
        <div class="stat">
            <div class="stat-value">{control_score:.2%}</div>
            <div class="stat-label">Control Avg Score</div>
        </div>
    </div>
    
    <div class="plot-container">
        <div id="layer-plot"></div>
    </div>
    
    <div class="plot-container">
        <div id="heatmap-plot"></div>
    </div>
    
    <div class="plot-container">
        <div id="comparison-plot"></div>
    </div>
    
    <div class="plot-container">
        <div id="concept-plot"></div>
    </div>
    
    <script>
        var layerPlot = {layer_plot.to_json()};
        var heatmapPlot = {heatmap_plot.to_json()};
        var comparisonPlot = {comparison_plot.to_json()};
        var conceptPlot = {concept_plot.to_json()};
        
        Plotly.newPlot('layer-plot', layerPlot.data, layerPlot.layout, {{responsive: true}});
        Plotly.newPlot('heatmap-plot', heatmapPlot.data, heatmapPlot.layout, {{responsive: true}});
        Plotly.newPlot('comparison-plot', comparisonPlot.data, comparisonPlot.layout, {{responsive: true}});
        Plotly.newPlot('concept-plot', conceptPlot.data, conceptPlot.layout, {{responsive: true}});
    </script>
</body>
</html>
"""

    output_path.write_text(html_content)
    print(f"Dashboard saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create interactive visualizations of introspection experiment results"
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs/",
        help="Directory containing eval logs (default: logs/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dashboard.html",
        help="Output HTML file path (default: dashboard.html)",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.logs_dir}...")
    df = analysis.load_and_process_data(args.logs_dir)
    print(f"Loaded {len(df)} samples")

    output_path = Path(args.output)
    create_full_dashboard(df, output_path)
    print(f"\nOpen {output_path} in a web browser to view the interactive dashboard.")


if __name__ == "__main__":
    main()
