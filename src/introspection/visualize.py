"""Interactive visualization of introspection experiment results using Plotly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from introspection import analysis


def create_interactive_dashboard(df: pd.DataFrame, output_path: Path) -> None:
    """Create a full HTML dashboard with dropdown filters."""
    # Create a sample identifier for joining coherent scores
    sample_cols = [
        "layer_percentage",
        "model_scale",
        "condition",
        "strength",
        "concept",
        "trial",
    ]

    # Pivot the data so each sample has columns for each grader_prompt score
    pivot_df = df.pivot_table(  # pyright: ignore[reportUnknownMemberType]
        index=sample_cols,
        columns="grader_prompt",
        values="score",
        aggfunc="first",
    ).reset_index()

    # Get grader prompts (excluding coherent_response for plotting)
    all_grader_prompts = [
        c
        for c in pivot_df.columns
        if c not in sample_cols  # pyright: ignore[reportUnknownMemberType]
    ]
    plot_grader_prompts = [g for g in all_grader_prompts if g != "coherent_response"]

    # Create two versions of aggregated data:
    # 1. Raw scores (no coherent filter)
    # 2. Coherent-filtered scores (only count if coherent_response == 1)

    # For raw data - aggregate each grader prompt independently
    raw_agg_list: list[pd.DataFrame] = []
    for grader in plot_grader_prompts:
        temp = pivot_df[sample_cols + [grader]].copy()  # pyright: ignore[reportUnknownMemberType]
        temp = temp.dropna(subset=[grader])  # pyright: ignore[reportUnknownMemberType]
        grouped = temp.groupby(  # pyright: ignore[reportUnknownMemberType]
            ["layer_percentage", "model_scale", "condition", "strength"],
            as_index=False,
        ).agg({grader: "mean"})
        grouped["grader_prompt"] = grader
        grouped = grouped.rename(columns={grader: "mean_score"})
        raw_agg_list.append(grouped)  # pyright: ignore[reportUnknownMemberType]

    raw_agg = pd.concat(raw_agg_list, ignore_index=True)  # pyright: ignore[reportUnknownArgumentType]

    # For coherent-filtered data - score = grader_score AND coherent_score
    # (non-coherent samples count as 0 for other grader prompts)
    coherent_filtered_list: list[pd.DataFrame] = []
    if "coherent_response" in pivot_df.columns:
        for grader in plot_grader_prompts:
            temp = pivot_df[sample_cols + [grader, "coherent_response"]].copy()  # pyright: ignore[reportUnknownMemberType]
            temp = temp.dropna(subset=[grader])  # pyright: ignore[reportUnknownMemberType]
            if len(temp) == 0:
                continue
            # Compute AND: score is 1 only if both grader and coherent are 1
            temp["combined_score"] = (temp[grader] * temp["coherent_response"]).astype(
                float
            )  # pyright: ignore[reportUnknownMemberType]
            grouped = temp.groupby(  # pyright: ignore[reportUnknownMemberType]
                ["layer_percentage", "model_scale", "condition", "strength"],
                as_index=False,
            ).agg({"combined_score": "mean"})
            grouped["grader_prompt"] = grader
            grouped = grouped.rename(columns={"combined_score": "mean_score"})
            coherent_filtered_list.append(grouped)  # pyright: ignore[reportUnknownMemberType]

        coherent_agg = pd.concat(coherent_filtered_list, ignore_index=True)  # pyright: ignore[reportUnknownArgumentType]
    else:
        coherent_agg = raw_agg.copy()

    # Convert to JSON-serializable format
    raw_records = raw_agg.to_dict(orient="records")  # pyright: ignore[reportUnknownMemberType]
    coherent_records = coherent_agg.to_dict(orient="records")  # pyright: ignore[reportUnknownMemberType]

    # Get unique values for dropdowns
    conditions = sorted(df["condition"].unique().tolist())  # pyright: ignore[reportUnknownMemberType]
    model_scales = sorted(
        df["model_scale"].unique().tolist(),  # pyright: ignore[reportUnknownMemberType]
        key=lambda x: float(x.replace("B", "")),
    )
    strengths = sorted(df["strength"].unique().tolist())  # pyright: ignore[reportUnknownMemberType]

    # Calculate stats
    intervention_score = df[df["condition"] == "intervention"]["score"].mean()
    control_score = df[df["condition"] == "control"]["score"].mean()

    # Color palette for grader prompts
    colors = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
    ]

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
        .controls {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        .control-group label {{
            font-weight: 600;
            color: #555;
            font-size: 14px;
        }}
        .control-group select {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            min-width: 150px;
            background: white;
        }}
        .checkbox-group {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding-top: 20px;
        }}
        .checkbox-group input {{
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        .checkbox-group label {{
            font-weight: 600;
            color: #555;
            font-size: 14px;
            cursor: pointer;
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
        .info {{
            color: #666;
            font-size: 13px;
            margin-top: 10px;
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

    <div class="controls">
        <div class="control-group">
            <label for="condition-select">Condition</label>
            <select id="condition-select">
                {"".join(f'<option value="{c}">{c}</option>' for c in conditions)}
            </select>
        </div>
        <div class="control-group">
            <label for="model-select">Model Size</label>
            <select id="model-select">
                {"".join(f'<option value="{m}">{m}</option>' for m in model_scales)}
            </select>
        </div>
        <div class="control-group">
            <label for="strength-select">Strength</label>
            <select id="strength-select">
                {"".join(f'<option value="{s}">{s}</option>' for s in strengths)}
            </select>
        </div>
        <div class="checkbox-group">
            <input type="checkbox" id="coherent-filter" checked>
            <label for="coherent-filter">Require coherent response</label>
        </div>
        <div class="info">
            💡 Click legend items to toggle grader prompts on/off
        </div>
    </div>
    
    <div class="plot-container">
        <div id="main-plot"></div>
    </div>
    
    <script>
        // Data with and without coherent filtering
        const rawData = {json.dumps(raw_records)};
        const coherentData = {json.dumps(coherent_records)};
        const graderPrompts = {json.dumps(plot_grader_prompts)};
        const colors = {json.dumps(colors)};
        
        // Track visibility state for each grader prompt (true = visible)
        const visibilityState = {{}};
        graderPrompts.forEach(prompt => visibilityState[prompt] = true);
        
        function updatePlot() {{
            const condition = document.getElementById('condition-select').value;
            const model = document.getElementById('model-select').value;
            const strength = document.getElementById('strength-select').value;
            const requireCoherent = document.getElementById('coherent-filter').checked;
            
            // Choose data source based on coherent filter
            const dataSource = requireCoherent ? coherentData : rawData;
            
            // Filter data
            let filtered = dataSource.filter(d => 
                d.condition === condition &&
                d.model_scale === model &&
                d.strength === parseFloat(strength)
            );
            
            // Group by grader_prompt and create traces
            const traces = [];
            graderPrompts.forEach((prompt, i) => {{
                const promptData = filtered.filter(d => d.grader_prompt === prompt);
                if (promptData.length === 0) return;
                
                // Sort by layer_percentage
                promptData.sort((a, b) => a.layer_percentage - b.layer_percentage);
                
                const x = promptData.map(d => d.layer_percentage);
                const y = promptData.map(d => d.mean_score);
                
                traces.push({{
                    x: x,
                    y: y,
                    mode: 'lines+markers',
                    name: prompt,
                    visible: visibilityState[prompt] ? true : 'legendonly',
                    line: {{ color: colors[i % colors.length] }},
                    marker: {{ size: 6 }}
                }});
            }});
            
            const coherentLabel = requireCoherent ? ' (coherent only)' : '';
            const layout = {{
                title: `Score by Layer Position (${{condition}})${{coherentLabel}}`,
                xaxis: {{ title: 'Layer Position (%)', range: [0, 100] }},
                yaxis: {{ title: 'Mean Score', range: [0, 1] }},
                height: 600,
                legend: {{
                    title: {{ text: 'Grader Prompt' }},
                    orientation: 'h',
                    y: -0.2
                }},
                hovermode: 'closest'
            }};
            
            Plotly.react('main-plot', traces, layout, {{responsive: true}});
        }}
        
        // Add event listeners for dropdowns
        document.getElementById('condition-select').addEventListener('change', updatePlot);
        document.getElementById('model-select').addEventListener('change', updatePlot);
        document.getElementById('strength-select').addEventListener('change', updatePlot);
        document.getElementById('coherent-filter').addEventListener('change', updatePlot);
        
        // Initial plot
        updatePlot();
        
        // Listen for legend clicks to track visibility state (must be after initial plot)
        document.getElementById('main-plot').on('plotly_legendclick', function(data) {{
            const traceName = data.data[data.curveNumber].name;
            visibilityState[traceName] = !visibilityState[traceName];
        }});
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
    create_interactive_dashboard(df, output_path)
    print(f"\nOpen {output_path} in a web browser to view the interactive dashboard.")


if __name__ == "__main__":
    main()
