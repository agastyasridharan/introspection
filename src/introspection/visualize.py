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

    # Get unique values for dropdowns
    model_scales = sorted(
        df["model_scale"].unique().tolist(),  # pyright: ignore[reportUnknownMemberType]
        key=lambda x: float(x.replace("B", "")),
    )
    strengths = sorted(df["strength"].unique().tolist())  # pyright: ignore[reportUnknownMemberType]

    # --- Compute delta (intervention - control) ---
    # Use coherent-gated data as the basis
    source_agg = coherent_agg if len(coherent_filtered_list) > 0 else raw_agg
    merge_keys = ["layer_percentage", "model_scale", "strength", "grader_prompt"]
    intervention_agg = source_agg[source_agg["condition"] == "intervention"]  # pyright: ignore[reportUnknownMemberType]
    control_agg = source_agg[source_agg["condition"] == "control"]  # pyright: ignore[reportUnknownMemberType]
    delta_df = intervention_agg.merge(  # pyright: ignore[reportUnknownMemberType]
        control_agg[merge_keys + ["mean_score"]],
        on=merge_keys,
        suffixes=("_intervention", "_control"),
    )
    delta_df["delta"] = (
        delta_df["mean_score_intervention"] - delta_df["mean_score_control"]
    )  # pyright: ignore[reportUnknownMemberType]
    delta_records = delta_df.to_dict(orient="records")  # pyright: ignore[reportUnknownMemberType]

    # Both-conditions data for grader comparison overlay (tab 2 section 4)
    both_conditions_records = source_agg.to_dict(orient="records")  # pyright: ignore[reportUnknownMemberType]

    # Compute summary stats for the explainer
    n_concepts = df["concept"].nunique()
    n_layers = df["layer_percentage"].nunique()
    n_trials = df["trial"].nunique()
    strength_range = f"{min(strengths)}&ndash;{max(strengths)}"

    # Color palette for grader prompts
    colors = [
        "#4a7c59",
        "#c25450",
        "#4a6fa5",
        "#8b6d3f",
        "#7a5c8a",
        "#5a8a7a",
        "#a0522d",
        "#6b7b8d",
    ]

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Introspection Experiment Results</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        *, *::before, *::after {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'EB Garamond', 'Georgia', 'Times New Roman', serif;
            font-size: 16px;
            max-width: 1800px;
            margin: 0 auto;
            padding: 24px 32px;
            background-color: #faf6f1;
            color: #2c2416;
            line-height: 1.55;
        }}
        h1 {{
            color: #1a1408;
            margin-bottom: 24px;
            font-weight: 500;
            font-size: 1.6rem;
            letter-spacing: -0.02em;
        }}
        h2 {{
            color: #2c2416;
            font-weight: 500;
            font-size: 1.35rem;
            letter-spacing: -0.01em;
        }}
        .subtitle {{
            font-size: 0.9rem;
            font-style: italic;
            color: #6b5d4d;
            margin-bottom: 10px;
        }}
        .subtitle a {{
            color: #4a6fa5;
            text-decoration: none;
        }}
        .subtitle a:hover {{
            text-decoration: underline;
        }}
        .panel {{
            background: #fffdf9;
            border: 1px solid #e8e0d4;
            border-radius: 6px;
            box-shadow: 0 2px 6px rgba(44, 36, 22, 0.05);
        }}
        .controls {{
            background: #fffdf9;
            border: 1px solid #e8e0d4;
            border-radius: 6px;
            box-shadow: 0 2px 6px rgba(44, 36, 22, 0.05);
            padding: 14px 18px;
            margin-bottom: 24px;
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .control-group label {{
            font-weight: 600;
            color: #6b5d4d;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .control-group select {{
            font-family: 'EB Garamond', 'Georgia', 'Times New Roman', serif;
            padding: 6px 10px;
            border: 1px solid #e8e0d4;
            border-radius: 4px;
            font-size: 14px;
            min-width: 140px;
            background: #fffdf9;
            color: #2c2416;
        }}
        .plot-container {{
            background: #fffdf9;
            border: 1px solid #e8e0d4;
            border-radius: 6px;
            box-shadow: 0 2px 6px rgba(44, 36, 22, 0.05);
            padding: 16px;
            margin-bottom: 24px;
        }}
        .section-header {{
            font-size: 1.2rem;
            font-weight: 500;
            color: #2c2416;
            letter-spacing: -0.01em;
            margin: 28px 0 8px 0;
            padding-bottom: 5px;
            border-bottom: 1px solid #e8e0d4;
        }}
        .section-desc {{
            color: #6b5d4d;
            font-size: 15px;
            margin-bottom: 14px;
            line-height: 1.6;
        }}
        .explainer {{
            background: #fffdf9;
            border: 1px solid #e8e0d4;
            border-radius: 6px;
            box-shadow: 0 2px 6px rgba(44, 36, 22, 0.05);
            padding: 22px 28px;
            margin-bottom: 24px;
            line-height: 1.65;
            font-size: 16px;
        }}
        .explainer p {{
            margin: 0 0 10px 0;
        }}
        .explainer p:last-child {{
            margin-bottom: 0;
        }}
        .explainer strong {{
            color: #2c2416;
            font-weight: 600;
        }}
        .term {{
            background: #f0ebe3;
            padding: 1px 6px;
            border-radius: 3px;
            font-size: 14px;
            white-space: nowrap;
        }}
        .model-legend {{
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(250, 246, 241, 0.95);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            border-bottom: 1px solid #e8e0d4;
            padding: 10px 20px;
            margin: 0 -32px 16px -32px;
            display: flex;
            align-items: center;
            gap: 24px;
            flex-wrap: wrap;
            font-size: 15px;
        }}
        .model-legend-label {{
            font-weight: 600;
            color: #6b5d4d;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .legend-swatch {{
            display: inline-block;
            width: 16px;
            height: 3px;
            border-radius: 1.5px;
            vertical-align: middle;
            margin-right: 5px;
        }}
        .legend-item {{
            color: #5a4a3a;
            font-size: 15px;
        }}
        .grid-container {{
            display: grid;
            gap: 14px;
            margin-bottom: 24px;
        }}
        .grid-cell {{
            background: #fffdf9;
            border: 1px solid #e8e0d4;
            border-radius: 6px;
            box-shadow: 0 2px 6px rgba(44, 36, 22, 0.05);
            padding: 10px;
        }}
    </style>
</head>
<body>
    <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px;">
        <h1 style="margin-bottom:0;">Introspection Experiment Results</h1>
        <span class="subtitle">Agastya Sridharan <a href="https://github.com/agastyasridharan" style="color:#6b5d4d; text-decoration:none; vertical-align:middle; margin-left:3px; transition:color 0.2s;" onmouseover="this.style.color='#2c2416'" onmouseout="this.style.color='#6b5d4d'"><svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" style="vertical-align:-2px;"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg></a></span>
    </div>
    <p class="subtitle" style="font-style:normal; margin-top:-4px;">
        Inspired by Anthropic&rsquo;s
        <a href="https://arxiv.org/abs/2601.01828">original paper</a>
        and this codebase&rsquo;s
        <a href="https://github.com/neevparikh/introspection">experiments</a>.
    </p>

    <div>

        <div class="explainer">
            <p><strong>The Experiment.</strong>
            We test whether language models can detect when a researcher injects a concept-specific
            activation pattern into their residual stream during inference. For each of
            <strong>{n_concepts} concept words</strong> (e.g. &ldquo;oceans,&rdquo; &ldquo;lightning,&rdquo;
            &ldquo;algorithms&rdquo;), we first extract a <em>steering vector</em>, which is the
            direction in activation space that distinguishes &ldquo;thinking about X&rdquo; from
            generic text processing. We then inject that vector into the model&rsquo;s hidden states
            while it answers: &ldquo;Do you detect an injected thought? If so, what is it about?&rdquo;</p>

            <p><strong>Control vs. Intervention.</strong>
            Every trial is paired: the <span class="term">control</span> response uses identical
            random seeds but no injection, whereas the <span class="term">intervention</span> response
            has the steering vector active. The difference between the two isolates the causal
            effect of the injection from prompt compliance or sampling noise.</p>

            <p><strong>Key parameters.</strong>
            <span class="term">Layer position (%)</span> is where in the transformer the vector is
            injected, normalized to 0&ndash;100% so different model sizes can be compared.
            Early layers (~0&ndash;20%) handle syntax, middle layers (~30&ndash;60%) compose features,
            and late layers (~70&ndash;100%) directly bias token prediction.
            <span class="term">Injection strength</span> ({strength_range}&times;) scales the
            steering vector&rsquo;s magnitude relative to natural activations: higher
            values force a stronger perturbation but may degrade coherence.
            Each configuration was run for <strong>{n_trials} trials</strong> across
            <strong>{n_layers} layer positions</strong>.</p>

            <p><strong>Grading.</strong>
            An external LLM (GPT-4) grades each response on four criteria of increasing strictness:</p>
            <p style="margin-left: 20px;">
            1.&ensp;<span class="term">coherent_response</span>: Is the output coherent? This is a sanity gate, and incoherent responses are excluded from other scores.<br>
            2.&ensp;<span class="term">affirmative_response</span>: Does the model claim to detect <em>any</em> injected thought? This measures prompt compliance.<br>
            3.&ensp;<span class="term">thinking_about_word</span>: Does the model describe thinking about the <em>specific</em> concept word? This measures behavioral leakage.<br>
            4.&ensp;<span class="term">affirmative + correct ID</span>: Does the model claim detection <em>and</em> correctly name the concept, in that order? This is the strictest criterion, designed to filter out confabulation.</p>

            <p><strong>Delta score.</strong>
            All plots below show
            <span class="term">&Delta; = intervention score &minus; control score</span>.
            A positive delta means the model detects the injection above its false-positive baseline,
            whereas a delta near zero means the model cannot tell the difference.
            All scores are coherence-gated, meaning a response only counts as a detection if it is also coherent.</p>
        </div>

        <div class="model-legend" id="sticky-legend" style="visibility:hidden; opacity:0; transition: opacity 0.2s ease;">
            <span class="model-legend-label">Model scale:</span>
            {"".join(f'<span class="legend-item"><span class="legend-swatch" style="background:{colors[i % len(colors)]};"></span>{m}</span>' for i, m in enumerate(model_scales))}
        </div>

        <div id="legend-sections">
        <div class="section-header">1. Detection by Injection Strength</div>
        <div class="section-desc">
            The <span class="term">affirmative_response_followed_by_correct_identification</span>
            delta, separated by injection strength.
            Each panel shows one strength value, and each line is a model scale.
        </div>
        <div class="grid-container" id="small-multiples-grid"></div>

        <div class="section-header">2. Detection Signal: Affirmative Response Followed by Correct Identification</div>
        <div class="section-desc">
            This plot shows the delta for the <span class="term">affirmative_response_followed_by_correct_identification</span>
            grader, averaged across all injection strengths.
            This grader requires the model to both claim it detects an injection and correctly
            name the concept, in that order. Each line represents a different model scale. Positive values above the
            dotted red baseline indicate the model detects the injected concept above chance.
        </div>
        <div class="plot-container">
            <div id="hero-delta-plot"></div>
        </div>

        </div><!-- end legend-sections -->

        <div class="section-header">3. Sweet Spot Heatmap</div>
        <div class="section-desc">
            For each model, the raw intervention detection rate for
            <span class="term">affirmative_response_followed_by_correct_identification</span>
            (coherence-gated) as a function of layer position (x-axis) and injection strength (y-axis).
            Darker green indicates a higher detection rate. The region with the deepest
            color corresponds to the optimal (layer, strength) combination.
        </div>
        <div class="grid-container" id="heatmap-grid"></div>

        <div class="section-header">4. Grader Comparison: What Kind of Signal?</div>
        <div class="section-desc">
            This plot shows all three non-coherence graders together for a single model and strength.
            The <strong>solid lines</strong> represent scores where there was an intervention,
            and the <strong>dashed lines</strong> represent the control condition with no injection.
            The gap between solid and dashed lines is the causal effect of injection.
            If only the <span class="term">affirmative</span> grader is high, the model is just saying &ldquo;yes&rdquo; (prompt compliance).
            If <span class="term">thinking_about_word</span> is also high, the concept is leaking into the output.
            If <span class="term">affirmative + correct ID</span> is high, the model may genuinely be detecting the injection.
        </div>
        <div class="controls">
            <div class="control-group">
                <label for="a-model-select">Model Size</label>
                <select id="a-model-select">
                    {"".join(f'<option value="{m}">{m}</option>' for m in model_scales)}
                </select>
            </div>
            <div class="control-group">
                <label for="a-strength-select">Strength</label>
                <select id="a-strength-select">
                    {"".join(f'<option value="{s}">{s}</option>' for s in strengths)}
                </select>
            </div>
        </div>
        <div class="plot-container">
            <div id="grader-comparison-title" style="text-align:center; font-size:16px; color:#2c2416; margin-bottom:4px;"></div>
            <div id="grader-comparison-plot"></div>
        </div>


    <script>
        const graderPrompts = {json.dumps(plot_grader_prompts)};
        const colors = {json.dumps(colors)};
        const deltaData = {json.dumps(delta_records)};
        const bothConditionsData = {json.dumps(both_conditions_records)};
        const modelScales = {json.dumps(model_scales)};
        const allStrengths = {json.dumps(strengths)};

        const plotlyFont = {{ family: 'EB Garamond, Georgia, serif' }};
        const plotlyBg = '#fffdf9';
        const plotlyGrid = '#e8e0d4';
        const plotlyText = '#2c2416';
        const plotlyMuted = '#5a4a3a';
        const modelColors = {{}};
        modelScales.forEach((m, i) => modelColors[m] = colors[i % colors.length]);

        // Render all sections on load
        renderHeroDelta();
        renderSmallMultiples();
        renderHeatmaps();
        renderGraderComparison();
        document.getElementById('a-model-select').addEventListener('change', renderGraderComparison);
        document.getElementById('a-strength-select').addEventListener('change', renderGraderComparison);

        // Show/hide model legend only when sections 1-2 are in view
        const legendEl = document.getElementById('sticky-legend');
        const legendSections = document.getElementById('legend-sections');
        const legendObserver = new IntersectionObserver((entries) => {{
            entries.forEach(e => {{
                legendEl.style.visibility = e.isIntersecting ? 'visible' : 'hidden';
                legendEl.style.opacity = e.isIntersecting ? '1' : '0';
            }});
        }}, {{ threshold: 0 }});
        legendObserver.observe(legendSections);

        // --- 1. Hero delta plot ---
        function renderHeroDelta() {{
            const strictGrader = 'affirmative_response_followed_by_correct_identification';
            const traces = [];
            modelScales.forEach(model => {{
                // Average delta across all strengths for the strict grader
                const mData = deltaData.filter(d =>
                    d.grader_prompt === strictGrader && d.model_scale === model
                );
                if (mData.length === 0) return;
                // Group by layer_percentage, average across strengths
                const byLayer = {{}};
                mData.forEach(d => {{
                    if (!byLayer[d.layer_percentage]) byLayer[d.layer_percentage] = [];
                    byLayer[d.layer_percentage].push(d.delta);
                }});
                const layers = Object.keys(byLayer).map(Number).sort((a,b) => a - b);
                const means = layers.map(l => {{
                    const vals = byLayer[l];
                    return vals.reduce((a,b) => a+b, 0) / vals.length;
                }});
                traces.push({{
                    x: layers, y: means,
                    mode: 'lines+markers',
                    name: model,
                    line: {{ color: modelColors[model], width: 2.5 }},
                    marker: {{ size: 7 }}
                }});
            }});
            Plotly.newPlot('hero-delta-plot', traces, {{
                title: {{ text: 'Affirmative Response & Correct ID Delta (Intervention \\u2212 Control), averaged across strengths', font: {{ ...plotlyFont, size: 16, color: plotlyText }} }},
                xaxis: {{ title: {{ text: 'Layer Position (%)', font: {{ ...plotlyFont, size: 14, color: plotlyMuted }} }}, range: [0, 100], gridcolor: plotlyGrid, tickfont: {{ ...plotlyFont, size: 12, color: plotlyMuted }} }},
                yaxis: {{ title: {{ text: '\\u0394 Score', font: {{ ...plotlyFont, size: 14, color: plotlyMuted }} }}, gridcolor: plotlyGrid, tickfont: {{ ...plotlyFont, size: 12, color: plotlyMuted }}, zeroline: true, zerolinecolor: '#c25450', zerolinewidth: 1.5 }},
                height: 460,
                margin: {{ t: 44, b: 56, l: 56, r: 24 }},
                plot_bgcolor: plotlyBg, paper_bgcolor: plotlyBg,
                legend: {{ font: {{ ...plotlyFont, size: 13, color: plotlyMuted }}, orientation: 'h', y: -0.22 }},
                hovermode: 'closest',
                shapes: [{{ type: 'line', x0: 0, x1: 100, y0: 0, y1: 0, line: {{ color: '#c25450', width: 1, dash: 'dot' }} }}]
            }}, {{responsive: true}});
        }}

        // --- 2. Small multiples: one cell per strength ---
        function renderSmallMultiples() {{
            const strictGrader = 'affirmative_response_followed_by_correct_identification';
            const grid = document.getElementById('small-multiples-grid');
            grid.style.gridTemplateColumns = `repeat(${{Math.min(allStrengths.length, 3)}}, 1fr)`;
            allStrengths.forEach((strength, sIdx) => {{
                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                const plotDiv = document.createElement('div');
                plotDiv.id = `sm-plot-${{sIdx}}`;
                cell.appendChild(plotDiv);
                grid.appendChild(cell);
                const traces = [];
                modelScales.forEach(model => {{
                    const mData = deltaData.filter(d =>
                        d.grader_prompt === strictGrader &&
                        d.model_scale === model &&
                        d.strength === strength
                    );
                    if (mData.length === 0) return;
                    mData.sort((a,b) => a.layer_percentage - b.layer_percentage);
                    traces.push({{
                        x: mData.map(d => d.layer_percentage),
                        y: mData.map(d => d.delta),
                        mode: 'lines+markers',
                        name: model,
                        line: {{ color: modelColors[model], width: 2 }},
                        marker: {{ size: 5 }},
                        showlegend: false
                    }});
                }});
                Plotly.newPlot(plotDiv, traces, {{
                    title: {{ text: `Strength ${{strength}}`, font: {{ ...plotlyFont, size: 14, color: plotlyText }} }},
                    xaxis: {{ title: {{ text: 'Layer %', font: {{ ...plotlyFont, size: 12, color: plotlyMuted }}, standoff: 6 }}, range: [0, 100], gridcolor: plotlyGrid, tickfont: {{ ...plotlyFont, size: 11, color: plotlyMuted }} }},
                    yaxis: {{ title: {{ text: '\\u0394 Score', font: {{ ...plotlyFont, size: 12, color: plotlyMuted }} }}, gridcolor: plotlyGrid, tickfont: {{ ...plotlyFont, size: 11, color: plotlyMuted }}, zeroline: true, zerolinecolor: '#c25450', zerolinewidth: 1 }},
                    height: 250, margin: {{ t: 32, b: 36, l: 48, r: 16 }},
                    plot_bgcolor: plotlyBg, paper_bgcolor: plotlyBg,
                    showlegend: false,
                    hovermode: 'closest',
                    shapes: [{{ type: 'line', x0: 0, x1: 100, y0: 0, y1: 0, line: {{ color: '#c25450', width: 1, dash: 'dot' }} }}]
                }}, {{responsive: true}});
            }});
        }}

        // --- 3. Heatmaps: one per model scale ---
        function renderHeatmaps() {{
            const strictGrader = 'affirmative_response_followed_by_correct_identification';
            const grid = document.getElementById('heatmap-grid');
            grid.style.gridTemplateColumns = `repeat(${{Math.min(modelScales.length, 2)}}, 1fr)`;
            // Use intervention scores from bothConditionsData
            modelScales.forEach((model, mIdx) => {{
                const mData = bothConditionsData.filter(d =>
                    d.grader_prompt === strictGrader &&
                    d.model_scale === model &&
                    d.condition === 'intervention'
                );
                if (mData.length === 0) return;
                const layers = [...new Set(mData.map(d => d.layer_percentage))].sort((a,b) => a - b);
                const strs = [...new Set(mData.map(d => d.strength))].sort((a,b) => a - b);
                // Build z matrix: rows = strengths, cols = layers
                const z = strs.map(s => layers.map(l => {{
                    const match = mData.find(d => d.layer_percentage === l && d.strength === s);
                    return match ? match.mean_score : null;
                }}));
                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                const plotDiv = document.createElement('div');
                plotDiv.id = `hm-plot-${{mIdx}}`;
                cell.appendChild(plotDiv);
                grid.appendChild(cell);
                Plotly.newPlot(plotDiv, [{{
                    x: layers.map(l => l.toFixed(1) + '%'),
                    y: strs.map(s => s.toString()),
                    z: z,
                    type: 'heatmap',
                    colorscale: [[0, '#fffdf9'], [0.5, '#a8c6a0'], [1, '#2d5a3d']],
                    colorbar: {{ title: {{ text: 'Score', font: {{ ...plotlyFont, size: 11 }} }}, tickfont: {{ ...plotlyFont, size: 10 }} }},
                    hoverongaps: false
                }}], {{
                    title: {{ text: model, font: {{ ...plotlyFont, size: 15, color: plotlyText }} }},
                    xaxis: {{ title: {{ text: 'Layer Position', font: {{ ...plotlyFont, size: 12, color: plotlyMuted }} }}, tickfont: {{ ...plotlyFont, size: 10, color: plotlyMuted }} }},
                    yaxis: {{ title: {{ text: 'Strength', font: {{ ...plotlyFont, size: 12, color: plotlyMuted }} }}, tickfont: {{ ...plotlyFont, size: 10, color: plotlyMuted }} }},
                    height: 300, margin: {{ t: 36, b: 52, l: 60, r: 16 }},
                    plot_bgcolor: plotlyBg, paper_bgcolor: plotlyBg
                }}, {{responsive: true}});
            }});
        }}

        // --- 4. Grader comparison: solid=intervention, dashed=control ---
        function renderGraderComparison() {{
            const model = document.getElementById('a-model-select').value;
            const strength = parseFloat(document.getElementById('a-strength-select').value);
            const traces = [];
            graderPrompts.forEach((grader, i) => {{
                ['intervention', 'control'].forEach(cond => {{
                    const cData = bothConditionsData.filter(d =>
                        d.grader_prompt === grader &&
                        d.model_scale === model &&
                        d.strength === strength &&
                        d.condition === cond
                    );
                    if (cData.length === 0) return;
                    cData.sort((a,b) => a.layer_percentage - b.layer_percentage);
                    traces.push({{
                        x: cData.map(d => d.layer_percentage),
                        y: cData.map(d => d.mean_score),
                        mode: 'lines+markers',
                        name: `${{grader}} (${{cond}})`,
                        line: {{ color: colors[i % colors.length], width: 2, dash: cond === 'control' ? 'dash' : 'solid' }},
                        marker: {{ size: cond === 'control' ? 4 : 6 }},
                        legendgroup: grader
                    }});
                }});
            }});
            document.getElementById('grader-comparison-title').textContent = `Grader Scores: ${{model}} at strength ${{strength}}`;
            Plotly.react('grader-comparison-plot', traces, {{
                xaxis: {{ title: {{ text: 'Layer Position (%)', font: {{ ...plotlyFont, size: 14, color: plotlyMuted }}, standoff: 2 }}, range: [-2, 102], gridcolor: plotlyGrid, tickfont: {{ ...plotlyFont, size: 12, color: plotlyMuted }} }},
                yaxis: {{ title: {{ text: 'Mean Score', font: {{ ...plotlyFont, size: 14, color: plotlyMuted }} }}, range: [-0.03, 1.03], gridcolor: plotlyGrid, tickfont: {{ ...plotlyFont, size: 12, color: plotlyMuted }} }},
                height: 480,
                margin: {{ t: 16, b: 36, l: 56, r: 24 }},
                plot_bgcolor: plotlyBg, paper_bgcolor: plotlyBg,
                legend: {{ font: {{ ...plotlyFont, size: 12, color: plotlyMuted }}, orientation: 'v', x: 1.02, y: 1 }},
                hovermode: 'closest'
            }}, {{responsive: true}});
        }}
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
