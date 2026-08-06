# %% [markdown]
# # arc3-agi Experiment Analysis
#
# Explore and compare population evolution experiments stored in PostgreSQL.
#
# **Quick start:**
# 1. Set `DATABASE_URL` if you are not using the local development database.
# 2. Run cells in order from the top.
# 3. Change `EXPERIMENT_ID` / `EXPERIMENT_IDS` in the relevant cells to select what to plot.
# 4. All charts are interactive - zoom, pan, hover for exact values.

import importlib

# %%
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots

import arc3_agi.experiment as experiment_mod

# Reload during notebook execution so long-lived kernels pick up code changes.
experiment_mod = importlib.reload(experiment_mod)
ExperimentStore = experiment_mod.ExperimentStore
resolve_database_url = experiment_mod.resolve_database_url

EXPERIMENT_ID = None  # <-- change me
EXPERIMENT_IDS = []  # <-- change me
DATABASE_URL = os.environ.get("DATABASE_URL")
store = ExperimentStore(DATABASE_URL)

# %% [markdown]
# ## All experiments

# %%
experiments = store.list_experiments()
display(
    experiments[
        ["id", "name", "description", "run_id", "created_at", "pop_count", "gen_count"]
    ][-20:]
)

if EXPERIMENT_ID is None:
    EXPERIMENT_ID = experiments["id"].iloc[-1]  # <-- change me
if not EXPERIMENT_IDS:
    EXPERIMENT_IDS = experiments["id"].iloc[-4:].tolist()  # <-- change me

# %% [markdown]
# ## Single-experiment analysis
#
# Set `EXPERIMENT_ID` to the `id` from the table above.

# %%
df = store.load_stats(EXPERIMENT_ID)
exp_name = experiments.loc[experiments["id"] == EXPERIMENT_ID, "name"].iat[0]  # type: ignore

print(
    f"Experiment '{exp_name}'  |  {df['pop_id'].nunique()} populations  |  {df['generation'].max()} generations"
)
df.head()

# %% [markdown]
# ### Fan plot — mean fitness per population per generation
#
# Each faint line is one population; the bold line is the mean across all populations.

# %%
fig = go.Figure()

for pop_id, grp in df.groupby("pop_id"):
    fig.add_trace(
        go.Scatter(
            x=grp["generation"],
            y=grp["mean_fitness"],
            mode="lines",
            line=dict(color="steelblue", width=0.4),
            opacity=0.35,
            showlegend=False,
            hovertemplate=f"pop {pop_id}<br>gen %{{x}}<br>mean %{{y:.3f}}<extra></extra>",
        )
    )

mean_of_means = df.groupby("generation")["mean_fitness"].mean().reset_index()
fig.add_trace(
    go.Scatter(
        x=mean_of_means["generation"],
        y=mean_of_means["mean_fitness"],
        mode="lines",
        line=dict(color="darkblue", width=2.5),
        name="mean of means",
    )
)

fig.update_layout(
    title=f"[{exp_name}] Mean fitness — all populations",
    xaxis_title="Generation",
    yaxis_title="Mean fitness",
    template="plotly_white",
    height=500,
)
fig.show()

# %% [markdown]
# ### Max fitness fan plot

# %%
fig = go.Figure()

for pop_id, grp in df.groupby("pop_id"):
    fig.add_trace(
        go.Scatter(
            x=grp["generation"],
            y=grp["max_fitness"],
            mode="lines",
            line=dict(color="tomato", width=0.4),
            opacity=0.35,
            showlegend=False,
            hovertemplate=f"pop {pop_id}<br>gen %{{x}}<br>max %{{y:.3f}}<extra></extra>",
        )
    )

mean_of_max = df.groupby("generation")["max_fitness"].mean().reset_index()
fig.add_trace(
    go.Scatter(
        x=mean_of_max["generation"],
        y=mean_of_max["max_fitness"],
        mode="lines",
        line=dict(color="darkred", width=2.5),
        name="mean of max",
    )
)

fig.update_layout(
    title=f"[{exp_name}] Max fitness — all populations",
    xaxis_title="Generation",
    yaxis_title="Max fitness",
    template="plotly_white",
    height=500,
)
fig.show()

# %% [markdown]
# ### Aggregate ribbon — mean ± 1 std across all populations

# %%
agg = (
    df.groupby("generation")
    .agg(
        mean_mean=("mean_fitness", "mean"),
        std_mean=("mean_fitness", "std"),
        mean_max=("max_fitness", "mean"),
        std_max=("max_fitness", "std"),
    )
    .reset_index()
)

fig = go.Figure()

for label, col, colour in [
    ("Mean fitness", "mean_mean", "steelblue"),
    ("Max fitness", "mean_max", "tomato"),
]:
    std_col = col.replace("mean_", "std_")
    upper = agg[col] + agg[std_col]
    lower = agg[col] - agg[std_col]
    fig.add_trace(
        go.Scatter(
            x=pd.concat([agg["generation"], agg["generation"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor=colour,
            opacity=0.15,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg["generation"],
            y=agg[col],
            mode="lines",
            line=dict(color=colour, width=2),
            name=label,
        )
    )

fig.update_layout(
    title=f"[{exp_name}] Fitness ribbon (mean ± 1 std across populations)",
    xaxis_title="Generation",
    yaxis_title="Fitness",
    template="plotly_white",
    height=500,
)
fig.show()

# %% [markdown]
# ### Generation wall-clock time

# %%
time_agg = (
    df.groupby("generation")
    .agg(
        mean_dur=("duration_s", "mean"),
        max_dur=("duration_s", "max"),
    )
    .reset_index()
)


fig = px.line(
    time_agg,
    x="generation",
    y=["mean_dur", "max_dur"],
    labels={"value": "Duration (s)", "generation": "Generation", "variable": "Metric"},
    title=f"[{exp_name}] Generation wall-clock time",
    template="plotly_white",
    render_mode="svg",
)

fig.show()

# %% [markdown]
# ### Checkpoint diagnostics — clause/state visibility

# %%
if not hasattr(store, "load_checkpoint_diagnostics"):
    # Recreate from the reloaded class when the store object came from a stale import.
    store = ExperimentStore(DATABASE_URL)

diag_df = store.load_checkpoint_diagnostics(EXPERIMENT_ID)
if diag_df.empty:
    print(
        "No checkpoint diagnostics found for this experiment. "
        "Run diagnostics ingestion or ensure checkpoints exist under runs/<run_id>/pop_*/gen_*."
    )
else:
    print(
        f"Diagnostics rows: {len(diag_df)} "
        f"| populations: {diag_df['pop_id'].nunique()} "
        f"| generations: {diag_df['generation'].nunique()}"
    )
    display(diag_df.head())

# %% [markdown]
# #### Clause ambivalence vs opinionatedness

# %%
if not diag_df.empty:
    amb = (
        diag_df.groupby("generation")
        .agg(
            zero_literal_clause_rate=("zero_literal_clause_rate", "mean"),
            clause_density_mean=("clause_density_mean", "mean"),
            polarity_abs_mean=("polarity_abs_mean", "mean"),
        )
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=amb["generation"],
            y=amb["zero_literal_clause_rate"],
            mode="lines+markers",
            name="zero-literal clause rate",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=amb["generation"],
            y=amb["clause_density_mean"],
            mode="lines+markers",
            name="clause density mean",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=amb["generation"],
            y=amb["polarity_abs_mean"],
            mode="lines+markers",
            name="polarity |pos-neg|/active",
        )
    )
    fig.update_layout(
        title=f"[{exp_name}] Clause ambivalence/opinionatedness",
        xaxis_title="Generation",
        yaxis_title="Rate",
        template="plotly_white",
        height=500,
    )
    fig.show()

# %% [markdown]
# #### State-bit entropy and row density

# %%
if not diag_df.empty:
    state_diag = (
        diag_df.groupby("generation")
        .agg(
            literal_state_entropy_mean=("literal_state_entropy_mean", "mean"),
            state_row_density_mean=("state_row_density_mean", "mean"),
            response_row_density_mean=("response_row_density_mean", "mean"),
            state_response_density_gap=("state_response_density_gap", "mean"),
        )
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=state_diag["generation"],
            y=state_diag["literal_state_entropy_mean"],
            mode="lines+markers",
            name="literal-state entropy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=state_diag["generation"],
            y=state_diag["state_row_density_mean"],
            mode="lines+markers",
            name="state row density",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=state_diag["generation"],
            y=state_diag["response_row_density_mean"],
            mode="lines+markers",
            name="response row density",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=state_diag["generation"],
            y=state_diag["state_response_density_gap"],
            mode="lines+markers",
            name="state-response density gap",
        )
    )
    fig.update_layout(
        title=f"[{exp_name}] State-bit entropy and density",
        xaxis_title="Generation",
        yaxis_title="Metric value",
        template="plotly_white",
        height=500,
    )
    fig.show()

# %% [markdown]
# #### Clause churn between checkpoints

# %%
if not diag_df.empty and diag_df["clause_churn_rate"].notna().any():
    churn_df = (
        diag_df.dropna(subset=["clause_churn_rate"])
        .groupby("generation")["clause_churn_rate"]
        .mean()
        .reset_index()
    )
    fig = px.line(
        churn_df,
        x="generation",
        y="clause_churn_rate",
        markers=True,
        title=f"[{exp_name}] Clause churn rate between checkpoints",
        labels={"clause_churn_rate": "Churn rate", "generation": "Generation"},
        template="plotly_white",
        render_mode="svg",
    )
    fig.show()

# %% [markdown]
# #### Diagnostic interpretation

# %%
if not diag_df.empty:
    trend_df = (
        diag_df.groupby("generation")
        .agg(
            zero_literal_clause_rate=("zero_literal_clause_rate", "mean"),
            clause_density_mean=("clause_density_mean", "mean"),
            polarity_abs_mean=("polarity_abs_mean", "mean"),
            literal_state_entropy_mean=("literal_state_entropy_mean", "mean"),
            state_row_density_mean=("state_row_density_mean", "mean"),
            response_row_density_mean=("response_row_density_mean", "mean"),
            state_response_density_gap=("state_response_density_gap", "mean"),
            clause_churn_rate=("clause_churn_rate", "mean"),
        )
        .reset_index()
    )

    first = trend_df.iloc[0]
    last = trend_df.iloc[-1]

    delta_zero = float(
        last["zero_literal_clause_rate"] - first["zero_literal_clause_rate"]
    )
    delta_density = float(last["clause_density_mean"] - first["clause_density_mean"])
    delta_polarity = float(last["polarity_abs_mean"] - first["polarity_abs_mean"])
    delta_entropy = float(
        last["literal_state_entropy_mean"] - first["literal_state_entropy_mean"]
    )
    final_gap = float(last["state_response_density_gap"])

    churn_series = trend_df["clause_churn_rate"].dropna()
    churn_mean = float(churn_series.mean()) if not churn_series.empty else float("nan")

    ambivalence_text = (
        "decreasing"
        if delta_zero < -1e-6
        else "increasing" if delta_zero > 1e-6 else "flat"
    )
    specificity_text = (
        "increasing"
        if delta_density > 1e-6
        else "decreasing" if delta_density < -1e-6 else "flat"
    )
    polarity_text = (
        "increasing"
        if delta_polarity > 1e-6
        else "decreasing" if delta_polarity < -1e-6 else "flat"
    )
    entropy_text = (
        "increasing"
        if delta_entropy > 1e-6
        else "decreasing" if delta_entropy < -1e-6 else "flat"
    )
    state_bias_text = (
        "response-biased"
        if final_gap < 0
        else "state-biased" if final_gap > 0 else "balanced"
    )

    print("Interpretation summary")
    print(
        f"- Clausal ambivalence (zero-literal rate): {ambivalence_text} (delta {delta_zero:+.4f})."
    )
    print(
        f"- Clause specificity (density): {specificity_text} (delta {delta_density:+.4f})."
    )
    print(
        f"- Clause opinionatedness (polarity): {polarity_text} (delta {delta_polarity:+.4f})."
    )
    print(f"- Literal-state entropy: {entropy_text} (delta {delta_entropy:+.4f}).")
    print(
        f"- State vs response emphasis: {state_bias_text} (final gap {final_gap:+.4f})."
    )
    if churn_series.empty:
        print(
            "- Clause churn: unavailable (need at least two checkpoint generations per population)."
        )
    else:
        print(
            f"- Clause churn: mean {churn_mean:.4f} across comparable checkpoint steps."
        )

    print("\nGuidance")
    if final_gap < -0.005:
        print(
            "- The model is allocating more clause detail to response bits than state bits."
        )
    elif final_gap > 0.005:
        print(
            "- The model is allocating more clause detail to state bits than response bits."
        )
    else:
        print("- State and response rows are similarly detailed.")

    if delta_zero < 0 and delta_polarity > 0:
        print(
            "- Clauses are becoming less ambivalent and more opinionated over training."
        )
    elif delta_zero > 0 and delta_polarity < 0:
        print(
            "- Clauses are drifting toward more ambivalence and less directional preference."
        )
    else:
        print(
            "- Clause ambivalence/opinion trends are mixed; inspect per-pop traces for heterogeneity."
        )

# %% [markdown]
# ---
# ## Multi-experiment comparison
#
# Set `EXPERIMENT_IDS` to a list of ids to compare on the same chart.

# %%

exp_dfs = {}
exp_names = {}
for eid in EXPERIMENT_IDS:
    exp_dfs[eid] = store.load_stats(eid)
    row = experiments.loc[experiments["id"] == eid]
    exp_names[eid] = row["name"].iat[0] if not row.empty else str(eid)

print("Loaded:", {eid: exp_names[eid] for eid in EXPERIMENT_IDS})

# %% [markdown]
# ### Overlay — mean-of-means fitness per experiment

# %%
fig = go.Figure()

colours = px.colors.qualitative.Plotly

for i, eid in enumerate(EXPERIMENT_IDS):
    df_e = exp_dfs[eid]
    agg_e = (
        df_e.groupby("generation")
        .agg(
            mean_mean=("mean_fitness", "mean"),
            std_mean=("mean_fitness", "std"),
        )
        .reset_index()
    )
    colour = colours[i % len(colours)]

    upper = agg_e["mean_mean"] + agg_e["std_mean"]
    lower = agg_e["mean_mean"] - agg_e["std_mean"]
    fig.add_trace(
        go.Scatter(
            x=pd.concat([agg_e["generation"], agg_e["generation"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor=colour,
            opacity=0.12,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg_e["generation"],
            y=agg_e["mean_mean"],
            mode="lines",
            line=dict(color=colour, width=2),
            name=exp_names[eid],
        )
    )

fig.update_layout(
    title="Mean-of-means fitness comparison (ribbon = ±1 std across populations)",
    xaxis_title="Generation",
    yaxis_title="Mean fitness",
    template="plotly_white",
    height=500,
)
fig.show()

# %% [markdown]
# ### Overlay — average generation duration per experiment

# %%
fig = go.Figure()

for i, eid in enumerate(EXPERIMENT_IDS):
    df_e = exp_dfs[eid]
    average_duration = df_e.groupby("generation")["duration_s"].mean().reset_index()
    colour = colours[i % len(colours)]

    fig.add_trace(
        go.Scatter(
            x=average_duration["generation"],
            y=average_duration["duration_s"],
            mode="lines",
            line=dict(color=colour, width=2),
            name=exp_names[eid],
        )
    )

fig.update_layout(
    title="Average generation duration comparison",
    xaxis_title="Generation",
    yaxis_title="Average duration (s)",
    template="plotly_white",
    height=500,
)
fig.show()

# %% [markdown]
# ### Side-by-side fan plots

# %%
n = len(EXPERIMENT_IDS)
fig = make_subplots(
    rows=1,
    cols=n,
    subplot_titles=[exp_names[eid] for eid in EXPERIMENT_IDS],
    shared_yaxes=True,
)

colours = px.colors.qualitative.Plotly

for col_idx, eid in enumerate(EXPERIMENT_IDS, start=1):
    df_e = exp_dfs[eid]
    colour = colours[(col_idx - 1) % len(colours)]

    for pop_id, grp in df_e.groupby("pop_id"):
        fig.add_trace(
            go.Scatter(
                x=grp["generation"],
                y=grp["mean_fitness"],
                mode="lines",
                line=dict(color=colour, width=0.4),
                opacity=0.3,
                showlegend=False,
                hovertemplate=f"pop {pop_id}<br>gen %{{x}}<br>mean %{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

    mean_of_means = df_e.groupby("generation")["mean_fitness"].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=mean_of_means["generation"],
            y=mean_of_means["mean_fitness"],
            mode="lines",
            line=dict(color=colour, width=2.5),
            name=exp_names[eid],
        ),
        row=1,
        col=col_idx,
    )

fig.update_layout(
    title="Side-by-side fan plots — mean fitness per population",
    template="plotly_white",
    height=500,
)
fig.update_xaxes(title_text="Generation")
fig.update_yaxes(title_text="Mean fitness", col=1)
fig.show()

# %% [markdown]
# ### Overlay — max fitness comparison

# %%
fig = go.Figure()

for i, eid in enumerate(EXPERIMENT_IDS):
    df_e = exp_dfs[eid]
    agg_e = (
        df_e.groupby("generation")
        .agg(
            mean_max=("max_fitness", "mean"),
            std_max=("max_fitness", "std"),
        )
        .reset_index()
    )
    colour = colours[i % len(colours)]

    upper = agg_e["mean_max"] + agg_e["std_max"]
    lower = agg_e["mean_max"] - agg_e["std_max"]
    fig.add_trace(
        go.Scatter(
            x=pd.concat([agg_e["generation"], agg_e["generation"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor=colour,
            opacity=0.12,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg_e["generation"],
            y=agg_e["mean_max"],
            mode="lines",
            line=dict(color=colour, width=2),
            name=exp_names[eid],
        )
    )

fig.update_layout(
    title="Max fitness comparison (ribbon = ±1 std across populations)",
    xaxis_title="Generation",
    yaxis_title="Max fitness",
    template="plotly_white",
    height=500,
)
fig.show()

fig = go.Figure()

for i, eid in enumerate(EXPERIMENT_IDS):
    df_e = exp_dfs[eid]
    best_of_best = df_e.groupby("generation")["max_fitness"].max().reset_index()
    colour = colours[i % len(colours)]

    fig.add_trace(
        go.Scatter(
            x=best_of_best["generation"],
            y=best_of_best["max_fitness"],
            mode="lines",
            line=dict(color=colour, width=2.5),
            name=exp_names[eid],
        )
    )

fig.update_layout(
    title="Best-of-best fitness comparison (max of max_fitness per generation)",
    xaxis_title="Generation",
    yaxis_title="Best fitness",
    template="plotly_white",
    height=500,
)
fig.show()


# %% [markdown]
# ### Best of the best — single peak fitness per experiment
#
# This chart shows the single highest fitness value observed anywhere in each experiment.

# %%
best_of_best = []
empty_experiment_ids = []

for eid in EXPERIMENT_IDS:
    df_e = exp_dfs[eid]
    if df_e.empty:
        empty_experiment_ids.append(eid)
        continue
    best_row = df_e.loc[df_e["max_fitness"].idxmax()]
    best_of_best.append(
        {
            "experiment_id": eid,
            "experiment_name": exp_names[eid],
            "best_fitness": best_row["max_fitness"],
            "generation": best_row["generation"],
            "pop_id": best_row["pop_id"],
        }
    )

if empty_experiment_ids:
    print(f"Skipped experiments with no generation stats: {empty_experiment_ids}")

best_of_best_df = pd.DataFrame(
    best_of_best,
    columns=[
        "experiment_id",
        "experiment_name",
        "best_fitness",
        "generation",
        "pop_id",
    ],
)

fig = px.bar(
    best_of_best_df,
    x="experiment_name",
    y="best_fitness",
    text="best_fitness",
    labels={
        "experiment_name": "Experiment",
        "best_fitness": "Best fitness",
    },
    title="Best of the best fitness comparison (single peak per experiment)",
    template="plotly_white",
)

fig.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
    hovertemplate=("%{x}<br>" "Best fitness: %{y:.3f}<br>" "<extra></extra>"),
)

fig.update_layout(
    xaxis_tickangle=-25,
    height=500,
)

fig.show()

# %%
