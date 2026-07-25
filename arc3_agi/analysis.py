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

# %%
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots

from arc3_agi.experiment import ExperimentStore, resolve_database_url

DATABASE_URL = os.environ.get("DATABASE_URL")
store = ExperimentStore(DATABASE_URL)

# %% [markdown]
# ## All experiments

# %%
experiments = store.list_experiments()
display(
    experiments[
        ["id", "name", "description", "run_id", "created_at", "pop_count", "gen_count"]
    ]
)

# %% [markdown]
# ## Single-experiment analysis
#
# Set `EXPERIMENT_ID` to the `id` from the table above.

# %%
EXPERIMENT_ID = 42  # <-- change me

df = store.load_stats(EXPERIMENT_ID)
exp_name = experiments.loc[experiments["id"] == EXPERIMENT_ID, "name"].iat[0]

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
# ---
# ## Multi-experiment comparison
#
# Set `EXPERIMENT_IDS` to a list of ids to compare on the same chart.

# %%
EXPERIMENT_IDS = [1, 2, 42, 44, 45]  # <-- change me

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

for eid in EXPERIMENT_IDS:
    df_e = exp_dfs[eid]
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

best_of_best_df = pd.DataFrame(best_of_best)

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
