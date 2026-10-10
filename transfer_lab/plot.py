"""Offline interactive held-out loss and transition-cost comparison."""
import argparse
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

parser = argparse.ArgumentParser()
parser.add_argument("--run",default="results/confidence-transfer-v1")
out = Path(parser.parse_args().run)
r = json.loads((out/"results.json").read_text())
cases = ["reference_mixed",*r["config"]["ood_cases"]]
fig = make_subplots(rows=1,cols=2,subplot_titles=("Overall log loss", "Loss after entry into learnable mode (first16)"))
colors = {"recency":"#ca972c","fixed_shrinkage":"#8796a8","frozen_gate":"#238d98","online_mixture":"#8b67bc"}
for family,color in colors.items():
    fig.add_trace(go.Bar(name=family,x=cases,y=[r["summary"][c][family]["log_loss"] for c in cases],marker_color=color),row=1,col=1)
    fig.add_trace(go.Bar(name=family,showlegend=False,x=cases,
        y=[r["summary"][c][family]["slices"]["learnable_entry_first16"]["metrics"]["log_loss"] for c in cases],marker_color=color),row=1,col=2)
for col in (1,2):
    fig.update_yaxes(title_text="Nats (lower is better)",row=1,col=col)
    fig.update_xaxes(tickangle=-20,row=1,col=col)
fig.update_layout(title="Frozen confidence transfer — recommended local default: "+r["gates"]["recommended_confidence_default"],
    template="plotly_white",barmode="group",height=620,width=1300,
    margin=dict(t=90,b=160),legend=dict(orientation="h",y=-.35))
fig.write_html(out/"comparison.html",include_plotlyjs=True,full_html=True,auto_open=False)
print(out/"comparison.html")
