"""
Project Workflow Diagram Generator
---------------------------------------------------------
Visualizes the ML project workflow as a sequence of Epics,
each containing its own User Stories.

Requires: graphviz (pip install graphviz) and Graphviz system binary
           (https://graphviz.org/download/)

Run:
    python project_workflow_diagram.py
Output:
    project_workflow_diagram.png
"""

from graphviz import Digraph
from html import escape as _esc


def build_workflow_diagram():
    dot = Digraph("ProjectWorkflow", format="png")
    dot.attr(rankdir="TB", splines="line", bgcolor="white", nodesep="0.6", ranksep="0.8")
    dot.attr("node", fontname="Helvetica", shape="plain")
    dot.attr("edge", fontname="Helvetica", fontsize="11", color="#5F5E5A", penwidth="1.5")

    # ---------- Epic data ----------
    epics = [
        {
            "id": "epic1",
            "title": "Epic 1: Data Collection & Architecture Design",
            "header_color": "#185FA5",
            "fill_color": "#E6F1FB",
            "stories": [
                "Download dataset & store in project directory",
                "Define application architecture (data flow, ML workflow, deployment)"
            ]
        },
        {
            "id": "epic2",
            "title": "Epic 2: Visualizing & Analyzing the Data",
            "header_color": "#0F6E56",
            "fill_color": "#E1F5EE",
            "stories": [
                "Import and read dataset using Pandas",
                "Perform univariate analysis",
                "Conduct bivariate analysis",
                "Perform multivariate analysis"
            ]
        },
        {
            "id": "epic3",
            "title": "Epic 3: Data Pre-Processing",
            "header_color": "#993C1D",
            "fill_color": "#FAECE7",
            "stories": [
                "Handle missing values, duplicates & inconsistencies",
                "Balance dataset for fair class representation",
                "Scale & normalize numerical features",
                "Split dataset into train and test sets"
            ]
        },
        {
            "id": "epic4",
            "title": "Epic 4: Model Building",
            "header_color": "#854F0B",
            "fill_color": "#FAEEDA",
            "stories": [
                "Train & evaluate Decision Tree model",
                "Build & test Random Forest model",
                "Implement KNN model & analyze accuracy",
                "Train XGBoost model & compare performance",
                "Evaluate all models & save best-performing one"
            ]
        },
        {
            "id": "epic5",
            "title": "Epic 5: Application Building",
            "header_color": "#993556",
            "fill_color": "#FBEAF0",
            "stories": [
                "Design & develop HTML pages for UI",
                "Build Flask app & integrate trained ML model",
                "Run, test & validate the application"
            ]
        },
    ]

    # ---------- Build each epic as an HTML-table node ----------
    for epic in epics:
        story_rows = "".join(
            f'<TR><TD ALIGN="LEFT" BGCOLOR="white" CELLPADDING="8">'
            f'<FONT POINT-SIZE="11">Story {i + 1}: {_esc(story)}</FONT></TD></TR>'
            for i, story in enumerate(epic["stories"])
        )

        label = f'''
        <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0"
               BGCOLOR="{epic['fill_color']}" COLOR="{epic['header_color']}">
            <TR><TD BGCOLOR="{epic['header_color']}" CELLPADDING="10">
                <FONT COLOR="white" POINT-SIZE="13"><B>{_esc(epic['title'])}</B></FONT>
            </TD></TR>
            {story_rows}
        </TABLE>>'''

        dot.node(epic["id"], label=label)

    # ---------- Sequential flow between epics ----------
    for i in range(len(epics) - 1):
        dot.edge(epics[i]["id"], epics[i + 1]["id"], label="next", fontcolor="#5F5E5A")

    return dot


if __name__ == "__main__":
    diagram = build_workflow_diagram()
    output_path = diagram.render("project_workflow_diagram", cleanup=True)
    print(f"Workflow diagram saved to: {output_path}")