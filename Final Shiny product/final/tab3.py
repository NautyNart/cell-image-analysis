from shiny import ui, render
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import numpy as np
from PIL import Image

def draw_safe_heatmap(heat_data, img_path, title):
    if heat_data is None:
        heat_data = np.zeros((7, 7))

    if img_path is None:
        fig = go.Figure(go.Heatmap(
            z=heat_data,
            colorscale="magma",
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Attention<br>Weight",
                    side="top"
                ),
                thickness=12,
                len=0.82,
                y=0.5,
                tickvals=[0.04, 0.96],
                ticktext=["Low", "High"],
                ticks="",
                x=1.03
            ),
            zmin=0, zmax=1
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=13)) if title else None,
            height=390,
            width=390,
            xaxis_visible=False,
            yaxis_visible=False,
            margin=dict(l=8, r=55, t=10 if not title else 35, b=8)
        )
        return fig

    try:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_np = np.array(img)
        heat_scaled = np.repeat(np.repeat(heat_data, 32, axis=0), 32, axis=1)

        fig = go.Figure()
        fig.add_trace(go.Image(z=img_np))
        fig.add_trace(go.Heatmap(
            z=heat_scaled,
            colorscale="magma",
            opacity=0.5,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Attention<br>Weight",
                    side="top"
                ),
                thickness=12,
                len=0.82,
                y=0.5,
                tickvals=[0.04, 0.96],
                ticktext=["Low", "High"],
                ticks="",
                x=1.03
            ),
            zmin=0, zmax=1
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=13)) if title else None,
            height=390,
            width=390,
            xaxis_visible=False,
            yaxis_visible=False,
            margin=dict(l=8, r=55, t=10 if not title else 35, b=8)
        )
        return fig
    except:
        fig = go.Figure(go.Heatmap(
            z=heat_data,
            colorscale="magma",
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Attention<br>Weight",
                    side="top"
                ),
                thickness=12,
                len=0.82,
                y=0.5,
                tickvals=[0.04, 0.96],
                ticktext=["Low", "High"],
                ticks="",
                x=1.03
            ),
            zmin=0, zmax=1
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=13)) if title else None,
            height=390,
            width=390,
            xaxis_visible=False,
            yaxis_visible=False,
            margin=dict(l=8, r=55, t=10 if not title else 35, b=8)
        )
        return fig

tab3_ui = ui.nav_panel(
    "Model Attention",
    ui.div(
        ui.div(
            ui.h1(
                "Attention Map Comparison",
                style="font-size: 30px; font-weight: 750; margin-bottom: 8px;"
            ),

            ui.p(
                "Visual comparison of model attention under varying spatial contextual richness.",
                style="font-size: 16px; margin-bottom: 28px;", class_="condition-subtitle"
            ),

            ui.row(
                ui.column(
                    4,
                    ui.div(
                        ui.h4("100px Patch"),
                        ui.p("Full spatial context", class_="condition-subtitle"),
                        output_widget("hm1"),
                        class_="plot-card"
                    )
                ),
                ui.column(
                    4,
                    ui.div(
                        ui.h4("50px Patch"),
                        ui.p("Reduced spatial context", class_="condition-subtitle"),
                        output_widget("hm2"),
                        class_="plot-card"
                    )
                ),
                ui.column(
                    4,
                    ui.div(
                        ui.h4("Masked Patch"),
                        ui.p("Local morphology only", class_="condition-subtitle"),
                        output_widget("hm3"),
                        class_="plot-card"
                    )
                ),
            ),

            ui.div(
                ui.h3("Interpreting the Attention Maps"),
                ui.p("Attention maps provide a qualitative indication of which image regions contributed most strongly to model predictions under each spatial context condition."),
                ui.p("Comparing attention distributions across the 100px, 50px, and masked inputs allows assessment of whether the model relies primarily on local nuclear morphology or broader tissue-level architecture."),
                ui.p("Attention maps provide qualitative interpretability only and should not be treated as definitive mechanistic explanations of model decision-making."),
                class_="muted-panel"
            ),

            class_="content-wrap"
        ),
        class_="page-shell"
    )
)

def tab3_server(input, output, session):
    @output
    @render_widget
    def hm1():
        return draw_safe_heatmap(
            session.heat100(),
            session.upload_100px_path,
            ""
        )

    @output
    @render_widget
    def hm2():
        return draw_safe_heatmap(
            session.heat50(),
            session.upload_50px_path,
            ""
        )

    @output
    @render_widget
    def hm3():
        return draw_safe_heatmap(
            session.heatmask(),
            session.upload_masked_path,
            ""
        )
