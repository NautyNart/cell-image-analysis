from shiny import App, ui
from tab2 import tab2_ui, tab2_server

app_ui = ui.page_navbar(
    ui.head_content(
        ui.tags.style("""
            .progress { display: none; }
        """)
    ),
    tab2_ui,
    title=ui.div(
        ui.span("Breast Cancer Cell Classifier", 
                style="font-weight: bold; font-size: 18px;"),
        ui.span(" | DATA3888 Group Project", 
                style="font-size: 13px; color: #aaa; margin-left: 8px;"),
    ),
)

def server(input, output, session):
    tab2_server(input, output, session)

app = App(app_ui, server)