from shiny import App, ui

from tab1 import tab1_ui, tab1_server
from tab2 import tab2_ui, tab2_server
from tab3 import tab3_ui, tab3_server

app_ui = ui.page_navbar(
    ui.head_content(
        ui.tags.style("""
          :root {
              --bg: #f6f2fb;
              --card: #ffffff;
              --text: #1f2933;
              --muted: #5f6b76;
              --border: #ded4ea;
              --accent: #6f4aa8;
              --accent-soft: #eee7f7;
              --shadow: 0 4px 14px rgba(58, 39, 86, 0.08);
          }
      
          body {
              background: var(--bg);
              color: var(--text);
              font-size: 15px;
          }
      
          .navbar {
              background: #ffffff;
              border-bottom: 1px solid var(--border);
          }
      
          .navbar-brand {
              font-size: 20px;
              font-weight: 600;
          }
      
          .nav-link {
              font-size: 15px;
          }
      
          .nav-link.active {
              border-bottom: 3px solid var(--accent);
              font-weight: 650;
          }
      
          .page-shell {
              background: var(--bg);
              min-height: 100vh;
              padding: 32px 24px 56px 24px;
          }
      
          .content-wrap {
              max-width: 1320px;
              margin: 0 auto;
          }
      
          .hero {
              text-align: center;
              padding: 34px 20px 28px 20px;
              margin-bottom: 24px;
          }
      
          .hero h1 {
              font-size: 32px;
              font-weight: 750;
              margin-bottom: 8px;
              color: var(--text);
          }
      
          .hero p {
              font-size: 16px;
              color: var(--muted);
              margin-bottom: 0;
          }
      
          h1 {
              font-size: 30px;
              font-weight: 750;
              color: var(--text);
          }
      
          h3 {
              font-size: 24px;
              font-weight: 650;
              color: var(--text);
          }
      
          h4 {
              font-size: 22px;
              font-weight: 650;
              color: var(--text);
          }
      
          p, li {
              font-size: 15px;
              line-height: 1.55;
              color: var(--text);
          }
      
          .research-box {
              background: var(--accent-soft);
              border-left: 5px solid var(--accent);
              padding: 24px 32px;
              margin-bottom: 32px;
              border-radius: 10px;
          }
      
          .research-card, .section-card, .plot-card {
              background: var(--card);
              border: 1px solid var(--border);
              border-radius: 14px;
              box-shadow: var(--shadow);
          }
      
          .research-card {
              padding: 24px;
              height: 100%;
          }
      
          .section-card {
              padding: 26px;
              margin-top: 28px;
          }
      
          .plot-card {
              padding: 18px;
              height: 100%;
          }
      
          .muted-panel {
              background: var(--accent-soft);
              border-left: 5px solid var(--accent);
              border-radius: 12px;
              padding: 24px;
              margin-top: 28px;
          }
      
          .condition-subtitle {
              color: var(--muted);
              font-size: 15px;
              margin-top: -6px;
              margin-bottom: 16px;
          }
      
          .small-muted {
              color: var(--muted);
              font-size: 13px;
          }
      
          .cell-red {
              color: #c73545;
              font-weight: 700;
          }
      
          .cell-blue {
              color: #4f7fa4;
              font-weight: 700;
          }
      
          .cell-orange {
              color: #d98845;
              font-weight: 700;
          }
          
          .cell-ref-title {
              font-size: 16px;
              font-weight: 700;
              margin-bottom: 6px;
              color: var(--text);
          }
          
          .cell-ref-text {
              font-size: 14px;
              line-height: 1.45;
              color: var(--muted);
              margin-bottom: 0;
          }
          
          .upload-panel {
              margin-bottom: 28px;
          }
          
          .upload-box {
              background: #ffffff;
              border: 1.5px solid #b99bd6;
              border-radius: 12px;
              padding: 16px;
              height: 100%;
              box-shadow: 0 3px 10px rgba(92, 64, 128, 0.07);
          }
          
          .upload-box::before {
              content: "";
              display: block;
              height: 4px;
              width: 44px;
              background: var(--accent);
              border-radius: 999px;
              margin-bottom: 14px;
          }
          
          .upload-box .form-control {
              font-size: 14px;
          }
          
          .upload-label-title {
              font-size: 15px;
              font-weight: 700;
              margin-bottom: 2px;
              color: var(--text);
          }
          
          .upload-label-subtitle {
              font-size: 13px;
              color: var(--muted);
              margin-bottom: 10px;
          }
          
          .plot-card {
              background: var(--card);
              border: 1px solid var(--border);
              border-radius: 14px;
              box-shadow: var(--shadow);
              padding: 18px;
              height: auto;
              min-height: 0;
              display: flex;
              flex-direction: column;
          }
          
          .plot-card img {
              display: block;
              width: 220px;
              height: 220px;
              object-fit: cover;
              margin: 12px auto 18px auto;
          }
          
          .shiny-image-output {
              height: auto !important;
              min-height: 0 !important;
              margin-bottom: 8px !important;
          }
          
          .prediction-label {
              font-size: 18px;
              font-weight: 700;
              margin-top: 6px;
              margin-bottom: 4px;
          }
          
          .prediction-desc {
              font-size: 13px;
              line-height: 1.4;
              color: var(--muted);
              min-height: 38px;
              margin-bottom: 8px;
          }
          
          .js-plotly-plot {
              margin-top: 0;
          }
          
          .progress {
              display: none;
          }
          
      """),
    ),
    tab1_ui,
    tab2_ui,
    tab3_ui,
    title="Breast Cancer Tumour Subtype Analysis"
)

def server(input, output, session):
    tab1_server(input, output, session)
    tab2_server(input, output, session)
    tab3_server(input, output, session)

app = App(app_ui, server)
