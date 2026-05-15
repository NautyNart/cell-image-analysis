from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH_100px  = '/Users/apple/Desktop/Shiny App ViT/vit_best_100px.pth'
MODEL_PATH_50px   = '/Users/apple/Desktop/Shiny App ViT/vit_best_50px.pth'
MODEL_PATH_masked = '/Users/apple/Desktop/Shiny App ViT/vit_best_masked.pth'

NUM_CLASSES  = 3
CLASS_NAMES  = ['Tumor', 'DCIS_1', 'DCIS_2']
CLASS_DESC   = {
    'Tumor':  'Invasive tumour cells that have broken through the ductal boundary and infiltrated the surrounding tissue stroma.',
    'DCIS_1': 'Ductal Carcinoma In Situ subtype 1 — cancer cells confined within the milk ducts, with uniform nuclear arrangement and mild variability.',
    'DCIS_2': 'Ductal Carcinoma In Situ subtype 2 — a more aggressive in situ subtype showing denser cell packing and higher nuclear variability.',
}
CLASS_COLORS = {'Tumor': '#E63946', 'DCIS_1': '#457B9D', 'DCIS_2': '#F4A261'}
MEAN = [0.7236391989344737, 0.6037140915846041, 0.8286546417788325]
STD  = [0.1448797646380704, 0.1477115746885807, 0.0756316864656238]
IMG_SIZE = 224

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

MASK_KEEP_RATIO = 0.30
MASK_FILL       = np.array([123, 116, 103], dtype=np.uint8)


# ── ViT model ────────────────────────────────────────────────────────────────
def build_vit(weights_path):
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

MODEL_100px  = build_vit(MODEL_PATH_100px)
MODEL_50px   = build_vit(MODEL_PATH_50px)
MODEL_masked = build_vit(MODEL_PATH_masked)


# ── Transform ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ── Mask function ─────────────────────────────────────────────────────────────
def apply_mask(img):
    arr    = np.array(img)
    h, w   = arr.shape[:2]
    cx, cy = w // 2, h // 2
    radius = int(min(h, w) * MASK_KEEP_RATIO)
    Y, X   = np.ogrid[:h, :w]
    inside = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
    result = np.full_like(arr, fill_value=MASK_FILL)
    result[inside] = arr[inside]
    return Image.fromarray(result)


# ── Prediction ───────────────────────────────────────────────────────────────
def predict(img, model):
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1).squeeze().numpy()
    pred_label = CLASS_NAMES[int(np.argmax(probs))]
    return pred_label, probs


# ── UI ───────────────────────────────────────────────────────────────────────
tab2_ui = ui.nav_panel(
    "Cell Type Prediction",
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Upload Cell Images"),
            ui.p("Upload one patch for each condition to compare how spatial context affects prediction:",
                 style="font-size: 12px; color: #555;"),
            ui.br(),
            ui.input_file("upload_100px", "100px patch (full context)",
                          accept=[".png", ".jpg", ".jpeg"], multiple=False),
            ui.input_file("upload_50px", "50px patch (reduced context)",
                          accept=[".png", ".jpg", ".jpeg"], multiple=False),
            ui.input_file("upload_masked", "Masked patch (30% context)",
                          accept=[".png", ".jpg", ".jpeg"], multiple=False),
            ui.hr(),
            ui.p("The model will classify each patch into one of three tumour subtypes: "
                 "Tumor, DCIS_1, or DCIS_2.",
                 style="color: grey; font-size: 12px;"),
            ui.div(style="padding-bottom: 20px;"),
            width=350,
        ),
        ui.row(
            ui.column(
                4,
                ui.div(
                    ui.h4("100px (Full Context)"),
                    ui.output_image("show_image_100px"),
                    ui.output_ui("prediction_label_100px"),
                    ui.output_ui("prediction_desc_100px"),
                    output_widget("prob_chart_100px"),
                    style="background: white; border-radius: 8px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);"
                )
            ),
            ui.column(
                4,
                ui.div(
                    ui.h4("50px (Reduced Context)"),
                    ui.output_image("show_image_50px"),
                    ui.output_ui("prediction_label_50px"),
                    ui.output_ui("prediction_desc_50px"),
                    output_widget("prob_chart_50px"),
                    style="background: white; border-radius: 8px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);"
                )
            ),
            ui.column(
                4,
                ui.div(
                    ui.h4("Masked (30% Context)"),
                    ui.output_image("show_image_masked"),
                    ui.output_ui("prediction_label_masked"),
                    ui.output_ui("prediction_desc_masked"),
                    output_widget("prob_chart_masked"),
                    style="background: white; border-radius: 8px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);"
                )
            ),
        ),
    ),
)


# ── Server ───────────────────────────────────────────────────────────────────
def tab2_server(input, output, session):

    @reactive.calc
    def image_100px():
        f = input.upload_100px()
        if not f:
            return None, None
        return Image.open(f[0]['datapath']).convert('RGB'), f[0]['datapath']

    @reactive.calc
    def image_50px():
        f = input.upload_50px()
        if not f:
            return None, None
        return Image.open(f[0]['datapath']).convert('RGB'), f[0]['datapath']

    @reactive.calc
    def image_masked():
        f = input.upload_masked()
        if not f:
            return None, None
        return Image.open(f[0]['datapath']).convert('RGB'), f[0]['datapath']

    @output
    @render.image
    def show_image_100px():
        img, path = image_100px()
        if img is None:
            return None
        return {"src": path, "width": "220px", "height": "220px"}

    @output
    @render.image
    def show_image_50px():
        img, path = image_50px()
        if img is None:
            return None
        return {"src": path, "width": "220px", "height": "220px"}

    @output
    @render.image
    def show_image_masked():
        img, path = image_masked()
        if img is None:
            return None
        return {"src": path, "width": "220px", "height": "220px"}

    @output
    @render.ui
    def prediction_label_100px():
        img, _ = image_100px()
        if img is None:
            return ui.p("No image uploaded.", style="color: grey; font-size: 13px;")
        pred_label, _ = predict(img, MODEL_100px)
        return ui.span(f"▶ {pred_label}",
                       style=f"font-size: 18px; font-weight: bold; color: {CLASS_COLORS[pred_label]};")

    @output
    @render.ui
    def prediction_label_50px():
        img, _ = image_50px()
        if img is None:
            return ui.p("No image uploaded.", style="color: grey; font-size: 13px;")
        pred_label, _ = predict(img, MODEL_50px)
        return ui.span(f"▶ {pred_label}",
                       style=f"font-size: 18px; font-weight: bold; color: {CLASS_COLORS[pred_label]};")

    @output
    @render.ui
    def prediction_label_masked():
        img, _ = image_masked()
        if img is None:
            return ui.p("No image uploaded.", style="color: grey; font-size: 13px;")
        img_masked = apply_mask(img)
        pred_label, _ = predict(img_masked, MODEL_masked)
        return ui.span(f"▶ {pred_label}",
                       style=f"font-size: 18px; font-weight: bold; color: {CLASS_COLORS[pred_label]};")

    @output
    @render.ui
    def prediction_desc_100px():
        img, _ = image_100px()
        if img is None:
            return None
        pred_label, _ = predict(img, MODEL_100px)
        return ui.p(CLASS_DESC[pred_label],
                    style="color: #444; font-size: 12px; font-style: italic;")

    @output
    @render.ui
    def prediction_desc_50px():
        img, _ = image_50px()
        if img is None:
            return None
        pred_label, _ = predict(img, MODEL_50px)
        return ui.p(CLASS_DESC[pred_label],
                    style="color: #444; font-size: 12px; font-style: italic;")

    @output
    @render.ui
    def prediction_desc_masked():
        img, _ = image_masked()
        if img is None:
            return None
        img_masked = apply_mask(img)
        pred_label, _ = predict(img_masked, MODEL_masked)
        return ui.p(CLASS_DESC[pred_label],
                    style="color: #444; font-size: 12px; font-style: italic;")

    def make_prob_chart(img, model):
        if img is None:
            fig = go.Figure()
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[dict(text="Upload an image",
                                  showarrow=False,
                                  font=dict(size=12, color="grey"))],
                height=250,
                margin=dict(t=10, b=10),
            )
            return fig
        _, probs = predict(img, model)
        fig = go.Figure(go.Bar(
            x=CLASS_NAMES,
            y=probs,
            marker_color=[CLASS_COLORS[c] for c in CLASS_NAMES],
            text=[f'{p:.1%}' for p in probs],
            textposition='outside',
        ))
        fig.update_layout(
            yaxis=dict(range=[0, 1.15], tickformat='.0%'),
            xaxis=dict(title=''),
            plot_bgcolor='white',
            margin=dict(t=10, b=30, l=40, r=10),
            height=250,
        )
        return fig

    @render_widget
    def prob_chart_100px():
        img, _ = image_100px()
        return make_prob_chart(img, MODEL_100px)

    @render_widget
    def prob_chart_50px():
        img, _ = image_50px()
        return make_prob_chart(img, MODEL_50px)

    @render_widget
    def prob_chart_masked():
        img, _ = image_masked()
        if img is not None:
            img = apply_mask(img)
        return make_prob_chart(img, MODEL_masked)
