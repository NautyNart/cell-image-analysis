from pathlib import Path

from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go


heat_100 = reactive.Value(np.zeros((7, 7)))
heat_50 = reactive.Value(np.zeros((7, 7)))
heat_masked = reactive.Value(np.zeros((7, 7)))

def get_safe_heatmap(model, img):
    try:
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = model.backbone(x)
            if feat.dim() == 4:
                map = feat.mean(dim=-1).squeeze().cpu().numpy()
            else:
                map = feat.mean(dim=-1).squeeze().reshape(7,7).cpu().numpy()
            map = (map - map.min()) / (map.max() - map.min() + 1e-8)
            return map
    except:
        return np.zeros((7,7))

def predict_with_heatmap(orig_predict, img, model, heat_val):
    if img is None:
        return None, None
    label, prob = orig_predict(img, model)
    heat = get_safe_heatmap(model, img)
    heat_val.set(heat)
    return label, prob


BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH_100px  = MODELS_DIR / "100px_best_model.pth"
MODEL_PATH_50px   = MODELS_DIR / "50px_best_model.pth"
MODEL_PATH_masked = MODELS_DIR / "masked_best_model.pth"

NUM_CLASSES  = 3
CLASS_NAMES = ['Tumour', 'DCIS_1', 'DCIS_2']

CLASS_DESC = {
    'Tumour': 'Merged invasive tumour class containing Invasive_Tumor and Prolif_Invasive_Tumor annotations.',
    'DCIS_1': 'Ductal carcinoma in situ subtype identified from Xenium-integrated cell annotations.',
    'DCIS_2': 'Second ductal carcinoma in situ subtype identified from Xenium-integrated cell annotations.',
}

CLASS_COLORS = {
    'Tumour': '#C94C5A',
    'DCIS_1': '#4F7FA4',
    'DCIS_2': '#E89A5B'
}
MEAN = [0.7252153312, 0.6010258969, 0.8289264835]
STD  = [0.1467957122, 0.1483485278, 0.07618737224]
IMG_SIZE = 224

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

MASK_KEEP_RATIO = 0.30
MASK_FILL       = np.array([123, 116, 103], dtype=np.uint8)

class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        assert patch_size == 4
        assert embed_dim % 8 == 0
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.grid_size   = (img_size[0] // patch_size[0],
                            img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten     = flatten
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for _ in range(2):
            stem += [
                nn.Conv2d(input_dim, output_dim,
                          kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            ]
            input_dim  = output_dim
            output_dim = output_dim * 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x

def ctranspath():
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
    model.head = nn.Identity()
    embed_dim = model.patch_embed.proj.out_channels
    model.patch_embed = ConvStem(
        img_size=224, patch_size=4, in_chans=3,
        embed_dim=embed_dim, norm_layer=nn.LayerNorm, flatten=True
    )
    return model

class CTranspathClassifier(nn.Module):
    def __init__(self, backbone, num_classes=3, feat_dim=768, dropout=0.5):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.mean(dim=(1, 2))
        elif features.dim() == 3:
            features = features.mean(dim=1)
        return self.classifier(features)

def load_model(weights_path):
    backbone = ctranspath()
    model    = CTranspathClassifier(backbone, num_classes=NUM_CLASSES)
    state    = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

MODEL_100px  = load_model(MODEL_PATH_100px)
MODEL_50px   = load_model(MODEL_PATH_50px)
MODEL_masked = load_model(MODEL_PATH_masked)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

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

def predict(img, model):
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1).squeeze().numpy()
    pred_label = CLASS_NAMES[int(np.argmax(probs))]
    return pred_label, probs

tab2_ui = ui.nav_panel(
    "Prediction",
    ui.div(
        ui.div(
            ui.div(
                ui.h3("Upload Matched Patch Set"),
                ui.p(
                    "Upload corresponding 100px, 50px, and masked patches generated from the same cell location.",
                    class_="condition-subtitle"
                ),
            
                ui.row(
                    ui.column(
                        4,
                        ui.div(
                            ui.div("100px Patch", class_="upload-label-title"),
                            ui.div("Full spatial context", class_="upload-label-subtitle"),
                            ui.input_file(
                                "upload_100px",
                                None,
                                accept=[".png", ".jpg", ".jpeg"],
                                multiple=False
                            ),
                            class_="upload-box"
                        )
                    ),
                    ui.column(
                        4,
                        ui.div(
                            ui.div("50px Patch", class_="upload-label-title"),
                            ui.div("Reduced spatial context", class_="upload-label-subtitle"),
                            ui.input_file(
                                "upload_50px",
                                None,
                                accept=[".png", ".jpg", ".jpeg"],
                                multiple=False
                            ),
                            class_="upload-box"
                        )
                    ),
                    ui.column(
                        4,
                        ui.div(
                            ui.div("Masked Patch", class_="upload-label-title"),
                            ui.div("Local morphology only", class_="upload-label-subtitle"),
                            ui.input_file(
                                "upload_masked",
                                None,
                                accept=[".png", ".jpg", ".jpeg"],
                                multiple=False
                            ),
                            class_="upload-box"
                        )
                    )
                ),
            
                ui.p(
                    "Predictions are generated independently for each spatial context condition.",
                    class_="small-muted",
                    style="margin-top: 18px;"
                ),
                class_="section-card upload-panel"
            ),
            
            ui.row(
                ui.column(
                    4,
                    ui.div(
                        ui.h4("100px Patch"),
                        ui.p("Full spatial context", class_="condition-subtitle"),
                        ui.output_image("show_image_100px"),
                        ui.output_ui("prediction_label_100px"),
                        ui.output_ui("prediction_desc_100px"),
                        output_widget("prob_chart_100px"),
                        class_="plot-card"
                    )
                ),
                ui.column(
                    4,
                    ui.div(
                        ui.h4("50px Patch"),
                        ui.p("Reduced spatial context", class_="condition-subtitle"),
                        ui.output_image("show_image_50px"),
                        ui.output_ui("prediction_label_50px"),
                        ui.output_ui("prediction_desc_50px"),
                        output_widget("prob_chart_50px"),
                        class_="plot-card"
                    )
                ),
                ui.column(
                    4,
                    ui.div(
                        ui.h4("Masked Patch"),
                        ui.p("Local morphology only", class_="condition-subtitle"),
                        ui.output_image("show_image_masked"),
                        ui.output_ui("prediction_label_masked"),
                        ui.output_ui("prediction_desc_masked"),
                        output_widget("prob_chart_masked"),
                        class_="plot-card"
                    )
                ),
                style="margin-top: 28px;"
            ),

            class_="content-wrap"
        ),
        class_="page-shell"
    )
)


def tab2_server(input, output, session):
    session.heat100 = heat_100
    session.heat50 = heat_50
    session.heatmask = heat_masked

    session.upload_100px_path = None
    session.upload_50px_path = None
    session.upload_masked_path = None

    @reactive.calc
    def image_100px():
        f = input.upload_100px()
        if not f:
            session.upload_100px_path = None
            return None, None
        path = f[0]['datapath']
        session.upload_100px_path = path
        return Image.open(path).convert('RGB'), path

    @reactive.calc
    def image_50px():
        f = input.upload_50px()
        if not f:
            session.upload_50px_path = None
            return None, None
        path = f[0]['datapath']
        session.upload_50px_path = path
        return Image.open(path).convert('RGB'), path

    @reactive.calc
    def image_masked():
        f = input.upload_masked()
        if not f:
            session.upload_masked_path = None
            return None, None
        path = f[0]['datapath']
        session.upload_masked_path = path
        return Image.open(path).convert('RGB'), path

    @output
    @render.image
    def show_image_100px():
        img, p = image_100px()
        return {"src": p, "width":"220px", "height":"220px"} if img else None
    @output
    @render.image
    def show_image_50px():
        img, p = image_50px()
        return {"src": p, "width":"220px", "height":"220px"} if img else None
    @output
    @render.image
    def show_image_masked():
        img, p = image_masked()
        return {"src": p, "width":"220px", "height":"220px"} if img else None

    @output
    @render.ui
    def prediction_label_100px():
        img, _ = image_100px()
        if not img: return ui.p("No image uploaded.", style="color:grey")
        l, _ = predict_with_heatmap(predict, img, MODEL_100px, heat_100)
        return ui.div(f"▶ {l}", class_="prediction-label", style=f"color:{CLASS_COLORS[l]};")
    @output
    @render.ui
    def prediction_label_50px():
        img, _ = image_50px()
        if not img: return ui.p("No image uploaded.", style="color:grey")
        l, _ = predict_with_heatmap(predict, img, MODEL_50px, heat_50)
        return ui.div(f"▶ {l}", class_="prediction-label", style=f"color:{CLASS_COLORS[l]};")
    @output
    @render.ui
    def prediction_label_masked():
        img, _ = image_masked()
        if not img: return ui.p("No image uploaded.", style="color:grey")
        m = apply_mask(img)
        l, _ = predict_with_heatmap(predict, m, MODEL_masked, heat_masked)
        return ui.div(f"▶ {l}", class_="prediction-label", style=f"color:{CLASS_COLORS[l]};")

    @output
    @render.ui
    def prediction_desc_100px():
        img, _ = image_100px()
        if not img: return
        l, _ = predict(img, MODEL_100px)
        return ui.p(CLASS_DESC[l], class_="prediction-desc")
    @output
    @render.ui
    def prediction_desc_50px():
        img, _ = image_50px()
        if not img: return
        l, _ = predict(img, MODEL_50px)
        return ui.p(CLASS_DESC[l], class_="prediction-desc")
    @output
    @render.ui
    def prediction_desc_masked():
        img, _ = image_masked()
        if not img: return
        m = apply_mask(img)
        l, _ = predict(m, MODEL_masked)
        return ui.p(CLASS_DESC[l], class_="prediction-desc")

    def make_prob_chart(img, model):
        if not img:
            fig = go.Figure()
            fig.update_layout(
                xaxis_visible=False,
                yaxis_visible=False,
                height=210,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="white",
                plot_bgcolor="white"
            )
            return fig
        _, prob = predict(img, model)
        fig = go.Figure(go.Bar(x=CLASS_NAMES, y=prob, marker_color=[CLASS_COLORS[c] for c in CLASS_NAMES], text=[f"{p:.1%}" for p in prob]))
        fig.update_layout(
            yaxis_range=[0, 1.05],
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=210,
            margin=dict(l=35, r=10, t=10, b=35)
        )
        return fig

    @render_widget
    def prob_chart_100px(): return make_prob_chart(image_100px()[0], MODEL_100px)
    @render_widget
    def prob_chart_50px(): return make_prob_chart(image_50px()[0], MODEL_50px)
    @render_widget
    def prob_chart_masked():
        img = image_masked()[0]
        if img: img = apply_mask(img)
        return make_prob_chart(img, MODEL_masked)
