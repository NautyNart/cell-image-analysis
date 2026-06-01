from shiny import ui

tab1_ui = ui.nav_panel(
    "Overview",
    ui.div(
        ui.div(
            ui.div(
                ui.h1("Breast Cancer Tumour Subtype Classifier"),
                ui.p("Evaluating the effect of spatial context on deep learning-based subtype classification"),
                class_="hero"
            ),

            ui.div(
                ui.h4("Research Question"),
                ui.p("How does spatial contextual richness affect deep learning-based classification of breast cancer tumour subtypes?"),
                class_="research-box"
            ),

            ui.row(
                ui.column(
                    4,
                    ui.div(
                        ui.h4("Dataset"),
                        ui.p("H&E image patches were generated from Xenium-profiled human breast cancer tissue sections. Cell subtype annotations were informed through integration of spatial transcriptomic and single-cell RNA sequencing data, following the framework described by Janesick et al. (2023)."),
                        ui.p("The classification task focuses on three tumour-related classes:"),
                        ui.tags.ul(
                            ui.tags.li("Tumour"),
                            ui.tags.li("DCIS_1"),
                            ui.tags.li("DCIS_2")
                        ),
                        class_="research-card"
                    )
                ),
                ui.column(
                    4,
                    ui.div(
                        ui.h4("Model"),
                        ui.p("The application uses a fine-tuned CTransPath architecture, a pathology-specific CNN-transformer hybrid pretrained on large-scale histopathology datasets."),
                        ui.p("Predictions are compared across three spatial context conditions:"),
                        ui.tags.ul(
                            ui.tags.li("100px patches: full spatial context"),
                            ui.tags.li("50px patches: reduced spatial context"),
                            ui.tags.li("masked patches: local morphology only")
                        ),
                        ui.p("This experimental design enables controlled comparison of how spatial contextual richness influences subtype classification behaviour."),
                        class_="research-card"
                    )
                ),
                ui.column(
                    4,
                    ui.div(
                        ui.h4("Prediction Workflow"),
                        ui.tags.ol(
                            ui.tags.li("Upload matched 100px, 50px, and masked patches from the same cell location"),
                            ui.tags.li("Compare predicted subtype probabilities across context conditions"),
                            ui.tags.li("Assess how prediction confidence and subtype assignment vary across context conditions")
                        ),
                        class_="research-card"
                    )
                )
            ),

            ui.div(
                ui.h4("Cell Type Reference"),
                ui.row(
                    ui.column(
                        4,
                        ui.div(
                            ui.div("Tumour", class_="cell-ref-title"),
                            ui.p("Merged invasive tumour class containing Invasive_Tumor and Prolif_Invasive_Tumor annotations.",
                                 class_="cell-ref-text"),
                        )
                    ),
                    ui.column(
                        4,
                        ui.div(
                            ui.div("DCIS_1", class_="cell-ref-title"),
                            ui.p("Ductal carcinoma in situ subtype identified from Xenium-integrated cell annotations.",
                                 class_="cell-ref-text"),
                        )
                    ),
                    ui.column(
                        4,
                        ui.div(
                            ui.div("DCIS_2", class_="cell-ref-title"),
                            ui.p("Second ductal carcinoma in situ subtype identified from Xenium-integrated cell annotations.",
                                 class_="cell-ref-text"),
                        )
                    )
                ),
                class_="section-card"
            ),
            
            ui.div(
                ui.h4("Why Spatial Context Matters"),
                ui.p("Histopathological interpretation relies on more than isolated cellular morphology. Tissue architecture, stromal organisation, and local spatial relationships often contribute to tumour subtype identification, particularly in morphologically similar regions."),
                ui.p("This project investigates whether deep learning models utilise surrounding tissue context when classifying breast cancer tumour subtypes from H&E image patches. By comparing predictions across full-context, reduced-context, and masked patch conditions, the app provides an interactive framework for examining how contextual richness influences model behaviour and classification confidence."),
                ui.p("The attention visualisations further allow qualitative assessment of whether model focus shifts from local nuclear morphology toward broader tissue-level structure as spatial context increases."),
                class_="section-card"
            ),

            class_="content-wrap"
        ),
        class_="page-shell"
    )
)

def tab1_server(input, output, session):
    pass
