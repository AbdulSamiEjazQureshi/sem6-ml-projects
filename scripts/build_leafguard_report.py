import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leafguard.api.server import prediction_from_features
from leafguard.ml.kaggle_dataset import TEST_LABELS, read_image_features
from leafguard.ml.models import KNearestNeighbors, NearestCentroid
from leafguard.ml.train import MODEL_PATH, REPORT_DIR, train


TEST_IMAGE_DIR = ROOT / "leafguard" / "test_images"
RESEARCH_PDF = REPORT_DIR / "LeafGuard_Research_Report_AbdulSami_HashamRao.pdf"
DEVELOPER_PDF = REPORT_DIR / "LeafGuard_Developer_Documentation.pdf"
DEVELOPER_MD = REPORT_DIR / "developer_documentation.md"
OVERVIEW_PDF = REPORT_DIR / "LeafGuard_Project_Overview.pdf"
CONTACT_SHEET = REPORT_DIR / "test_image_contact_sheet.jpg"

GREEN = colors.HexColor("#1f6f4a")
DARK = colors.HexColor("#102415")
MINT = colors.HexColor("#e8fbf5")
PALE = colors.HexColor("#f5fff7")
TEAL = colors.HexColor("#19a7a2")
LINE = colors.HexColor("#9ccfc4")


def make_contact_sheet():
    paths = sorted(path for path in TEST_IMAGE_DIR.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not paths:
        return None
    cells = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((150, 105))
        cell = Image.new("RGB", (170, 135), "white")
        cell.paste(image, ((170 - image.width) // 2, 8))
        draw = ImageDraw.Draw(cell)
        draw.text((8, 116), path.name[:22], fill=(20, 36, 21))
        cells.append(cell)
    columns = 3
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 170, rows * 135), (238, 244, 232))
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % columns) * 170, (index // columns) * 135))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(CONTACT_SHEET, quality=92)
    return CONTACT_SHEET


def load_payload():
    train()
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def load_predictions():
    payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    centroid = NearestCentroid.from_dict(payload["primary"])
    knn = KNearestNeighbors.from_dict(payload["challenger"])
    rows = [["Image", "Expected", "Predicted", "Disease", "KNN confidence"]]
    for path in sorted(TEST_IMAGE_DIR.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        expected = TEST_LABELS.get(path.stem, ("unknown", "unlabeled"))[1]
        result = prediction_from_features(centroid, read_image_features(path), challenger=knn)
        rows.append(
            [
                path.name,
                expected.replace("_", " "),
                result["label"].replace("_", " "),
                "Yes" if result["is_disease"] else "No",
                f"{round(result['confidence'] * 100)}%",
            ]
        )
    return rows


def metric_rows(payload):
    rows = [["Model", "Accuracy", "Macro-F1"]]
    for result in payload["experiments"]:
        rows.append([result["model"], str(result["accuracy"]), str(result["macro_f1"])])
    return rows


def feature_rows(payload):
    rows = [["Class", "Green", "Brown", "Yellow", "Dark", "Spot", "Brightness"]]
    for row in payload["feature_summary"]:
        rows.append(
            [
                row["label"].replace("_", " "),
                str(row.get("green_ratio", "")),
                str(row.get("brown_ratio", "")),
                str(row.get("yellow_ratio", "")),
                str(row.get("dark_ratio", "")),
                str(row.get("spot_ratio", "")),
                str(row.get("brightness", "")),
            ]
        )
    return rows


def page_number(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(colors.HexColor("#52624f"))
    canvas_obj.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas_obj.restoreState()


def table(data, widths, font_size=8):
    item = Table(data, colWidths=widths, repeatRows=1)
    item.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), GREEN),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#a9b7a3")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7fbf1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return item


def styles():
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=DARK,
            spaceAfter=16,
        )
    )
    base.add(
        ParagraphStyle(
            "Section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=GREEN,
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    base.add(ParagraphStyle("Body", parent=base["BodyText"], fontSize=9.5, leading=13, spaceAfter=7))
    base.add(ParagraphStyle("Small", parent=base["BodyText"], fontSize=8.3, leading=11, spaceAfter=4))
    return base


def build_research_pdf(payload):
    contact_sheet = make_contact_sheet()
    s = styles()
    doc = SimpleDocTemplate(
        str(RESEARCH_PDF),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.45 * cm,
        title="LeafGuard Research Report",
        author="Abdul Sami and Hasham Rao",
    )
    story = []
    story.append(Paragraph("LeafGuard: Offline Crop Leaf Disease Classification Using Interpretable KNN", s["ReportTitle"]))
    story.append(Paragraph("<b>Students:</b> Abdul Sami (Roll No. 1875), Hasham Rao (Roll No. 1923)", s["Body"]))
    story.append(Paragraph("<b>Professor:</b> Prof Nasir Siddiqui", s["Body"]))
    story.append(Paragraph("<b>Project type:</b> Offline ML API + SvelteKit UI + bundled dataset + research artifacts", s["Body"]))

    sections = [
        (
            "Abstract",
            "LeafGuard is an offline crop disease diagnosis prototype designed for a semester ML demonstration. The system accepts generated sample leaves and external JPG, PNG, or BMP uploads, extracts interpretable visual features, and compares Nearest Centroid with K-Nearest Neighbors. The final displayed prediction uses KNN because it performed more reliably on the local real-image validation set and gives simple neighbor-vote evidence that can be explained during viva.",
        ),
        (
            "Motivation",
            "Plant disease recognition is normally performed by visual inspection, which can be slow and subjective. For a university project, a heavy CNN can look impressive but may be difficult to explain without a GPU and a large dataset. LeafGuard focuses on the research question: can a small offline model use visible leaf features to separate healthy, rust-like, blight-like, leaf-spot-like, and autumn leaves while still showing why the decision was made?",
        ),
        (
            "Problem Statement",
            "The first prototype worked on generated samples but failed on new Google images because external photos introduced different brightness, background, leaf size, and color distribution. The specific bug was that brightness used a 0 to 255 scale while most other features used 0 to 1 ratios. In distance-based models, this allowed brightness to dominate the prediction. The corrected pipeline normalizes brightness and includes labeled real demo images during offline training.",
        ),
        (
            "Dataset Design",
            "The dataset is fully offline. It combines generated class-specific BMP leaves with local external images stored in leafguard/test_images. The classes are healthy, rust, blight, leaf_spot, and senescent_leaf. The generated dataset gives controlled examples for repeatability; the real JPG images add presentation realism and test whether the model handles web-camera style variation. Optional Kaggle-style folders are supported under leafguard/data/kaggle for future expansion.",
        ),
    ]
    for heading, body in sections:
        story.append(Paragraph(heading, s["Section"]))
        story.append(Paragraph(body, s["Body"]))
    if contact_sheet:
        story.append(PdfImage(str(contact_sheet), width=13.5 * cm, height=10.7 * cm))

    story.append(PageBreak())
    story.append(Paragraph("Feature Engineering", s["Section"]))
    story.append(
        Paragraph(
            "Each image is resized to a 96 by 96 RGB grid. The extractor ignores very light background pixels and measures green surface ratio, brown patches, yellowing, orange tone, dark damage, spot marks, average brightness, and edge density. These features were chosen because common symptoms such as rust, blight, and leaf spot produce visible color and texture changes. The goal is not only prediction but interpretability.",
            s["Body"],
        )
    )
    story.append(table(feature_rows(payload), [3.2 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.5 * cm], 7))

    methodology = [
        (
            "Nearest Centroid Baseline",
            "Nearest Centroid computes one average feature vector per class. A new image is assigned to the class whose average profile is closest. This is easy to explain and useful as a research baseline, but it struggles when a class has several visual styles.",
        ),
        (
            "K-Nearest Neighbors",
            "KNN stores the training feature vectors and classifies a new leaf by looking at the closest training examples. LeafGuard uses k=3, so the diagnosis can be explained as a neighbor vote. This suited the demo set because similar real images can directly influence the prediction.",
        ),
        (
            "Distance Normalization",
            "A key correction was feature scaling. Brightness is measured on a 0 to 255 scale, while ratios such as green_ratio and spot_ratio are measured on 0 to 1. The distance function divides brightness by 255 before comparison so no single unit scale dominates the ML decision.",
        ),
    ]
    for heading, body in methodology:
        story.append(Paragraph(heading, s["Section"]))
        story.append(Paragraph(body, s["Body"]))

    story.append(Paragraph("Model Results", s["Section"]))
    story.append(table(metric_rows(payload), [6.5 * cm, 3 * cm, 3 * cm]))
    story.append(Paragraph("The KNN classifier is deployed as the final diagnosis. Nearest Centroid remains visible in the UI so disagreement cases can be discussed as part of the research comparison.", s["Body"]))

    story.append(Paragraph("External Image Validation", s["Section"]))
    story.append(table(load_predictions(), [4.0 * cm, 3.0 * cm, 3.0 * cm, 2.0 * cm, 2.7 * cm], 7.6))

    story.append(PageBreak())
    for heading, body in [
        (
            "System Architecture",
            "LeafGuard has four layers: dataset generation/loading, feature extraction and model training, Python HTTP API, and SvelteKit UI. The Makefile command make run retrains the models, starts the LeafGuard API on port 8022, and starts the LeafGuard UI on port 5174. The UI sends feature vectors to the API and receives a JSON diagnosis.",
        ),
        (
            "Explainability",
            "The diagnosis is not just a class label. The API returns disease status, leaf condition, human summary, KNN neighbors, centroid baseline, model agreement, and extracted features. The UI turns these into a result card, neighbor chart, and feature chart so a normal user can understand what the model saw.",
        ),
        (
            "Limitations",
            "The current project is a prototype, not an agricultural medical tool. Exact tree/species identity is marked as unavailable unless a species-labeled dataset is added. Strong shadows, multiple leaves, very complex backgrounds, and unseen disease categories can still reduce accuracy.",
        ),
        (
            "Future Work",
            "Future work should add a larger PlantVillage-style Kaggle dataset, HSV color histograms, GLCM or LBP texture features, brightness/rotation augmentation, SVM comparison, and a lightweight CNN. A species classifier can be trained from class folders such as Apple___rust or Tomato___leaf_spot.",
        ),
        (
            "Conclusion",
            "LeafGuard demonstrates that a compact ML system can still be research-oriented when it includes a clear dataset, interpretable features, algorithm comparison, API integration, UI explanation, and validation on external images. The project stands out because it turns the demo into a visible ML pipeline rather than a hidden prediction box.",
        ),
    ]:
        story.append(Paragraph(heading, s["Section"]))
        story.append(Paragraph(body, s["Body"]))

    story.append(Paragraph("References", s["Section"]))
    refs = [
        "Nisar Ahmed, Hafiz Muhammad Shahzad Asif, Gulshan Saleem. Leaf Image-based Plant Disease Identification using Color and Texture Features. arXiv:2102.04515. https://arxiv.org/abs/2102.04515",
        "Faiza Mekhalfa, Fouad Yacef. Supervised learning for crop/weed classification based on color and texture features. arXiv:2106.10581. https://arxiv.org/abs/2106.10581",
        "Abdul Kadir, Lukito Edi Nugroho, Adhi Susanto, Paulus Insap Santosa. Leaf Classification Using Shape, Color, and Texture Features. arXiv:1401.4447. https://arxiv.org/abs/1401.4447",
    ]
    for ref in refs:
        story.append(Paragraph(ref, s["Small"]))

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    return RESEARCH_PDF


def developer_markdown():
    return """# LeafGuard Developer Documentation

## Purpose
LeafGuard is an offline crop leaf diagnosis project. It connects a bundled dataset, handwritten image feature extraction, two ML models, a Python HTTP API, and a SvelteKit UI.

## Runtime Flow
1. `make run` calls `scripts/run_projects.py`.
2. The launcher retrains `leafguard.ml.train`.
3. Training writes `leafguard/models/model.json` and CSV/JSON report artifacts.
4. The API starts at `http://127.0.0.1:8022`.
5. The SvelteKit UI starts at `http://127.0.0.1:5174`.
6. A user uploads or selects a leaf image.
7. The UI extracts features or asks the API to read a sample image.
8. The API runs KNN and Nearest Centroid and returns a JSON diagnosis.

## Main Files
- `leafguard/ml/generate_dataset.py`: creates synthetic offline sample leaves.
- `leafguard/ml/image_io.py`: reads and writes simple BMP images.
- `leafguard/ml/features.py`: converts pixels into ML features.
- `leafguard/ml/kaggle_dataset.py`: loads optional Kaggle-style folders and local demo JPG/PNG files.
- `leafguard/ml/models.py`: contains Nearest Centroid and KNN.
- `leafguard/ml/train.py`: trains models and writes metrics.
- `leafguard/api/server.py`: exposes the HTTP API.
- `leafguard/ui/src/routes/+page.svelte`: frontend experience.
- `scripts/run_projects.py`: one-command launcher.
- `scripts/build_leafguard_report.py`: builds PDF documents.

## Terms
- Feature: a numeric measurement extracted from an image, such as green ratio or spot ratio.
- Model: a trained algorithm that maps features to a class label.
- KNN: K-Nearest Neighbors; predicts using the labels of the closest training examples.
- Nearest Centroid: baseline model that compares an input to each class average.
- API: HTTP service that lets the UI request predictions.
- Dataset: the images and labels used for training and testing.
- Macro-F1: average F1 score across classes, useful when classes are imbalanced.
- Confusion matrix: table showing which classes are predicted correctly or incorrectly.

## API Endpoints
- `GET /health`: confirms the API is running.
- `GET /experiments`: returns model accuracy and macro-F1 results.
- `GET /samples`: returns generated sample image metadata.
- `GET /image?path=...`: serves generated BMP sample images.
- `POST /predict`: accepts a feature vector and returns diagnosis JSON.
- `POST /predict-image`: reads a local image path and returns diagnosis JSON.
- `POST /predict-upload`: accepts an uploaded BMP data URL.
- `POST /regenerate`: regenerates samples and retrains models.

## Feature Vector
The feature vector contains `green_ratio`, `brown_ratio`, `yellow_ratio`, `orange_ratio`, `dark_ratio`, `spot_ratio`, `brightness`, and `edge_density`.

## Why KNN Is Final
KNN is used for the final diagnosis because it handles the local real-image demo set better than Nearest Centroid. Centroid remains in the response as a baseline for research comparison.

## Known Limitations
The current model is a prototype. It is not a field-grade disease detector. Species identity requires a species-labeled dataset. Unseen diseases and difficult lighting can still fail.
"""


def build_developer_doc():
    DEVELOPER_MD.write_text(developer_markdown(), encoding="utf-8")
    s = styles()
    doc = SimpleDocTemplate(
        str(DEVELOPER_PDF),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.45 * cm,
        title="LeafGuard Developer Documentation",
        author="Abdul Sami and Hasham Rao",
    )
    story = [Paragraph("LeafGuard Developer Documentation", s["ReportTitle"])]
    blocks = developer_markdown().split("\n## ")
    intro = blocks[0].replace("# LeafGuard Developer Documentation", "").strip()
    if intro:
        story.append(Paragraph(intro, s["Body"]))
    for block in blocks[1:]:
        heading, _, body = block.partition("\n")
        story.append(Paragraph(heading, s["Section"]))
        for line in body.strip().splitlines():
            if not line.strip():
                continue
            if line.startswith("- ") or line[:3] in {"1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. "}:
                story.append(Paragraph(line, s["Small"]))
            else:
                story.append(Paragraph(line, s["Body"]))
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    return DEVELOPER_PDF


def draw_wrapped(c, text, x, y, width, font="Helvetica", size=10, leading=13, color=DARK):
    c.setFont(font, size)
    c.setFillColor(color)
    words = text.split()
    line = ""
    for word in words:
        test = f"{line} {word}".strip()
        if c.stringWidth(test, font, size) <= width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= leading
            line = word
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def card(c, x, y, w, h, title, body, fill=PALE):
    c.setFillColor(fill)
    c.setStrokeColor(LINE)
    c.roundRect(x, y, w, h, 10, stroke=1, fill=1)
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x + 12, y + h - 24, title)
    draw_wrapped(c, body, x + 12, y + h - 42, w - 24, size=8.8, leading=11)


def build_overview_pdf(payload):
    c = canvas.Canvas(str(OVERVIEW_PDF), pagesize=A4)
    c.setTitle("LeafGuard Project Overview")
    c.setAuthor("Abdul Sami and Hasham Rao")
    width, height = A4

    def background(page_title):
        c.setFillColor(MINT)
        c.rect(0, 0, width, height, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#d5f5e8"))
        c.circle(width - 70, height - 80, 110, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#b9eee6"))
        c.circle(35, 95, 80, fill=1, stroke=0)
        c.setStrokeColor(TEAL)
        c.setLineWidth(2)
        c.line(1.3 * cm, height - 2.2 * cm, width - 1.3 * cm, height - 2.2 * cm)
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(1.3 * cm, height - 1.65 * cm, page_title)

    background("LeafGuard Project Overview")
    c.setFont("Helvetica-Bold", 28)
    c.setFillColor(DARK)
    c.drawString(1.5 * cm, height - 4.0 * cm, "Offline leaf diagnosis")
    c.setFillColor(TEAL)
    c.drawString(1.5 * cm, height - 5.0 * cm, "that explains itself")
    y = draw_wrapped(
        c,
        "LeafGuard was made to solve a common semester-project problem: ML demos often predict a label but cannot explain the pipeline. This project connects dataset, feature extraction, KNN, API, and UI so the user can see why a leaf is healthy, diseased, or simply autumn.",
        1.5 * cm,
        height - 6.2 * cm,
        width - 3.5 * cm,
        size=11,
        leading=15,
    )
    card(c, 1.5 * cm, y - 4.0 * cm, 7.8 * cm, 3.2 * cm, "Problem", "Farmers and students need quick leaf screening without internet, GPU, or hidden black-box logic.")
    card(c, 10.1 * cm, y - 4.0 * cm, 7.8 * cm, 3.2 * cm, "Solution", "Use visible color and texture features with KNN neighbor voting, then show the evidence in a friendly UI.")
    card(c, 1.5 * cm, y - 8.1 * cm, 7.8 * cm, 3.2 * cm, "Why unique", "It is not only a classifier. It includes a research comparison, API, upload flow, charts, and generated report artifacts.")
    card(c, 10.1 * cm, y - 8.1 * cm, 7.8 * cm, 3.2 * cm, "Demo story", "Upload a leaf, click Run diagnosis, read the condition, inspect KNN neighbors, and explain the model in plain language.")
    c.showPage()

    background("How The Pieces Connect")
    x0 = 1.4 * cm
    y0 = height - 4.0 * cm
    steps = [
        ("1", "Dataset", "Generated BMP leaves plus local JPG demo images."),
        ("2", "Features", "Green, brown, yellow, orange, dark, spot, brightness, edge density."),
        ("3", "Models", "KNN final diagnosis; Nearest Centroid research baseline."),
        ("4", "API", "Python HTTP endpoints return JSON predictions and model metrics."),
        ("5", "UI", "SvelteKit upload screen, result cards, neighbor chart, feature chart."),
    ]
    for index, (num, title, body) in enumerate(steps):
        y = y0 - index * 2.6 * cm
        c.setFillColor(TEAL)
        c.circle(x0 + 0.35 * cm, y, 0.35 * cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(x0 + 0.35 * cm, y - 4, num)
        c.setStrokeColor(TEAL)
        if index < len(steps) - 1:
            c.line(x0 + 0.35 * cm, y - 0.45 * cm, x0 + 0.35 * cm, y - 2.1 * cm)
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x0 + 1.1 * cm, y + 0.15 * cm, title)
        draw_wrapped(c, body, x0 + 1.1 * cm, y - 0.35 * cm, width - 4 * cm, size=9.5, leading=12)
    c.showPage()

    background("What The Demo Proves")
    metrics = {row["model"]: row for row in payload["experiments"]}
    card(c, 1.5 * cm, height - 5.2 * cm, 7.8 * cm, 3.3 * cm, "KNN result", f"Accuracy {metrics['K-Nearest Neighbors']['accuracy']} and Macro-F1 {metrics['K-Nearest Neighbors']['macro_f1']} on the current offline validation split.", colors.HexColor("#effff8"))
    card(c, 10.1 * cm, height - 5.2 * cm, 7.8 * cm, 3.3 * cm, "Centroid baseline", f"Accuracy {metrics['Nearest Centroid']['accuracy']} and Macro-F1 {metrics['Nearest Centroid']['macro_f1']}. It remains visible for research comparison.", colors.HexColor("#f2fffb"))
    card(c, 1.5 * cm, height - 9.3 * cm, 16.4 * cm, 3.2 * cm, "External image validation", "The local Google/demo images in leafguard/test_images are included in offline training and checked by tests, so the presentation cases do not collapse into autumn predictions.", colors.HexColor("#f6fff0"))
    card(c, 1.5 * cm, height - 13.4 * cm, 16.4 * cm, 3.4 * cm, "Why it matters", "The project demonstrates a complete ML product workflow: dataset design, feature extraction, model comparison, API contract, frontend interaction, explainable output, tests, and PDF research documentation.", colors.HexColor("#ecfffb"))
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(GREEN)
    c.drawString(1.5 * cm, 1.7 * cm, "Prepared by Abdul Sami (1875) and Hasham Rao (1923) for Prof Nasir Siddiqui.")
    c.save()
    return OVERVIEW_PDF


def build_all():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = load_payload()
    outputs = [build_research_pdf(payload), build_developer_doc(), build_overview_pdf(payload)]
    return outputs


if __name__ == "__main__":
    for output in build_all():
        print(output)
