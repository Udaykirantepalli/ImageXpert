"""
Streamlit entry point for the AI-Powered Image Analysis platform.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.contours import analyze_contours
from src.edge_detection import CANNY_PRESETS, compare_edges
from src.filtering import all_filters
from src.image_analysis import analyze_image
from src.image_loader import ImageUploadError, load_image
from src.image_quality import assess_quality
from src.preprocessing import clahe, equalize, to_gray
from src.recommendations import generate_recommendations
from src.visualization import comparison_chart, csv_bytes, png_bytes


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Image Analysis",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# BRIGHT UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background: linear-gradient(
            135deg,
            #fff7ed 0%,
            #eff6ff 45%,
            #f0fdf4 100%
        );
    }

    .main {
        background: transparent;
    }

    /* =====================================================
       HEADER
       ===================================================== */

    .hero {
        background: linear-gradient(
            135deg,
            #2563eb 0%,
            #7c3aed 50%,
            #db2777 100%
        );

        padding: 32px 35px;
        border-radius: 22px;
        margin-bottom: 25px;

        box-shadow:
            0 10px 30px rgba(37, 99, 235, 0.20);
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero p {
        color: #f8fafc !important;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* =====================================================
       SECTION HEADINGS
       ===================================================== */

    .section-title {
        color: #1e3a8a;
        font-size: 1.55rem;
        font-weight: 750;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .section-subtitle {
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }

    /* =====================================================
       METRIC CARDS
       ===================================================== */

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 18px;
        box-shadow:
            0 5px 18px rgba(30, 64, 175, 0.08);
    }

    div[data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #2563eb !important;
        font-weight: 800;
    }

    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #eff6ff 0%,
            #f5f3ff 50%,
            #fff1f2 100%
        );

        border-right: 1px solid #dbeafe;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1e40af !important;
    }

    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        background: linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );

        color: white !important;

        border: none;
        border-radius: 12px;

        font-weight: 700;

        padding: 10px 18px;

        box-shadow:
            0 5px 15px rgba(37, 99, 235, 0.20);

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #1d4ed8,
            #6d28d9
        );

        transform: translateY(-1px);

        box-shadow:
            0 8px 20px rgba(37, 99, 235, 0.28);
    }

    /* =====================================================
       DOWNLOAD BUTTONS
       ===================================================== */

    .stDownloadButton > button {
        background: #ecfeff !important;
        color: #0369a1 !important;
        border: 1px solid #67e8f9 !important;
        border-radius: 10px;
        font-weight: 650;
    }

    .stDownloadButton > button:hover {
        background: #cffafe !important;
        color: #075985 !important;
    }

    /* =====================================================
       TABS
       ===================================================== */

    button[data-baseweb="tab"] {
        color: #475569;
        font-weight: 650;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb !important;
    }

    /* =====================================================
       EXPANDERS
       ===================================================== */

    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 14px;
        box-shadow:
            0 4px 15px rgba(30, 64, 175, 0.06);
    }

    /* =====================================================
       DATAFRAME
       ===================================================== */

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #dbeafe;
    }

    /* =====================================================
       INFO / SUCCESS / WARNING / ERROR
       ===================================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* =====================================================
       UPLOAD BOX
       ===================================================== */

    section[data-testid="stFileUploaderDropzone"] {
        background: #ffffff;
        border: 2px dashed #93c5fd;
        border-radius: 14px;
    }

    /* =====================================================
       IMAGE CONTAINERS
       ===================================================== */

    div[data-testid="stImage"] {
        background: #ffffff;
        padding: 8px;
        border-radius: 14px;
        box-shadow:
            0 5px 15px rgba(30, 64, 175, 0.08);
    }

    /* =====================================================
       CODE BLOCKS
       ===================================================== */

    pre {
        background: #f8fafc !important;
        border: 1px solid #dbeafe !important;
        border-radius: 10px !important;
    }

    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        text-align: center;
        padding: 25px;
        margin-top: 30px;
        color: #64748b;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

    <h1>
     AI-Powered Image Analysis
    & Edge Detection Platform
    </h1>

    <p>
    Interactive Computer Vision & Digital Image
    Processing System
    </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CENTRAL UPLOAD AREA
# ============================================================

with st.container(border=True):

    st.markdown(
        '<div class="section-title" style="text-align:center;">'
        '📤 Upload an Image to Begin'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle" style="text-align:center;">'
        'Supported formats: JPG, JPEG and PNG<br><br>'
        'Analyze image quality, apply filters, compare edge-detection '
        'algorithms, detect contours and receive intelligent processing '
        'recommendations.'
        '</div>',
        unsafe_allow_html=True,
    )

    _, upload_column, _ = st.columns((1, 2, 1))

    with upload_column:

        upload = st.file_uploader(
            "📤 Upload Image",
            type=["jpg", "jpeg", "png"],
            help=(
                "JPG/PNG only. Maximum 15 MB. "
                "Image contents are validated."
            ),
        )


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🖼️ Image Processing"
    )

    st.caption(
        "Upload an image and explore its "
        "computer-vision characteristics."
    )

    st.markdown(
        "### ⚙️ Processing Controls"
    )

    kernel = st.select_slider(
        "🔲 Filter Kernel",
        options=[3, 5, 7],
        value=5,
    )

    canny_preset = st.selectbox(
        "🎯 Canny Threshold",
        list(CANNY_PRESETS),
        index=1,
    )

    sobel_kernel = st.select_slider(
        "〽️ Sobel Kernel",
        options=[3, 5, 7],
        value=3,
    )

    min_area = st.slider(
        "🔵 Minimum Contour Area",
        10,
        2000,
        100,
        10,
    )

    st.divider()

    if st.button(
        "🔄 Reset Application",
        use_container_width=True,
    ):
        st.session_state.clear()
        st.rerun()


# ============================================================
# NO IMAGE
# ============================================================

if not upload:
    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    loaded = load_image(
        upload.getvalue()
    )

except ImageUploadError as exc:

    st.error(str(exc))
    st.stop()

except Exception:

    st.error(
        "Unable to process this image. "
        "Please upload a valid JPG or PNG file."
    )

    st.stop()


# ============================================================
# PROCESS IMAGE
# ============================================================

@st.cache_data(
    show_spinner=False
)
def process(
    image_bytes: bytes,
    kernel_size: int,
    preset: str,
    sobel: int,
    area: int,
):

    image = load_image(
        image_bytes
    ).array

    analysis = analyze_image(
        image
    )

    quality = assess_quality(
        image
    )

    filters = all_filters(
        image,
        kernel_size
    )

    edges, comparison = compare_edges(
        image,
        preset,
        sobel
    )

    contour_image, contour_data = analyze_contours(
        image,
        area
    )

    canny_density = next(
        row["edge_density"]
        for row in comparison
        if row["algorithm"] == "Canny"
    )

    recommendations = generate_recommendations(
        analysis,
        quality,
        canny_density
    )

    return (
        image,
        analysis,
        quality,
        filters,
        edges,
        comparison,
        contour_image,
        contour_data,
        recommendations,
    )


# ============================================================
# RUN PROCESSING
# ============================================================

try:

    (
        image,
        analysis,
        quality,
        filters,
        edge_images,
        comparison,
        contour_image,
        contour_data,
        recommendations,
    ) = process(
        upload.getvalue(),
        kernel,
        canny_preset,
        sobel_kernel,
        min_area,
    )

except Exception:

    st.error(
        "Processing could not be completed safely. "
        "Try a different image or less complex settings."
    )

    st.stop()


# ============================================================
# RESIZE WARNING
# ============================================================

if loaded.was_resized:

    st.warning(
        "⚠️ The image was resized to a safe maximum "
        "dimension before analysis."
    )


# ============================================================
# QUICK SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📊 Image Analysis Summary</div>',
    unsafe_allow_html=True,
)

summary1, summary2, summary3, summary4 = st.columns(4)

with summary1:

    st.metric(
        "📐 Resolution",
        f"{analysis['width']} × {analysis['height']}",
    )

with summary2:

    st.metric(
        "💡 Brightness",
        f"{quality['brightness']:.1f}",
    )

with summary3:

    st.metric(
        "✨ Sharpness",
        quality["sharpness"],
    )

with summary4:

    st.metric(
        "🏆 Quality",
        f"{quality['quality_score']}/100",
    )


st.divider()


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "🖼️ Image Overview",
        "⚙️ Preprocessing",
        "🎨 Filtering",
        "〽️ Edge Detection",
        "📊 Comparison",
        "🔵 Contour Analysis",
        "🧠 AI Recommendations",
    ]
)


# ============================================================
# TAB 1 — IMAGE OVERVIEW
# ============================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">🖼️ Image Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Detailed information extracted from the uploaded image."
        "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        (3, 2)
    )

    with left:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True,
        )

    with right:

        metrics = {
            "Resolution": (
                f"{analysis['width']} × "
                f"{analysis['height']}"
            ),
            "Channels": analysis["channels"],
            "Aspect Ratio": analysis["aspect_ratio"],
            "Brightness": (
                f"{quality['brightness']:.1f}"
            ),
            "Contrast": (
                f"{quality['contrast']:.1f}"
            ),
            "Sharpness": quality["sharpness"],
            "Noise": quality["noise_level"],
            "Quality Score": (
                f"{quality['quality_score']}/100"
            ),
            "Classification": quality[
                "classification"
            ],
        }

        st.dataframe(
            pd.DataFrame(
                metrics.items(),
                columns=[
                    "Metric",
                    "Value",
                ],
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.download_button(
        "⬇️ Download Analysis CSV",
        csv_bytes(
            pd.DataFrame(
                [analysis | quality]
            )
        ),
        "image_analysis.csv",
        "text/csv",
    )


# ============================================================
# TAB 2 — PREPROCESSING
# ============================================================

with tabs[1]:

    st.markdown(
        '<div class="section-title">'
        "⚙️ Image Preprocessing"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "Compare different preprocessing techniques.",
    )

    preprocessing_results = [
        (
            "Original",
            image,
        ),
        (
            "Grayscale",
            to_gray(image),
        ),
        (
            "Equalized",
            equalize(image),
        ),
        (
            "CLAHE",
            clahe(image),
        ),
    ]

    cols = st.columns(4)

    for col, (
        label,
        result,
    ) in zip(
        cols,
        preprocessing_results,
    ):

        with col:

            st.image(
                result,
                caption=label,
                use_container_width=True,
            )


# ============================================================
# TAB 3 — FILTERING
# ============================================================

with tabs[2]:

    st.markdown(
        '<div class="section-title">'
        "🎨 Image Filtering"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "Compare multiple noise-reduction and smoothing filters."
    )

    cols = st.columns(
        len(filters)
    )

    for col, (
        name,
        result,
    ) in zip(
        cols,
        filters.items(),
    ):

        with col:

            st.image(
                result,
                caption=name,
                use_container_width=True,
            )

    st.divider()

    selected_filter = st.selectbox(
        "Select filter result to download",
        list(filters),
    )

    st.download_button(
        "⬇️ Download Processed Image",
        png_bytes(
            filters[selected_filter]
        ),
        f"{selected_filter.lower()}_filtered.png",
        "image/png",
    )


# ============================================================
# TAB 4 — EDGE DETECTION
# ============================================================

with tabs[3]:

    st.markdown(
        '<div class="section-title">'
        "〽️ Edge Detection"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Compare five classical edge-detection algorithms:
        Roberts, Prewitt, Sobel, Laplacian and Canny.
        """
    )

    cols = st.columns(
        len(edge_images)
    )

    for col, (
        name,
        result,
    ) in zip(
        cols,
        edge_images.items(),
    ):

        with col:

            st.image(
                result,
                caption=name,
                use_container_width=True,
            )

    st.divider()

    selected_edge = st.selectbox(
        "Select edge result to download",
        list(edge_images),
    )

    st.download_button(
        "⬇️ Download Edge Image",
        png_bytes(
            edge_images[selected_edge]
        ),
        f"{selected_edge.lower()}_edges.png",
        "image/png",
    )


# ============================================================
# TAB 5 — COMPARISON
# ============================================================

with tabs[4]:

    st.markdown(
        '<div class="section-title">'
        "📊 Edge Detection Comparison"
        "</div>",
        unsafe_allow_html=True,
    )

    frame = pd.DataFrame(
        comparison
    )

    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
    )

    chart1, chart2 = st.columns(2)

    with chart1:

        st.plotly_chart(
            comparison_chart(
                frame,
                "edge_density",
            ),
            use_container_width=True,
        )

    with chart2:

        st.plotly_chart(
            comparison_chart(
                frame,
                "processing_time_ms",
            ),
            use_container_width=True,
        )

    st.download_button(
        "⬇️ Download Comparison CSV",
        csv_bytes(frame),
        "edge_comparison.csv",
        "text/csv",
    )


# ============================================================
# TAB 6 — CONTOUR ANALYSIS
# ============================================================

with tabs[5]:

    st.markdown(
        '<div class="section-title">'
        "🔵 Contour & Object Boundary Analysis"
        "</div>",
        unsafe_allow_html=True,
    )

    st.image(
        contour_image,
        caption=(
            "Detected contours and "
            "bounding rectangles"
        ),
        use_container_width=True,
    )

    if contour_data[
        "number_of_contours"
    ]:

        contour_col1, contour_col2 = st.columns(2)

        with contour_col1:

            st.metric(
                "🔵 Detected Contours",
                contour_data[
                    "number_of_contours"
                ],
            )

        with contour_col2:

            st.metric(
                "📦 Approximate Objects",
                contour_data[
                    "approximate_object_count"
                ],
            )

        st.dataframe(
            pd.DataFrame(
                contour_data["objects"]
            ),
            hide_index=True,
            use_container_width=True,
        )

    else:

        st.info(
            "No significant contours were detected "
            "in this image."
        )

    st.download_button(
        "⬇️ Download Contour Image",
        png_bytes(
            contour_image
        ),
        "contours.png",
        "image/png",
    )


# ============================================================
# TAB 7 — AI RECOMMENDATIONS
# ============================================================

with tabs[6]:

    st.markdown(
        '<div class="section-title">'
        "🧠 Intelligent Image Processing Recommendations"
        "</div>",
        unsafe_allow_html=True,
    )

    st.info(
        """
        These recommendations are generated using transparent
        rule-based image-processing logic. They are not predictions
        from a trained AI or deep-learning model.
        """
    )

    for recommendation in recommendations:

        with st.expander(
            recommendation["recommendation"],
            expanded=True,
        ):

            st.markdown(
                f"""
                **💡 Why this recommendation?**

                {recommendation['reason']}
                """
            )

            st.success(
                f"Confidence: "
                f"{recommendation['confidence']}"
            )

            st.markdown(
                "**🔄 Suggested Processing Pipeline**"
            )

            st.code(
                recommendation[
                    "suggested_pipeline"
                ],
                language=None,
            )

    st.caption(
        "ℹ️ The image-quality score is an analytical "
        "heuristic and is not a scientifically validated "
        "universal quality measurement."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    <strong>
     AI-Powered Image Analysis & Edge Detection Platform
    </strong>

    <br><br>

    Built with Python • OpenCV • NumPy • Scikit-image •
    Streamlit • Plotly

    <br>

    Computer Vision & Digital Image Processing

    </div>
    """,
    unsafe_allow_html=True,
)
