import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'<w:left w:val="none"/>'
            f'<w:right w:val="none"/>'
            f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'<w:insideV w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr[0].append(borders)

def add_header(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(14 if level==1 else 10)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.bold = True
    if level == 1:
        run.font.size = Pt(15)
        run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D) # Navy
    elif level == 2:
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0) # Slate blue
    else:
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
    return p

def add_body_p(doc, text="", bold_prefix=None, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_b = p.add_run(bold_prefix)
        r_b.font.name = 'Calibri'
        r_b.font.size = Pt(10)
        r_b.font.bold = True
        r_b.font.color.rgb = RGBColor(0x1A, 0x20, 0x2C)
    if text:
        r = p.add_run(text)
        r.font.name = 'Calibri'
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
    return p

def add_styled_table(doc, headers, data, col_widths=None):
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, color="CBD5E0", sz="4")
    
    # Header
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], 'E2E8F0')
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        for run in p.runs:
            run.font.name = 'Calibri'
            run.font.bold = True
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
            
    # Data rows
    for r_idx, row_data in enumerate(data):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = 'F8FAFC' if r_idx % 2 == 1 else 'FFFFFF'
        for c_idx, cell_value in enumerate(row_data):
            row_cells[c_idx].text = str(cell_value)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=90, bottom=90, left=140, right=140)
            p = row_cells[c_idx].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.1
            for run in p.runs:
                run.font.name = 'Calibri'
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
                
    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)
                
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(4)
    return table

def generate_srs():
    doc = docx.Document()
    
    # Set page margins to standard 1 inch
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # ==========================================
    # COVER PAGE
    # ==========================================
    p_univ = doc.add_paragraph()
    p_univ.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_univ.paragraph_format.space_after = Pt(2)
    r = p_univ.add_run("GLA UNIVERSITY, GREATER NOIDA\n")
    r.font.name = 'Calibri'
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    r2 = p_univ.add_run("Greater Noida – 201310, Uttar Pradesh, India")
    r2.font.name = 'Calibri'
    r2.font.size = Pt(10)
    r2.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    p_gap = doc.add_paragraph()
    p_gap.paragraph_format.space_before = Pt(36)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(8)
    r_t = p_title.add_run("STROKE-LEVEL HANDWRITING SYNTHESIS:\nA VECTOR-OUTPUT GENERATIVE MODEL FOR COMPOSABLE HANDWRITTEN DOCUMENTS\n")
    r_t.font.name = 'Calibri'
    r_t.font.size = Pt(16)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    p_docname = doc.add_paragraph()
    p_docname.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_docname.paragraph_format.space_after = Pt(2)
    r_dn = p_docname.add_run("SOFTWARE REQUIREMENT SPECIFICATION")
    r_dn.font.name = 'Calibri'
    r_dn.font.size = Pt(13)
    r_dn.font.bold = True
    r_dn.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)

    p_ver = doc.add_paragraph()
    p_ver.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_ver.paragraph_format.space_after = Pt(24)
    r_v = p_ver.add_run("Version 1.0")
    r_v.font.name = 'Calibri'
    r_v.font.size = Pt(11)
    r_v.font.italic = True
    r_v.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    p_prog = doc.add_paragraph()
    p_prog.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_prog.paragraph_format.space_after = Pt(24)
    r_p = p_prog.add_run("BACHELOR OF TECHNOLOGY\nin\nCOMPUTER SCIENCE & ENGINEERING - AI")
    r_p.font.name = 'Calibri'
    r_p.font.size = Pt(11)
    r_p.font.bold = True
    r_p.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(4)
    r_s = p_sub.add_run("Submitted By\n")
    r_s.font.name = 'Calibri'
    r_s.font.size = Pt(10)
    r_s.font.color.rgb = RGBColor(0x71, 0x80, 0x96)
    r_name = p_sub.add_run("AGRIIM RASTOGI\n")
    r_name.font.name = 'Calibri'
    r_name.font.size = Pt(12)
    r_name.font.bold = True
    r_name.font.color.rgb = RGBColor(0x1A, 0x20, 0x2C)
    r_roll = p_sub.add_run("Roll No.: 22515500004   |   Section: ND")
    r_roll.font.name = 'Calibri'
    r_roll.font.size = Pt(10)
    r_roll.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    p_gap2 = doc.add_paragraph()
    p_gap2.paragraph_format.space_before = Pt(30)

    # Supervisory table
    sup_table = doc.add_table(rows=1, cols=2)
    sup_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_l = sup_table.rows[0].cells[0]
    cell_r = sup_table.rows[0].cells[1]
    cell_l.width = Inches(3.2)
    cell_r.width = Inches(3.2)
    
    p_l = cell_l.paragraphs[0]
    p_l.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sl = p_l.add_run("Under the Supervision of\n")
    r_sl.font.name = 'Calibri'
    r_sl.font.size = Pt(9.5)
    r_sl.font.bold = True
    r_sl2 = p_l.add_run("Palash Sir / Vansh Sir\nFaculty Mentors, GLA University")
    r_sl2.font.name = 'Calibri'
    r_sl2.font.size = Pt(9)
    r_sl2.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    p_r = cell_r.paragraphs[0]
    p_r.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sr = p_r.add_run("Reviewed By\n")
    r_sr.font.name = 'Calibri'
    r_sr.font.size = Pt(9.5)
    r_sr.font.bold = True
    r_sr2 = p_r.add_run("Project Review Committee\nGLA University, Greater Noida")
    r_sr2.font.name = 'Calibri'
    r_sr2.font.size = Pt(9)
    r_sr2.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    p_gap3 = doc.add_paragraph()
    p_gap3.paragraph_format.space_before = Pt(36)

    p_foot = doc.add_paragraph()
    p_foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_f = p_foot.add_run("DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING\nGLA UNIVERSITY, GREATER NOIDA – 201310\nSEPTEMBER, 2026")
    r_f.font.name = 'Calibri'
    r_f.font.size = Pt(9.5)
    r_f.font.bold = True
    r_f.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

    doc.add_page_break()

    # ==========================================
    # REVISION HISTORY
    # ==========================================
    add_header(doc, "Revision History", level=1)
    add_body_p(doc, "The revision history table logs every formal baseline modification to this specification across project development sprints.")
    
    rev_headers = ["Version", "Date", "Change", "Reason"]
    rev_data = [
        ["1.0", "05-09-2026", "Initial baseline requirements specification.", "First submission and evaluation."]
    ]
    add_styled_table(doc, rev_headers, rev_data, col_widths=[0.9, 1.1, 2.5, 2.0])

    # ==========================================
    # SECTION 1: INTRODUCTION
    # ==========================================
    add_header(doc, "1. Introduction", level=1)
    
    add_header(doc, "1.1 Purpose", level=2)
    add_body_p(doc, "This document specifies the software requirements for Stroke-Level Handwriting Synthesis v1.0, an offline deep-learning system that synthesizes realistic, human-like handwritten vector trajectories from digital text. It defines the formal functional and non-functional requirements against which the implementation, data preprocessing pipeline, and verification test cases are validated.")

    add_header(doc, "1.2 Scope", level=2)
    add_body_p(doc, "The system accepts arbitrary English Latin ASCII character strings (along with optional style seeds and sampling temperature controls) and synthesizes continuous, resolution-independent vector pen strokes formatted as Scalable Vector Graphics (SVG). It is designed for students, educators, and digital creators requiring authentic handwritten documents, study summaries, and scalable vector assets without cloud API latency, recurring subscription costs, or privacy vulnerabilities.")
    add_body_p(doc, "The system does not perform optical character recognition (OCR) or handwriting-to-text transcription. The system does not accept scanned raster document images for text extraction. The system does not interface directly with physical computer numerical control (CNC) or pen-plotter hardware drivers.")

    add_header(doc, "1.3 Definitions", level=2)
    def_headers = ["Term", "Definition"]
    def_data = [
        ["Vector Stroke", "A time-ordered sequence of two-dimensional coordinates (x, y) representing continuous on-paper pen movement between pen-down and pen-up events."],
        ["Online Handwriting", "Handwriting captured as a dynamic temporal sequence of pen movements (dx, dy, dt, pen_up), distinct from static 2D bitmap pixel images."],
        ["CSR Transcription", "Computer Supported Recognition ground-truth text recording what the human volunteer actually wrote, including spelling errors, distinct from the printed prompt."],
        ["X-Height (h_x)", "The vertical height of standard lowercase letters without ascenders or descenders (e.g., 'a', 'c', 'e', 'x'), used as a scale-invariant normalization unit."],
        ["Airborne Jump", "The spatial transition trajectory executed by the pen tip above the writing surface between the end of one stroke and the beginning of the next."]
    ]
    add_styled_table(doc, def_headers, def_data, col_widths=[1.8, 4.7])

    # ==========================================
    # SECTION 2: OVERALL DESCRIPTION
    # ==========================================
    add_header(doc, "2. Overall Description", level=1)
    
    add_header(doc, "2.1 User Classes", level=2)
    user_headers = ["User Class", "Who They Are", "What They May Do"]
    user_data = [
        ["End-User / Note Creator", "Student, educator, or professional generating synthetic handwritten notes.", "Input text prompts, adjust sampling temperature and style sliders, preview generated handwriting, and export standalone SVG vector documents."],
        ["ML Developer / Researcher", "Machine learning practitioner maintaining or extending the neural pipeline.", "Execute preprocessing on raw IAM archives, retrain sequence models, run automated test suites, and benchmark reconstruction drift metrics."]
    ]
    add_styled_table(doc, user_headers, user_data, col_widths=[1.8, 2.2, 2.5])

    add_header(doc, "2.2 Assumptions and Dependencies", level=2)
    add_body_p(doc, "The system depends on local Python 3.13 and PyTorch 2.6+ with CUDA 12.6 runtime acceleration on an NVIDIA RTX 3050 Laptop GPU (6 GB VRAM); when dedicated GPU hardware is unavailable, inference automatically falls back to host CPU execution with reduced generation speed.", bold_prefix="A1. Local Execution Environment: ")
    add_body_p(doc, "The system assumes that the official IAM-OnDB archive is extracted locally within 'datas/raw/iam-ondb/' across its standard subtrees ('ascii/', 'lineStrokes/', and 'original/'), requiring no runtime network access.", bold_prefix="A2. Offline Corpus Availability: ")
    add_body_p(doc, "The system relies entirely on local execution and has zero dependency on third-party cloud APIs or external web services, guaranteeing complete privacy for synthesized documents.", bold_prefix="A3. Zero External Cloud Dependencies: ")

    add_header(doc, "2.3 Constraints", level=2)
    add_body_p(doc, "The IAM-OnDB dataset is licensed strictly for non-commercial academic research; trained model checkpoints and processed datasets cannot be integrated into commercial SaaS platforms without proprietary data collection.", bold_prefix="C1. Academic Licensing Constraint: ")
    add_body_p(doc, "Training and inference batch sizes are bounded by the 6.0 GB VRAM capacity of the target laptop GPU, necessitating sequence length compression and memory-efficient ragged tensor packing.", bold_prefix="C2. Hardware Memory Limitation: ")
    add_body_p(doc, "Whiteboard digitizer sensors record points at a native ~100 Hz sampling rate; physical point density varies with pen velocity, requiring spatial arc-length interpolation to prevent redundant point clusters.", bold_prefix="C3. Digitizer Sampling Rate: ")

    # ==========================================
    # SECTION 3: DATA SOURCE AND BASELINE
    # ==========================================
    add_header(doc, "3. Data Source and Baseline", level=1)
    add_body_p(doc, "The table below establishes the empirical data parameters and baseline benchmark scores computed directly from inspecting the IAM-OnDB corpus.")

    ds_headers = ["Item", "Value", "Required now?"]
    ds_data = [
        ["Dataset name", "IAM On-Line Handwriting Database (IAM-OnDB)", "Yes"],
        ["Source and URL", "University of Bern, FKI (https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)", "Yes"],
        ["Licence / terms of use", "Free for non-commercial academic research and educational evaluation", "Yes"],
        ["Number of rows and columns", "12,178 raw lines across 1,907 forms; 3,731,052 coordinate points; ragged (N, 4) arrays with [dx, dy, pen_up, dt]", "Yes"],
        ["Prediction target", "Next-step coordinate offsets (dx, dy, dt) and binary pen lift state (pen_up) conditioned on text characters", "Yes"],
        ["Task type", "Multi-class discrete token classification (for dx, dy, dt) and binary classification (for pen_up)", "Yes"],
        ["Class balance", "Pen-down drawing steps: 92.1%; Pen-up airborne jumps: 7.9%", "Yes"],
        ["Primary evaluation metric", "Relative Position Drift Error (E_drift measured relative to line x-height)", "Yes"],
        ["Why that metric", "Pen strokes accumulate error via cumsum; per-step error is deceptively small, but cumulative drift causes words to collide or skew off baseline", "Yes"],
        ["Level 0 baseline and its score", "Uniform linear interpolation between word bounding boxes achieves E_drift = 1.28 (completely illegible)", "Yes"],
        ["Known data quality problems", "177 forms missing strokes, 2 forms with line-count mismatches, ~0.25% corrupt non-monotonic timestamps (e.g. -746s step), 1,319 forms (10.8%) missing writer ID", "Yes"],
        ["Level 1 baseline and its score", "Autoregressive 3-layer LSTM with bivariate Gaussian Mixture Density Network (Graves 2013) achieving E_drift = 0.22", "Later"],
        ["Target for this release", "Discrete dual-mode tokenized sequence model achieving median E_drift <= 0.05 with p95 <= 0.08", "Later"]
    ]
    add_styled_table(doc, ds_headers, ds_data, col_widths=[2.1, 3.6, 0.8])

    # ==========================================
    # SECTION 4: SYSTEM ARCHITECTURE
    # ==========================================
    add_header(doc, "4. System Architecture", level=1)
    add_body_p(doc, "The system architecture enforces a strict unidirectional dependency hierarchy. The model pipeline remains completely decoupled from dataset file formats, and the vector renderer serves as the single source of truth for coordinate conversion.")
    
    add_body_p(doc, 
        "+-------------------+        +--------------------+        +----------------------+\n"
        "|   Raw IAM Files   | -----> | Data Ingest & Prep | -----> | Dual-Mode Tokenizer  |\n"
        "| (XML / TXT / Org) |        |   (hws.iam / prep) |        |    (hws.tokenizer)   |\n"
        "+-------------------+        +--------------------+        +----------------------+\n"
        "                                                                      |\n"
        "                                                                      v\n"
        "+-------------------+        +--------------------+        +----------------------+\n"
        "| Vector SVG Output | <----- |   Vector Renderer  | <----- | Neural Sequence Model|\n"
        "| (Scalable Paths)  |        |    (hws.render)    |        | (LSTM / Transformer) |\n"
        "+-------------------+        +--------------------+        +----------------------+\n",
        bold_prefix="Figure 1: Modular System Architecture Pipeline\n"
    )

    arch_headers = ["Component", "Single Responsibility"]
    arch_data = [
        ["Data Ingestion (hws.iam)", "Parses raw IAM XML and TXT files, validates line pairings, inverts y-coordinates, and extracts writer IDs."],
        ["Preprocessing (hws.preprocess)", "Performs spatial arc-length resampling, computes relative offsets (dx, dy, dt), and normalizes coordinates by line x-height."],
        ["Tokenizer (hws.tokenizer)", "Discretizes continuous offsets into dual-mode integer tokens (linear move bins and asinh jump bins)."],
        ["Sequence Model", "Autoregressively predicts next-stroke token probabilities conditioned on input character text embeddings."],
        ["Vector Renderer (hws.render)", "Decodes tokens into continuous coordinates, applies cumulative summation, and exports standards-compliant SVG paths."]
    ]
    add_styled_table(doc, arch_headers, arch_data, col_widths=[2.2, 4.3])

    # ==========================================
    # SECTION 5: FUNCTIONAL REQUIREMENTS
    # ==========================================
    add_header(doc, "5. Functional Requirements", level=1)
    
    add_header(doc, "5.1 Data Ingestion and Preprocessing Requirements", level=2)
    add_body_p(doc, "The system shall parse IAM stroke XML files into an (N, 4) array of floating-point numbers representing x, y, pen_up, and relative elapsed time t.", bold_prefix="FR-1.1  ")
    add_body_p(doc, "The system shall reject input forms where the number of stroke XML files does not match the count of CSR transcription lines, returning an error specifying the mismatched count.", bold_prefix="FR-1.2  ")
    add_body_p(doc, "The system shall invert raw whiteboard sensor y-coordinates at parse time such that positive y increases upward across all internal tensors.", bold_prefix="FR-1.3  ")
    add_body_p(doc, "The system shall resample raw stroke trajectories at uniform spatial arc-length intervals determined by a user-specified step factor parameter.", bold_prefix="FR-1.4  ")
    add_body_p(doc, "The system shall normalize horizontal and vertical coordinate displacements by dividing each offset by the line's estimated x-height.", bold_prefix="FR-1.5  ")

    add_header(doc, "5.2 Model-Backed Generation Requirements", level=2)
    add_body_p(doc, "The system shall accept input text strings containing characters within the 81-character vocabulary and reject unsupported characters, returning an error message identifying the invalid character.", bold_prefix="FR-2.1  ")
    add_body_p(doc, "The system shall return a time-ordered sequence of coordinate offsets (dx, dy, dt, pen_up) for each validated text prompt.", bold_prefix="FR-2.2  ")
    add_body_p(doc, "The system shall allow the user to adjust the sampling temperature between 0.1 and 2.0 to control stroke curvature variation and legibility.", bold_prefix="FR-2.3  ")
    add_body_p(doc, "The system shall, when the neural model fails to generate a valid trajectory, fall back to rendering pre-tokenized template glyphs from the Level 0 baseline with a visible warning label.", bold_prefix="FR-2.4  ")
    add_body_p(doc, "The system shall allow the user to select between distinct writer style seeds to alter handwriting slant, stroke speed, and ligature spacing.", bold_prefix="FR-2.5  ")

    # ==========================================
    # SECTION 6: NON-FUNCTIONAL REQUIREMENTS
    # ==========================================
    add_header(doc, "6. Non-Functional Requirements", level=1)
    add_body_p(doc, "The non-functional requirements define the quantitative performance, security, observability, and efficiency limits of the system.")

    add_body_p(doc, "The system shall synthesize a 30-character handwritten line within 1.5 seconds when executing on an NVIDIA RTX 3050 Laptop GPU.", bold_prefix="NFR-1 (Latency):  ")
    add_body_p(doc, "The system shall execute entirely within the local host environment without transmitting user input text or generated trajectories across any external network interface.", bold_prefix="NFR-2 (Privacy):  ")
    add_body_p(doc, "The system shall log every data preparation execution to a timestamped JSON file recording skipped file counts, reconstruction error percentiles, and vocabulary sizes.", bold_prefix="NFR-3 (Auditability):  ")
    add_body_p(doc, "The data preparation pipeline shall compile all 12,178 corpus lines into processed HDF5 format in less than 180 seconds while consuming under 2.0 GB of system RAM.", bold_prefix="NFR-4 (Resource Efficiency):  ")

    # ==========================================
    # SECTION 7: BUILD PLAN
    # ==========================================
    add_header(doc, "7. Build Plan", level=1)
    
    add_header(doc, "7.1 Minimum Viable Product", level=2)
    add_body_p(doc, "The Minimum Viable Product (MVP) represents the smallest self-contained version of the application capable of demonstrating end-to-end data processing and stroke synthesis.")

    mvp_headers = ["FR / NFR ID", "Requirement (short)", "Why this one is in the MVP"]
    mvp_data = [
        ["FR-1.1", "Parse raw stroke XML files", "Without raw XML trajectory parsing, no coordinate data exists for downstream training."],
        ["FR-1.2", "Reject mismatched form counts", "Prevents misaligned text-stroke pairs from silently corrupting model training datasets."],
        ["FR-1.4", "Spatial arc-length resampling", "Compresses sequence length by 1.84x, making neural training feasible within 6 GB VRAM."],
        ["FR-2.1", "Validate input text against vocabulary", "Guarantees that the sequence model receives only valid character token embeddings."],
        ["FR-2.2", "Return vector stroke sequences", "Core capability required to synthesize and display visible handwriting."],
        ["NFR-1", "Line generation under 1.5 seconds", "Ensures the user interface remains interactive during document composition."],
        ["NFR-2", "100% offline local execution", "Guarantees complete document confidentiality without external cloud dependencies."]
    ]
    add_styled_table(doc, mvp_headers, mvp_data, col_widths=[1.1, 2.5, 2.9])

    add_body_p(doc, "Few-shot style conditioning on user-uploaded handwriting samples. Training a robust few-shot style encoder requires a mature base synthesis model and extensive validation, which is deferred to Sprint III.", bold_prefix="Deliberately not in the MVP: ")

    add_header(doc, "7.2 Build Order", level=2)
    add_body_p(doc, "The project requirements are mapped across fixed institutional sprint dates:")

    stage_headers = ["Stage", "What will be working by then", "FR / NFR IDs", "How I will know it is done"]
    stage_data = [
        ["Sprint I\n27 Sep 2026", "Working data ingestion, arc-length preprocessing, and visual HTML inspector.", "FR-1.1, FR-1.2, FR-1.3, FR-1.4, FR-1.5", "Running 'inspect_data.py' displays side-by-side raw vs preprocessed SVG samples in browser."],
        ["Sprint II\n17 Oct 2026", "Autoregressive neural sequence model with discrete token prediction.", "FR-2.1, FR-2.2, FR-2.3, NFR-1", "CLI script accepts text prompt and outputs synthesized single-line SVG in < 1.5 seconds."],
        ["Sprint III\n07 Nov 2026", "Style conditioning engine and multi-line paragraph layout generator.", "FR-2.4, FR-2.5, NFR-2", "Application generates multi-line documents with selectable style seeds and fallback handling."],
        ["Testing\n17 Nov 2026", "Comprehensive unit test suite, reconstruction drift validation, and bug logging.", "All FRs, NFR-3, NFR-4", "All 58 unit tests pass with documented zero-regression log and p95 drift < 0.08."],
        ["Deployment\n30 Nov 2026", "Standalone packaged desktop application and complete technical documentation.", "All FRs and NFRs", "Clean terminal installation executes end-to-end without developer intervention."]
    ]
    add_styled_table(doc, stage_headers, stage_data, col_widths=[1.2, 2.3, 1.3, 1.7])

    # ==========================================
    # SECTION 8: DATA REQUIREMENTS
    # ==========================================
    add_header(doc, "8. Data Requirements", level=1)
    
    add_header(doc, "8.1 Entities", level=2)
    add_body_p(doc, "The primary persistent entity representing a synthesized or training stroke sequence is defined below:")

    ent_headers = ["Field", "Type", "Constraint / Notes"]
    ent_data = [
        ["id", "UUID", "Primary key; unique identifier for each stroke sequence."],
        ["form_id", "String(16)", "Identifier of parent IAM form (e.g. 'a01-000u'); not null."],
        ["line_index", "Integer", "1-based line index within form; must be positive."],
        ["raw_text", "Text", "CSR transcription string; non-empty."],
        ["point_count", "Integer", "Total coordinate points in trajectory; must be >= 10."],
        ["writer_id", "Integer", "Foreign key to writer profile; nullable for unlabelled forms."],
        ["created_at", "Timestamp", "Timestamp of record generation; must not be future date."]
    ]
    add_styled_table(doc, ent_headers, ent_data, col_widths=[1.3, 1.3, 3.9])

    add_body_p(doc, "TextPrompt, WriterProfile, QuantizedTokenSequence, ExportedSVGDocument.", bold_prefix="Other entities: ")

    add_header(doc, "8.2 Entity Relationship Diagram", level=2)
    add_body_p(doc, "(Not required for v1.0 submission — Entity table defined in Section 8.1; formal ER diagram to be added in v1.1.)")

    # ==========================================
    # SECTION 9: DESIGN DECISIONS
    # ==========================================
    add_header(doc, "9. Design Decisions and Alternatives Considered", level=1)
    add_body_p(doc, "The following architectural decisions were selected based on the empirical mathematical properties of the IAM-OnDB corpus:")

    dec_headers = ["#", "Decision", "Alternative rejected", "Reason"]
    dec_data = [
        ["D1", "Discrete dual-mode tokenization (linear moves + asinh jumps).", "Continuous Mixture Density Networks (MDNs).", "MDNs suffer from variance collapse (sigma -> 0) causing log-loss explosion to NaN, and mode averaging. Discretization reformulates generation as Cross-Entropy, achieving zero NaN crashes and median position drift of 0.032 x-height."],
        ["D2", "Ragged 1D HDF5 storage with slice offset indices.", "Fixed-length zero-padded 2D arrays.", "Padding sequences to the maximum line length (857 points) wastes ~40% disk space and bakes a hard limit into the dataset. Ragged storage saves 22 MB per split and allows dynamic batch collation."],
        ["D3", "Direct SVG vector path export ('<path d=...>')", "Raster PNG image generation via GANs or Diffusion.", "Raster generation fixes image resolution and prevents clean scaling. SVG vector paths are resolution-independent, support interactive styling in CSS, and map directly to CNC pen plotters."]
    ]
    add_styled_table(doc, dec_headers, dec_data, col_widths=[0.4, 2.0, 1.8, 2.3])

    # ==========================================
    # SECTION 10: ACCEPTANCE CRITERIA
    # ==========================================
    add_header(doc, "10. Acceptance Criteria", level=1)
    add_body_p(doc, "The acceptance criteria define concrete, live demonstrations that prove the MVP meets its core requirements:")

    add_body_p(doc, "User supplies the input text string 'Calligrapher AI' in the CLI; the system synthesizes and saves an SVG vector file containing valid, legible stroke paths within 1.5 seconds.", bold_prefix="AC-1 (Live Generation):  ")
    add_body_p(doc, "User executes 'prepare_data.py'; the pipeline compiles 12,178 raw lines, logs reconstruction error percentiles, and outputs 'train.h5', 'val.h5', and 'test.h5' without throwing unhandled exceptions.", bold_prefix="AC-2 (Dataset Pipeline):  ")
    add_body_p(doc, "User inputs a string containing an unsupported character (e.g. 'Hello ~ World'); the system halts generation immediately, displays 'Unsupported character ~ at position 6', and writes no corrupt output file.", bold_prefix="AC-3 (Error Handling):  ")
    add_body_p(doc, "User adjusts sampling temperature from 0.5 to 1.5; the synthesized handwriting visually transitions from uniform, stiff lettering to loose, expressive cursive slants.", bold_prefix="AC-4 (Style Modulation):  ")

    # ==========================================
    # SECTION 11: OUT OF SCOPE
    # ==========================================
    add_header(doc, "11. Out of Scope for This Release", level=1)
    add_body_p(doc, "The following features are deliberately excluded from this release to ensure project delivery within schedule constraints:")

    add_body_p(doc, "Real-time interactive digital stylus canvas. Implementing a live touch drawing canvas for user stroke capture is excluded; this release focuses strictly on text-to-stroke synthesis.", bold_prefix="1. Interactive Stylus Input:  ")
    add_body_p(doc, "Multilingual non-Latin script synthesis. Synthesizing Arabic, Cyrillic, or Devanagari alphabets is excluded; the IAM training corpus is strictly English Latin.", bold_prefix="2. Non-Latin Language Scripts:  ")
    add_body_p(doc, "Few-shot online biometric signature verification. Authenticating human identity from handwritten signatures is a distinct recognition problem and is excluded.", bold_prefix="3. Biometric Identity Verification:  ")

    # ==========================================
    # APPENDICES
    # ==========================================
    add_header(doc, "Appendix A: State Model", level=1)
    add_body_p(doc, "(Not required for v1.0 submission — Document generation follows a synchronous request-response flow; state lifecycle modeling will be formalized in v1.1.)")

    add_header(doc, "Appendix B: Traceability", level=1)
    add_body_p(doc, "(Not required for v1.0 submission — Requirement task cards and unit test IDs are tracked in project repository test suite 'tests/test_tokenizer.py'.)")

    # Output paths
    out_docx = r"c:\Users\basub\OneDrive\Desktop\AR-Handwriting Synthesis\SRS_v1.0_22515500004_HandwritingSynthesis.docx"
    doc.save(out_docx)
    print(f"Generated DOCX: {out_docx}")
    return out_docx

if __name__ == '__main__':
    generate_srs()
