import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=50, bottom=50, left=90, right=90):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_table_borders(table, color="CBD5E0", sz="4", val="single"):
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
    p.paragraph_format.space_before = Pt(8 if level==1 else 5)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.bold = True
    if level == 1:
        run.font.size = Pt(12.5)
        run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    elif level == 2:
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)
    else:
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
    return p

def add_body_p(doc, text="", bold_prefix=None, space_after=2, italic_prefix=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.08
    if bold_prefix:
        r_b = p.add_run(bold_prefix)
        r_b.font.name = 'Calibri'
        r_b.font.size = Pt(9.0)
        r_b.font.bold = True
        r_b.font.color.rgb = RGBColor(0x1A, 0x20, 0x2C)
    if italic_prefix:
        r_i = p.add_run(italic_prefix)
        r_i.font.name = 'Calibri'
        r_i.font.size = Pt(9.0)
        r_i.font.italic = True
        r_i.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)
    if text:
        r = p.add_run(text)
        r.font.name = 'Calibri'
        r.font.size = Pt(9.0)
        r.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
    return p

def add_code_block(doc, text, caption=None):
    if caption:
        p_c = doc.add_paragraph()
        p_c.paragraph_format.keep_with_next = True
        p_c.paragraph_format.space_before = Pt(3)
        p_c.paragraph_format.space_after = Pt(1)
        r_c = p_c.add_run(caption)
        r_c.font.name = 'Calibri'
        r_c.font.size = Pt(8.5)
        r_c.font.bold = True
        r_c.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
        
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl, color="CBD5E0", sz="4", val="single")
    
    trPr = tbl.rows[0]._element.get_or_add_trPr()
    trPr.append(parse_xml('<w:cantSplit ' + nsdecls('w') + '/>'))
    
    cell = tbl.rows[0].cells[0]
    cell.width = Inches(6.5)
    set_cell_background(cell, "F8FAFC")
    set_cell_margins(cell, top=45, bottom=45, left=70, right=70)
    
    p_in = cell.paragraphs[0]
    p_in.paragraph_format.space_after = Pt(0)
    p_in.paragraph_format.line_spacing = 1.0
    run = p_in.add_run(text)
    run.font.name = 'Consolas'
    run.font.size = Pt(7.5)
    run.font.color.rgb = RGBColor(0x1A, 0x20, 0x2C)
    
    p_gap = doc.add_paragraph()
    p_gap.paragraph_format.space_before = Pt(0)
    p_gap.paragraph_format.space_after = Pt(2)

def add_styled_table(doc, headers, data, col_widths=None):
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, color="CBD5E0", sz="4")
    
    hdr_row = table.rows[0]
    hdr_trPr = hdr_row._element.get_or_add_trPr()
    hdr_trPr.append(parse_xml('<w:cantSplit ' + nsdecls('w') + '/>'))
    hdr_trPr.append(parse_xml('<w:tblHeader ' + nsdecls('w') + '/>'))
    
    hdr_cells = hdr_row.cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], 'E2E8F0')
        set_cell_margins(hdr_cells[i], top=50, bottom=50, left=80, right=80)
        p = hdr_cells[i].paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        for run in p.runs:
            run.font.name = 'Calibri'
            run.font.bold = True
            run.font.size = Pt(8.5)
            run.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
            
    for r_idx, row_data in enumerate(data):
        row = table.rows[r_idx + 1]
        row_trPr = row._element.get_or_add_trPr()
        row_trPr.append(parse_xml('<w:cantSplit ' + nsdecls('w') + '/>'))
        
        row_cells = row.cells
        bg_color = 'F8FAFC' if r_idx % 2 == 1 else 'FFFFFF'
        for c_idx, cell_value in enumerate(row_data):
            row_cells[c_idx].text = str(cell_value)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=45, bottom=45, left=80, right=80)
            p = row_cells[c_idx].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.04
            for run in p.runs:
                run.font.name = 'Calibri'
                run.font.size = Pt(8.0)
                run.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
                
    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)
                
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(2)
    return table

def generate_updated_srs():
    doc = docx.Document()
    
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)
        
    # ==========================================
    # PAGE 1: COVER
    # ==========================================
    p_univ = doc.add_paragraph()
    p_univ.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_univ.paragraph_format.space_after = Pt(2)
    r = p_univ.add_run("GLA UNIVERSITY, GREATER NOIDA\n")
    r.font.name = 'Calibri'
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    r2 = p_univ.add_run("Greater Noida - 201310, Uttar Pradesh, India")
    r2.font.name = 'Calibri'
    r2.font.size = Pt(9.5)
    r2.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    p_gap = doc.add_paragraph()
    p_gap.paragraph_format.space_before = Pt(28)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(6)
    r_t = p_title.add_run("STROKE-LEVEL HANDWRITING SYNTHESIS:\nA VECTOR-OUTPUT GENERATIVE MODEL FOR COMPOSABLE HANDWRITTEN DOCUMENTS\n")
    r_t.font.name = 'Calibri'
    r_t.font.size = Pt(15)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    p_docname = doc.add_paragraph()
    p_docname.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_docname.paragraph_format.space_after = Pt(2)
    r_dn = p_docname.add_run("SOFTWARE REQUIREMENT SPECIFICATION")
    r_dn.font.name = 'Calibri'
    r_dn.font.size = Pt(12.5)
    r_dn.font.bold = True
    r_dn.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)

    p_ver = doc.add_paragraph()
    p_ver.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_ver.paragraph_format.space_after = Pt(20)
    r_v = p_ver.add_run("Version 1.1")
    r_v.font.name = 'Calibri'
    r_v.font.size = Pt(10.5)
    r_v.font.italic = True
    r_v.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    p_prog = doc.add_paragraph()
    p_prog.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_prog.paragraph_format.space_after = Pt(20)
    r_p = p_prog.add_run("BACHELOR OF TECHNOLOGY\nin\nCOMPUTER SCIENCE & ENGINEERING - AI")
    r_p.font.name = 'Calibri'
    r_p.font.size = Pt(10.5)
    r_p.font.bold = True
    r_p.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(3)
    r_s = p_sub.add_run("Submitted By\n")
    r_s.font.name = 'Calibri'
    r_s.font.size = Pt(9.5)
    r_s.font.color.rgb = RGBColor(0x71, 0x80, 0x96)
    r_name = p_sub.add_run("AGRIIM RASTOGI\n")
    r_name.font.name = 'Calibri'
    r_name.font.size = Pt(11.5)
    r_name.font.bold = True
    r_name.font.color.rgb = RGBColor(0x1A, 0x20, 0x2C)
    r_roll = p_sub.add_run("Roll No.: 22515500004   |   Section: ND")
    r_roll.font.name = 'Calibri'
    r_roll.font.size = Pt(9.5)
    r_roll.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    p_gap2 = doc.add_paragraph()
    p_gap2.paragraph_format.space_before = Pt(24)

    sup_table = doc.add_table(rows=1, cols=2)
    sup_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_l = sup_table.rows[0].cells[0]
    cell_r = sup_table.rows[0].cells[1]
    cell_l.width = Inches(3.25)
    cell_r.width = Inches(3.25)
    
    p_l = cell_l.paragraphs[0]
    p_l.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sl = p_l.add_run("Under the Supervision of\n")
    r_sl.font.name = 'Calibri'
    r_sl.font.size = Pt(9.0)
    r_sl.font.bold = True
    r_sl2 = p_l.add_run("Palash Sir / Vansh Sir\nFaculty Mentors, GLA University")
    r_sl2.font.name = 'Calibri'
    r_sl2.font.size = Pt(8.5)
    r_sl2.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    p_r = cell_r.paragraphs[0]
    p_r.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sr = p_r.add_run("Reviewed By\n")
    r_sr.font.name = 'Calibri'
    r_sr.font.size = Pt(9.0)
    r_sr.font.bold = True
    r_sr2 = p_r.add_run("Gautam Mukherjee\nGLA University, Greater Noida")
    r_sr2.font.name = 'Calibri'
    r_sr2.font.size = Pt(8.5)
    r_sr2.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    p_gap3 = doc.add_paragraph()
    p_gap3.paragraph_format.space_before = Pt(28)

    p_foot = doc.add_paragraph()
    p_foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_f = p_foot.add_run("DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING\nGLA UNIVERSITY, GREATER NOIDA - 201310\nSEPTEMBER, 2026")
    r_f.font.name = 'Calibri'
    r_f.font.size = Pt(9.0)
    r_f.font.bold = True
    r_f.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

    doc.add_page_break()

    # ==========================================
    # PAGE 2: REVISION, SECTION 1, SECTION 2
    # ==========================================
    add_header(doc, "Revision History", level=1)
    rev_headers = ["Version", "Date", "Change", "Reason"]
    rev_data = [
        ["1.0", "05-09-2026", "Initial baseline requirements specification.", "First institutional submission."],
        ["1.1", "11-09-2026", "Detailed system architecture, sprint date ranges, entity schemas, and ER diagram.", "Feedback from Faculty Reviewer (Gautam Mukherjee)."]
    ]
    add_styled_table(doc, rev_headers, rev_data, col_widths=[0.7, 1.0, 2.8, 2.0])

    add_header(doc, "1. Introduction", level=1)
    add_header(doc, "1.1 Purpose", level=2)
    add_body_p(doc, "Specifies software requirements for Stroke-Level Handwriting Synthesis v1.1, an offline deep-learning system generating human-like vector stroke trajectories from digital text. Defines functional, data, and quality constraints for verification.")

    add_header(doc, "1.2 Scope", level=2)
    add_body_p(doc, "Accepts English ASCII strings, sampling temperatures, and style seeds to synthesize resolution-independent SVG vector handwriting for offline document generation. Excludes OCR, scanned image transcription, and direct CNC pen-plotter motor controls.")

    add_header(doc, "1.3 Definitions", level=2)
    def_headers = ["Term", "Definition"]
    def_data = [
        ["Vector Stroke", "Time-ordered 2D coordinate sequence (x, y) between pen-down and pen-up events."],
        ["Online Handwriting", "Handwriting recorded as dynamic temporal trajectories (dx, dy, dt, pen_up), distinct from static 2D bitmaps."],
        ["CSR Transcription", "Computer Supported Recognition ground-truth text of human writing (including writer typos), distinct from printed prompts."],
        ["X-Height (h_x)", "Vertical height of lowercase letters without ascenders/descenders (e.g., 'x'), used as normalization unit."],
        ["Airborne Jump", "Pen-tip spatial transition trajectory above paper between consecutive strokes."]
    ]
    add_styled_table(doc, def_headers, def_data, col_widths=[1.6, 4.9])

    add_header(doc, "2. Overall Description", level=1)
    add_header(doc, "2.1 User Classes", level=2)
    user_headers = ["User Class", "Who They Are", "What They May Do"]
    user_data = [
        ["End-User / Creator", "Student, educator, or creator generating notes.", "Input text, tune temperature/style seeds, preview handwriting, export standalone SVG files."],
        ["ML Developer", "Practitioner maintaining or extending the pipeline.", "Execute IAM preprocessing, train sequence models, run test suites, benchmark position drift."]
    ]
    add_styled_table(doc, user_headers, user_data, col_widths=[1.5, 2.2, 2.8])

    add_header(doc, "2.2 Assumptions and Dependencies", level=2)
    add_body_p(doc, "Local execution on Python 3.13, PyTorch 2.6+, CUDA 12.6, RTX 3050 GPU (6 GB VRAM); CPU fallback supported.", bold_prefix="A1. Environment: ")
    add_body_p(doc, "Local offline availability of official IAM-OnDB archives in 'datas/raw/iam-ondb/' ('ascii/', 'lineStrokes/', 'original/').", bold_prefix="A2. Corpus: ")
    add_body_p(doc, "Zero external cloud dependencies; complete local data confidentiality.", bold_prefix="A3. Privacy: ")

    add_header(doc, "2.3 Constraints", level=2)
    add_body_p(doc, "IAM-OnDB is licensed strictly for non-commercial academic research.", bold_prefix="C1. Licensing: ")
    add_body_p(doc, "Batch sizes bounded by 6.0 GB GPU VRAM, requiring sequence compression and ragged tensor packaging.", bold_prefix="C2. Hardware: ")
    add_body_p(doc, "Whiteboard sensor records at ~100 Hz; requires arc-length resampling to eliminate velocity-dependent point clusters.", bold_prefix="C3. Digitizer: ")

    doc.add_page_break()

    # ==========================================
    # PAGE 3: SECTION 3 (DATA SOURCE & BASELINE)
    # ==========================================
    add_header(doc, "3. Data Source and Baseline", level=1)
    add_body_p(doc, "Empirical parameters and baseline benchmark scores computed directly from inspecting the IAM-OnDB corpus:")

    ds_headers = ["Item", "Value", "Required now?"]
    ds_data = [
        ["Dataset name", "IAM On-Line Handwriting Database (IAM-OnDB)", "Yes"],
        ["Source and URL", "University of Bern, FKI (https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)", "Yes"],
        ["Licence / terms of use", "Free for non-commercial academic research and educational evaluation", "Yes"],
        ["Number of rows and columns", "12,178 raw lines across 1,907 forms (3,731,052 points); ragged (N, 4) arrays [dx, dy, pen_up, dt]", "Yes"],
        ["Prediction target", "Next-step coordinate deltas (dx, dy, dt) and binary pen_up conditioned on text", "Yes"],
        ["Task type", "Multi-class discrete token classification (dx, dy, dt) and binary classification (pen_up)", "Yes"],
        ["Class balance", "Pen-down drawing steps: 92.1%; Pen-up airborne jumps: 7.9%", "Yes"],
        ["Primary evaluation metric", "Relative Position Drift Error (E_drift measured relative to line x-height)", "Yes"],
        ["Why that metric", "Pen strokes accumulate error via cumsum; per-step loss hides cumulative baseline drift and word collision", "Yes"],
        ["Level 0 baseline and score", "Uniform linear interpolation between word bounding boxes achieves E_drift = 1.28 (illegible)", "Yes"],
        ["Known data quality problems", "177 forms missing strokes, 2 line-count mismatches, ~0.25% non-monotonic timestamps (-746s), 10.8% missing writer ID", "Yes"],
        ["Level 1 baseline and score", "Graves (2013) 3-layer LSTM with bivariate Gaussian MDN achieving E_drift = 0.22", "Later"],
        ["Target for this release", "Dual-mode discrete sequence model targeting median E_drift <= 0.05 (p95 <= 0.08)", "Later"]
    ]
    add_styled_table(doc, ds_headers, ds_data, col_widths=[1.8, 3.8, 0.9])

    doc.add_page_break()

    # ==========================================
    # PAGE 4: SECTION 4 (SYSTEM ARCHITECTURE)
    # ==========================================
    add_header(doc, "4. System Architecture", level=1)
    add_body_p(doc, "The architecture decouples offline corpus preprocessing/training from real-time vector inference and rendering:")
    
    add_code_block(doc,
        "+-----------------------------------------------------------------------------------------+\n"
        "|                 OFFLINE DATA PREPARATION & NEURAL TRAINING PIPELINE                     |\n"
        "+-----------------------------------------------------------------------------------------+\n"
        "|  +--------------------+        +---------------------+        +----------------------+  |\n"
        "|  | Raw IAM Archives   | -----> | Ingest & Validate   | -----> | Geometric Prep       |  |\n"
        "|  | (XML, TXT, Original|        | Module: hws.iam     |        | Module: hws.preproc  |  |\n"
        "|  +--------------------+        +---------------------+        +----------------------+  |\n"
        "|                                                                          |              |\n"
        "|  +--------------------+        +---------------------+        +----------v-----------+  |\n"
        "|  | PyTorch Training   | <----- | Ragged HDF5 Storage | <----- | Dual-Mode Tokenizer  |  |\n"
        "|  | Module: hws.train  |        | train/val/test.h5   |        | Module: hws.token    |  |\n"
        "|  +--------------------+        +---------------------+        +----------------------+  |\n"
        "+-----------------------------------------------------------------------------------------+",
        caption="Figure 1A: Offline Data Preparation & Model Training Subsystem"
    )

    add_code_block(doc,
        "+-----------------------------------------------------------------------------------------+\n"
        "|                  REAL-TIME SYNTHESIS & VECTOR RENDERING PIPELINE                        |\n"
        "+-----------------------------------------------------------------------------------------+\n"
        "|  +--------------------+        +---------------------+        +----------------------+  |\n"
        "|  | User Input Text    | -----> | Character Embedding | -----> | Autoregressive Model |  |\n"
        "|  | Text & Style Seed  |        | Module: hws.alphabet|        | Cross-Attention Core |  |\n"
        "|  +--------------------+        +---------------------+        +----------------------+  |\n"
        "|                                                                          |              |\n"
        "|  +--------------------+        +---------------------+        +----------v-----------+  |\n"
        "|  | SVG Document Export| <----- | Vector Path Layout  | <----- | Stroke Reconstruction|  |\n"
        "|  | File: document.svg |        | Module: hws.render  |        | cumsum & +y Inversion|  |\n"
        "|  +--------------------+        +---------------------+        +----------------------+  |\n"
        "+-----------------------------------------------------------------------------------------+",
        caption="Figure 1B: Real-Time Text-to-Stroke Inference & Vector Rendering Subsystem"
    )

    add_header(doc, "4.1 Core System Components", level=2)
    arch_headers = ["Component & Module", "Input", "Output", "Architectural Responsibility"]
    arch_data = [
        ["Corpus Ingestion\n(hws.iam)", "Raw IAM XML, CSR text, original XML.", "RawLine objects (N, 4).", "Parses XML captures, validates text-stroke parity, flips y-axis (+y=up), extracts writer IDs."],
        ["Geometry Preprocessing\n(hws.preprocess)", "Raw points (N, 4) from hws.iam.", "Resampled offsets (dx, dy, dt, pen_up).", "Executes arc-length resampling (1.84x compression), computes deltas, normalizes by x-height."],
        ["Character Vocabulary\n(hws.alphabet)", "Latin ASCII character strings.", "Integer token sequence (1D).", "Maintains immutable 81-char vocab with <pad> and <unk> tokens, preventing index-zero collision."],
        ["Dual-Mode Discretizer\n(hws.tokenizer)", "Continuous offsets & pen_up.", "Discrete tokens (0 to N_bins - 1).", "Quantizes moves into linear bins and heavy-tailed airborne jumps into uniform asinh bins."],
        ["Neural Sequence Model\n(hws.model)", "Char tokens & prior stroke tokens.", "Softmax logit probabilities.", "Autoregressively predicts next-step stroke token distributions conditioned via cross-attention."],
        ["Coordinate Reconstruction\n(hws.preprocess)", "Predicted token indices.", "Absolute Cartesian arrays (x, y).", "De-quantizes tokens, computes cumulative sum (cumsum), and arranges multi-line baselines."],
        ["Vector Rendering Engine\n(hws.render)", "Cartesian coordinate arrays.", "Valid SVG XML documents.", "Single source of truth for converting points into SVG paths ('M/L'), reversing y to SVG standard."]
    ]
    add_styled_table(doc, arch_headers, arch_data, col_widths=[1.4, 1.3, 1.3, 2.5])

    doc.add_page_break()

    # ==========================================
    # PAGE 5: SECTION 5 & 6 (REQUIREMENTS)
    # ==========================================
    add_header(doc, "5. Functional Requirements", level=1)
    add_header(doc, "5.1 Data Ingestion and Preprocessing", level=2)
    add_body_p(doc, "Parse IAM stroke XML files into (N, 4) arrays of x, y, pen_up, and elapsed time t.", bold_prefix="FR-1.1  ")
    add_body_p(doc, "Reject input forms with mismatched stroke file counts and CSR transcription lines, returning count error.", bold_prefix="FR-1.2  ")
    add_body_p(doc, "Invert whiteboard sensor y-coordinates at parse time so positive y increases upward across internal tensors.", bold_prefix="FR-1.3  ")
    add_body_p(doc, "Resample stroke trajectories at uniform spatial arc-length intervals determined by step factor.", bold_prefix="FR-1.4  ")
    add_body_p(doc, "Normalize horizontal and vertical coordinate displacements by dividing by estimated line x-height.", bold_prefix="FR-1.5  ")

    add_header(doc, "5.2 Model-Backed Generation", level=2)
    add_body_p(doc, "Accept strings within 81-character vocabulary; reject unsupported characters with positional error message.", bold_prefix="FR-2.1  ")
    add_body_p(doc, "Return time-ordered sequence of coordinate offsets (dx, dy, dt, pen_up) for each validated text prompt.", bold_prefix="FR-2.2  ")
    add_body_p(doc, "Allow adjustment of sampling temperature between 0.1 and 2.0 to control stroke curvature and legibility.", bold_prefix="FR-2.3  ")
    add_body_p(doc, "Fall back to rendering pre-tokenized Level 0 baseline glyphs with warning banner on generation failure.", bold_prefix="FR-2.4  ")
    add_body_p(doc, "Support selectable writer style seeds to modulate handwriting slant, speed, and ligature spacing.", bold_prefix="FR-2.5  ")

    add_header(doc, "6. Non-Functional Requirements", level=1)
    add_body_p(doc, "Synthesize a 30-character line within 1.5 seconds on an NVIDIA RTX 3050 Laptop GPU.", bold_prefix="NFR-1 (Latency): ")
    add_body_p(doc, "Execute entirely on local host without transmitting text prompts or strokes over network.", bold_prefix="NFR-2 (Privacy): ")
    add_body_p(doc, "Log preprocessing runs to timestamped JSON recording skipped files, drift percentiles, and vocabulary sizes.", bold_prefix="NFR-3 (Auditability): ")
    add_body_p(doc, "Compile all 12,178 corpus lines into HDF5 in < 180 seconds consuming < 2.0 GB system RAM.", bold_prefix="NFR-4 (Resource): ")

    doc.add_page_break()

    # ==========================================
    # PAGE 6: SECTION 7 (BUILD PLAN)
    # ==========================================
    add_header(doc, "7. Build Plan", level=1)
    add_header(doc, "7.1 Minimum Viable Product", level=2)
    add_body_p(doc, "The MVP represents the smallest self-contained version capable of demonstrating end-to-end data processing and vector stroke synthesis:")
    mvp_headers = ["FR / NFR ID", "Requirement (short)", "Why this one is in the MVP"]
    mvp_data = [
        ["FR-1.1", "Parse raw stroke XML files", "Mandatory data source; without raw trajectories no training data exists."],
        ["FR-1.2", "Reject mismatched form counts", "Prevents misaligned text-stroke pairs from corrupting model datasets."],
        ["FR-1.4", "Spatial arc-length resampling", "Compresses sequences 1.84x, making neural training feasible within 6 GB VRAM."],
        ["FR-2.1", "Validate text against vocabulary", "Guarantees sequence model receives only valid character token embeddings."],
        ["FR-2.2", "Return vector stroke sequences", "Core capability required to synthesize visible handwriting trajectories."],
        ["NFR-1", "Line generation under 1.5s", "Maintains responsive interactive user experience during note synthesis."],
        ["NFR-2", "100% offline local execution", "Guarantees complete document confidentiality without cloud APIs."]
    ]
    add_styled_table(doc, mvp_headers, mvp_data, col_widths=[1.1, 2.5, 2.9])
    add_body_p(doc, "Few-shot style conditioning on user-uploaded handwriting samples (deferred to Sprint III).", bold_prefix="Deliberately not in MVP: ")

    add_header(doc, "7.2 Build Order & Sprint Milestones", level=2)
    add_body_p(doc, "Project implementation schedule across sequential development sprints with concrete milestones and dates:")
    stage_ref_headers = ["Sprint Stage", "Date Range", "Target Scope", "Milestone Deliverable & Evidence"]
    stage_ref_data = [
        ["Sprint I", "01 Sep 2026 - 27 Sep 2026", "FR-1.1 to FR-1.5", "Data ingestion, arc-length resampling; inspect_data.py displays SVG samples."],
        ["Sprint II", "28 Sep 2026 - 17 Oct 2026", "FR-2.1 to FR-2.3, NFR-1", "Autoregressive model & CLI; synthesizes single-line SVG in < 1.5 seconds."],
        ["Sprint III", "18 Oct 2026 - 07 Nov 2026", "FR-2.4, FR-2.5, NFR-2", "Style conditioning & multi-line layout; generates multi-line documents."],
        ["Testing", "08 Nov 2026 - 17 Nov 2026", "All FRs, NFR-3, NFR-4", "58 unit tests pass; zero-regression log; reconstruction drift p95 < 0.08."],
        ["Deployment", "18 Nov 2026 - 30 Nov 2026", "All FRs and NFRs", "Packaged desktop distribution, user manual, and verified clean terminal run."],
        ["Final Viva", "01 Dec 2026 - 10 Dec 2026", "Full Release", "Institutional project demonstration, final report submission, and oral viva."]
    ]
    add_styled_table(doc, stage_ref_headers, stage_ref_data, col_widths=[1.0, 1.6, 1.3, 2.6])

    doc.add_page_break()

    # ==========================================
    # PAGE 7: SECTION 8 (DATA REQUIREMENTS: ENTITIES 1, 2, 3)
    # ==========================================
    add_header(doc, "8. Data Requirements", level=1)
    add_header(doc, "8.1 Entities", level=2)
    add_body_p(doc, "Five persistent entities model the data lifecycle from text input to vector rendering:")

    # Entity 1
    add_body_p(doc, "Captures input digital text, temperature, and line width formatting.", bold_prefix="Entity 1: TextPrompt -- ")
    p1_headers = ["Field Name", "Data Type", "Constraints & Validation Rules"]
    p1_data = [
        ["id", "UUID", "Primary key; uniquely identifies each request."],
        ["raw_content", "Text", "Raw input string; length 1 to 2,000 characters; not null."],
        ["cleaned_content", "Text", "Sanitized ASCII string restricted to 81-character vocabulary."],
        ["sampling_temp", "Float", "Sampling temperature in range [0.1, 2.0]; default 1.0."],
        ["target_width_mm", "Float", "Target horizontal line width in mm; positive; default 180.0."],
        ["created_at", "Timestamp", "System-generated UTC creation timestamp; not null."]
    ]
    add_styled_table(doc, p1_headers, p1_data, col_widths=[1.4, 1.1, 4.0])

    # Entity 2
    add_body_p(doc, "Represents individual handwriting stylistic characteristics and bias embeddings.", bold_prefix="Entity 2: WriterProfile -- ")
    p2_headers = ["Field Name", "Data Type", "Constraints & Validation Rules"]
    p2_data = [
        ["id", "UUID", "Primary key; unique internal profile identifier."],
        ["writer_id", "Integer", "Unique IAM volunteer writer ID; unique index; nullable for anonymous."],
        ["style_seed", "Integer", "Seed initializing autoregressive style embedding; non-negative."],
        ["mean_slant_deg", "Float", "Mean baseline slant angle in degrees; range [-45.0, +45.0]."],
        ["mean_velocity", "Float", "Mean pen-tip speed in sensor units/sec; positive float."],
        ["style_label", "String(64)", "Descriptive tag (e.g., 'Cursive-Slanted', 'Upright-Print'); not null."]
    ]
    add_styled_table(doc, p2_headers, p2_data, col_widths=[1.4, 1.1, 4.0])

    # Entity 3
    add_body_p(doc, "Continuous, time-ordered sequence of pen coordinates for one text line.", bold_prefix="Entity 3: StrokeSequence -- ")
    p3_headers = ["Field Name", "Data Type", "Constraints & Validation Rules"]
    p3_data = [
        ["id", "UUID", "Primary key; uniquely identifies each stroke trajectory."],
        ["prompt_id", "UUID", "FK referencing TextPrompt.id; nullable for raw IAM training lines."],
        ["writer_id", "Integer", "FK referencing WriterProfile.writer_id; nullable for anonymous lines."],
        ["form_id", "String(16)", "Parent IAM form ID (e.g., 'a01-000u'); nullable for synthetic lines."],
        ["line_index", "Integer", "1-based line index within document or form; positive integer."],
        ["raw_text", "Text", "CSR transcription string corresponding to trajectory; non-empty."],
        ["point_count", "Integer", "Total coordinate points in trajectory; must be >= 10."],
        ["stroke_count", "Integer", "Total pen-down segments (number of pen lifts + 1); positive integer."],
        ["duration_sec", "Float", "Total wall-clock duration of stroke writing in seconds; non-negative."],
        ["created_at", "Timestamp", "System-generated UTC creation timestamp; not null."]
    ]
    add_styled_table(doc, p3_headers, p3_data, col_widths=[1.4, 1.1, 4.0])

    doc.add_page_break()

    # ==========================================
    # PAGE 8: ENTITIES 4, 5 & RELATIONSHIPS
    # ==========================================
    # Entity 4
    add_body_p(doc, "Discrete integer token representation of stroke trajectory used by neural model.", bold_prefix="Entity 4: QuantizedTokenSequence -- ")
    p4_headers = ["Field Name", "Data Type", "Constraints & Validation Rules"]
    p4_data = [
        ["id", "UUID", "Primary key; uniquely identifies each tokenized sequence."],
        ["stroke_seq_id", "UUID", "FK referencing StrokeSequence.id; unique constraint (strict 1:1)."],
        ["dx_tokens", "Binary", "Contiguous int16 discrete token IDs for horizontal displacements."],
        ["dy_tokens", "Binary", "Contiguous int16 discrete token IDs for vertical displacements."],
        ["dt_tokens", "Binary", "Contiguous int16 discrete token IDs for time deltas."],
        ["pen_up_flags", "Binary", "Contiguous uint8 binary flags indicating pen-lift state after each step."],
        ["token_count", "Integer", "Total token count; must exactly equal parent StrokeSequence.point_count."]
    ]
    add_styled_table(doc, p4_headers, p4_data, col_widths=[1.4, 1.1, 4.0])

    # Entity 5
    add_body_p(doc, "Finalized resolution-independent vector graphic file saved to disk.", bold_prefix="Entity 5: ExportedSVGDocument -- ")
    p5_headers = ["Field Name", "Data Type", "Constraints & Validation Rules"]
    p5_data = [
        ["id", "UUID", "Primary key; uniquely identifies each exported document."],
        ["prompt_id", "UUID", "FK referencing TextPrompt.id; not null."],
        ["file_path", "String(256)", "Absolute local filesystem path where SVG is saved; unique index."],
        ["viewbox_width", "Float", "SVG viewBox horizontal width in pixels; positive float."],
        ["viewbox_height", "Float", "SVG viewBox vertical height in pixels; positive float."],
        ["line_count", "Integer", "Total handwritten lines composed in document; positive integer."],
        ["file_size_bytes", "Integer", "Size of exported SVG file on disk in bytes; positive integer."],
        ["exported_at", "Timestamp", "System-generated UTC export timestamp; not null."]
    ]
    add_styled_table(doc, p5_headers, p5_data, col_widths=[1.4, 1.1, 4.0])

    add_header(doc, "8.2 Entity Relationships", level=2)
    add_body_p(doc, "One input prompt can generate multiple stroke lines or style variations (1:N).", bold_prefix="1. TextPrompt to StrokeSequence (1 : N): ")
    add_body_p(doc, "A single writer profile conditions multiple stroke sequences across prompts (1:N).", bold_prefix="2. WriterProfile to StrokeSequence (1 : N): ")
    add_body_p(doc, "Each continuous trajectory maps to exactly one discrete tokenized sequence (1:1).", bold_prefix="3. StrokeSequence to QuantizedTokenSequence (1 : 1): ")
    add_body_p(doc, "A user prompt can be exported into multiple document iterations or styles (1:N).", bold_prefix="4. TextPrompt to ExportedSVGDocument (1 : N): ")
    add_body_p(doc, "An exported SVG document aggregates one or more line stroke sequences (N : 1).", bold_prefix="5. StrokeSequence to ExportedSVGDocument (N : 1): ")

    doc.add_page_break()

    # ==========================================
    # PAGE 9: COMPLETE ER DIAGRAM
    # ==========================================
    add_header(doc, "8.3 Complete Entity-Relationship (ER) Diagram", level=2)
    add_body_p(doc, "Visual architectural layout, primary keys (PK), foreign keys (FK), and cardinalities across all entities:")
    
    add_code_block(doc,
        "    +-------------------------------------------------------------------------+\n"
        "    |                               TextPrompt                                |\n"
        "    |-------------------------------------------------------------------------|\n"
        "    | PK  id               : UUID                                             |\n"
        "    |     raw_content      : Text                                             |\n"
        "    |     cleaned_content  : Text                                             |\n"
        "    |     sampling_temp    : Float                                            |\n"
        "    |     target_width_mm  : Float                                            |\n"
        "    |     created_at       : Timestamp                                        |\n"
        "    +-------------------------------------------------------------------------+\n"
        "           | 1                                                 | 1\n"
        "           |                                                   |\n"
        "           | generates                                         | produces\n"
        "           |                                                   |\n"
        "           v N                                                 v N\n"
        "    +-------------------------------+           +-----------------------------+\n"
        "    |        StrokeSequence         |           |     ExportedSVGDocument     |\n"
        "    |-------------------------------|           |-----------------------------|\n"
        "    | PK  id           : UUID       |     N   1 | PK  id             : UUID   |\n"
        "    | FK  prompt_id    : UUID       | >-------< | FK  prompt_id      : UUID   |\n"
        "    | FK  writer_id    : Integer    |  renders  |     file_path      : String |\n"
        "    |     form_id      : String(16) |   into    |     viewbox_width  : Float  |\n"
        "    |     line_index   : Integer    |           |     viewbox_height : Float  |\n"
        "    |     raw_text     : Text       |           |     line_count     : Integer|\n"
        "    |     point_count  : Integer    |           |     file_size_bytes: Integer|\n"
        "    |     stroke_count : Integer    |           |     exported_at    : Timest.|\n"
        "    |     duration_sec : Float      |           +-----------------------------+\n"
        "    |     created_at   : Timestamp  |\n"
        "    +-------------------------------+\n"
        "         ^ N                  | 1\n"
        "         |                    |\n"
        "         | conditions         | tokenized into\n"
        "         |                    |\n"
        "         | 1                  v 1\n"
        "    +--------------------+  +--------------------------------+\n"
        "    |   WriterProfile    |  |     QuantizedTokenSequence     |\n"
        "    |--------------------|  |--------------------------------|\n"
        "    | PK  id             |  | PK  id            : UUID       |\n"
        "    | UQ  writer_id      |  | FK  stroke_seq_id : UUID (UQ)  |\n"
        "    |     style_seed     |  |     dx_tokens     : Binary     |\n"
        "    |     mean_slant_deg |  |     dy_tokens     : Binary     |\n"
        "    |     mean_velocity  |  |     dt_tokens     : Binary     |\n"
        "    |     style_label    |  |     pen_up_flags  : Binary     |\n"
        "    +--------------------+  |     token_count   : Integer    |\n"
        "                            +--------------------------------+",
        caption="Figure 2: Complete Entity-Relationship (ER) Architectural Diagram"
    )

    doc.add_page_break()

    # ==========================================
    # PAGE 10: SECTION 9, 10, 11 & SIGN-OFF
    # ==========================================
    add_header(doc, "9. Design Decisions and Alternatives Considered", level=1)
    dec_headers = ["#", "Decision", "Alternative rejected", "Reason"]
    dec_data = [
        ["D1", "Discrete dual-mode tokenization (linear moves + asinh jumps).", "Continuous Mixture Density Networks (MDNs).", "MDNs suffer from variance collapse (sigma -> 0) causing NaN gradients and mode collapse. Discretization reformulates generation as Cross-Entropy, achieving zero NaN crashes and median drift of 0.032 x-height."],
        ["D2", "Ragged 1D HDF5 storage with slice offset indices.", "Fixed-length zero-padded 2D arrays.", "Padding sequences to maximum length (857 points) wastes ~40% disk space and bakes a hard limit. Ragged storage saves 22 MB per split and enables dynamic batch collation."],
        ["D3", "Direct SVG vector path export ('<path d=...>')", "Raster PNG image generation via GANs or Diffusion.", "Raster generation fixes resolution and causes pixelation. SVG paths are resolution-independent, styleable in CSS, and map directly to CNC pen plotters."]
    ]
    add_styled_table(doc, dec_headers, dec_data, col_widths=[0.4, 2.0, 1.8, 2.3])

    add_header(doc, "10. Acceptance Criteria", level=1)
    add_body_p(doc, "CLI accepts 'Calligrapher AI' and outputs valid, legible SVG stroke file within 1.5s.", bold_prefix="AC-1 (Live Generation): ")
    add_body_p(doc, "Running 'prepare_data.py' compiles 12,178 lines and outputs 'train.h5', 'val.h5', 'test.h5' without errors.", bold_prefix="AC-2 (Dataset Pipeline): ")
    add_body_p(doc, "Input with unsupported char ('Hello ~ World') halts immediately with clear positional error message.", bold_prefix="AC-3 (Error Handling): ")
    add_body_p(doc, "Varying temperature (0.5 to 1.5) modulates stroke curvature from stiff lettering to expressive cursive.", bold_prefix="AC-4 (Style Modulation): ")

    add_header(doc, "11. Out of Scope for This Release", level=1)
    add_body_p(doc, "Live touch drawing canvas for user stroke capture; release focuses strictly on text-to-stroke.", bold_prefix="1. Interactive Stylus Input: ")
    add_body_p(doc, "Multilingual non-Latin script synthesis; IAM training corpus is strictly English Latin.", bold_prefix="2. Non-Latin Language Scripts: ")
    add_body_p(doc, "Few-shot online biometric signature verification; identity authentication is a separate problem.", bold_prefix="3. Biometric Verification: ")

    p_sign_hdr = doc.add_paragraph()
    p_sign_hdr.paragraph_format.space_before = Pt(14)
    p_sign_hdr.paragraph_format.space_after = Pt(4)
    r_sh = p_sign_hdr.add_run("Institutional Evaluation & Faculty Sign-Off Block")
    r_sh.font.name = 'Calibri'
    r_sh.font.size = Pt(10)
    r_sh.font.bold = True
    r_sh.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)

    sign_table = doc.add_table(rows=2, cols=3)
    sign_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(sign_table, color="CBD5E0", sz="4")
    
    sign_headers = ["Candidate Details", "Faculty Supervision", "Project Review Committee"]
    for c_idx, h_text in enumerate(sign_headers):
        c = sign_table.rows[0].cells[c_idx]
        c.text = h_text
        set_cell_background(c, 'E2E8F0')
        set_cell_margins(c, top=50, bottom=50, left=70, right=70)
        p = c.paragraphs[0]
        for r in p.runs:
            r.font.name = 'Calibri'
            r.font.bold = True
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)

    sign_contents = [
        "Name: Agriim Rastogi\nRoll No.: 22515500004\nSection: ND\n\nSignature: ________________\nDate: 11-09-2026",
        "Palash Sir / Vansh Sir\nFaculty Mentors\nDepartment of CSE\n\nSignature: ________________\nDate: 11-09-2026",
        "Gautam Mukherjee\nProject Reviewer\nGLA University, Greater Noida\n\nSignature: ________________\nDate: 11-09-2026"
    ]
    for c_idx, content in enumerate(sign_contents):
        c = sign_table.rows[1].cells[c_idx]
        c.text = content
        set_cell_background(c, 'FFFFFF')
        set_cell_margins(c, top=50, bottom=50, left=70, right=70)
        p = c.paragraphs[0]
        p.paragraph_format.line_spacing = 1.08
        for r in p.runs:
            r.font.name = 'Calibri'
            r.font.size = Pt(8.0)
            r.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

    for row in sign_table.rows:
        row_trPr = row._element.get_or_add_trPr()
        row_trPr.append(parse_xml('<w:cantSplit ' + nsdecls('w') + '/>'))
        for c_idx in range(3):
            row.cells[c_idx].width = Inches(2.16)

    out_docx = r"c:\Users\basub\OneDrive\Desktop\AR-Handwriting Synthesis\SRS_v1.0_22515500004_HandwritingSynthesis.docx"
    doc.save(out_docx)
    print(f"Successfully generated concise official DOCX: {out_docx}")
    return out_docx

if __name__ == '__main__':
    generate_updated_srs()
