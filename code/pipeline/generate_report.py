"""
MIP Screening Pipeline — Feature 6: HTML Report Auto-Generation
================================================================
Reads all stage result files and generates a self-contained HTML report
with embedded images (base64), inline CSS, and zebra-striped tables.

Usage:
    python generate_report.py
"""

import base64
import csv
import datetime
import glob
import json
import pathlib

from .config import (
    TEMPLATE_NAME,
    TEMPLATE_SMILES,
    OUTPUT_DIR,
    OUTPUT_DIRS,
    MONOMER_LIBRARY,
    SOLVENTS,
    INTERFERENT_LIBRARY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: pathlib.Path) -> dict | list | None:
    """Load a JSON file; return None if it does not exist."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: pathlib.Path) -> list[dict] | None:
    """Load a CSV file as a list of dicts; return None if missing."""
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _embed_image(path: pathlib.Path) -> str | None:
    """Return a base64-encoded data-URI string for an image, or None."""
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}.get(suffix, "image/png")
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _not_executed() -> str:
    """Placeholder HTML for a stage that has not been run."""
    return '<p style="color:#888; font-style:italic;">미실행 (Not executed)</p>'


def _table(headers: list[str], rows: list[list], caption: str = "") -> str:
    """Build an HTML table with zebra striping."""
    parts = ['<table>']
    if caption:
        parts.append(f'<caption style="caption-side:top; font-weight:bold; '
                      f'margin-bottom:6px;">{caption}</caption>')
    parts.append("<thead><tr>")
    for h in headers:
        parts.append(f"<th>{h}</th>")
    parts.append("</tr></thead><tbody>")
    for i, row in enumerate(rows):
        bg = "#f9f9f9" if i % 2 == 0 else "#ffffff"
        parts.append(f'<tr style="background:{bg};">')
        for cell in row:
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _section(title: str, body: str) -> str:
    return f'<div class="section"><h2>{title}</h2>\n{body}\n</div>\n'


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

INLINE_CSS = """
body {
    background: #ffffff;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: #222;
    max-width: 1100px;
    margin: 0 auto;
    padding: 30px 40px;
}
h1 { border-bottom: 3px solid #2c3e50; padding-bottom: 8px; }
h2 { color: #2c3e50; margin-top: 32px; }
.section { margin-bottom: 28px; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0 18px 0;
    font-size: 14px;
}
th, td {
    border: 1px solid #ccc;
    padding: 7px 12px;
    text-align: center;
}
th { background: #2c3e50; color: #fff; }
caption { text-align: left; font-size: 15px; }
img.plot { max-width: 700px; width: 100%; margin: 12px 0; border: 1px solid #ddd; }
.meta { color: #555; font-size: 14px; margin: 4px 0; }
.recommend-box {
    background: #eaf6ff;
    border-left: 4px solid #2980b9;
    padding: 14px 18px;
    margin: 12px 0;
}
.gallery { display: flex; flex-wrap: wrap; gap: 14px; }
.gallery img { max-width: 320px; border: 1px solid #ddd; }
"""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_header(now: datetime.datetime) -> str:
    """Section 1 — Header with template info and config summary."""
    lines = [
        f"<h1>MIP Screening Report — {TEMPLATE_NAME}</h1>",
        f'<p class="meta"><b>Template SMILES:</b> <code>{TEMPLATE_SMILES}</code></p>',
        f'<p class="meta"><b>Execution date:</b> {now.strftime("%Y-%m-%d %H:%M:%S")}</p>',
        '<h3>Configuration Summary</h3>',
        "<ul>",
        f"<li><b>Monomers ({len(MONOMER_LIBRARY)}):</b> {', '.join(MONOMER_LIBRARY.keys())}</li>",
        f"<li><b>Solvents:</b> {', '.join(SOLVENTS.keys())}</li>",
        f"<li><b>Interferents:</b> {', '.join(INTERFERENT_LIBRARY.keys())}</li>",
        "</ul>",
    ]
    return "\n".join(lines)


def _build_stage1(out: pathlib.Path) -> str:
    """Section 2 — xTB binding energy table."""
    all_data = _load_json(out / "stage1" / "stage1_all.json")
    top_data = _load_json(out / "stage1" / "stage1_top.json")
    if all_data is None and top_data is None:
        return _section("Stage 1 — xTB Fast Screening", _not_executed())

    body_parts = []

    # all results table
    if isinstance(all_data, list):
        headers = ["Rank", "Monomer", "xTB Binding Energy (kcal/mol)"]
        rows = []
        for i, entry in enumerate(all_data, 1):
            if isinstance(entry, dict):
                name = entry.get("monomer", entry.get("name", ""))
                be = entry.get("dE_kcal", entry.get("binding_energy", entry.get("dE", "")))
                if isinstance(be, (int, float)):
                    be = f"{be:+.3f}"
                rows.append([i, name, be])
        body_parts.append(_table(headers, rows, "All monomer xTB results"))
    elif isinstance(all_data, dict):
        headers = ["Rank", "Monomer", "xTB Binding Energy (kcal/mol)"]
        rows = []
        sorted_items = sorted(all_data.items(),
                              key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
        for i, (name, be) in enumerate(sorted_items, 1):
            if isinstance(be, (int, float)):
                be = f"{be:+.3f}"
            rows.append([i, name, be])
        body_parts.append(_table(headers, rows, "All monomer xTB results"))

    # top monomers
    if top_data is not None:
        if isinstance(top_data, list):
            if top_data and isinstance(top_data[0], str):
                body_parts.append(
                    f"<p><b>Top monomers passed to Stage 2:</b> {', '.join(top_data)}</p>")
            elif top_data and isinstance(top_data[0], dict):
                headers = ["Rank", "Monomer", "xTB Binding Energy (kcal/mol)"]
                rows = []
                for i, entry in enumerate(top_data, 1):
                    name = entry.get("monomer", entry.get("name", ""))
                    be = entry.get("dE_kcal", entry.get("binding_energy", entry.get("dE", "")))
                    if isinstance(be, (int, float)):
                        be = f"{be:+.3f}"
                    rows.append([i, name, be])
                body_parts.append(_table(headers, rows, "Top monomers for Stage 2"))

    return _section("Stage 1 — xTB Fast Screening", "\n".join(body_parts))


def _build_stage2(out: pathlib.Path) -> str:
    """Section 3 — DFT binding energy table by solvent with BSSE."""
    dft_data = _load_json(out / "stage2" / "stage2_dft.json")
    if dft_data is None:
        return _section("Stage 2 — DFT Binding Energy", _not_executed())

    body_parts = []

    if isinstance(dft_data, dict):
        # Expected: {monomer: {solvent: {dE, bsse_dE, ...}}}
        solvent_names = sorted({s for solvs in dft_data.values()
                                for s in (solvs.keys() if isinstance(solvs, dict) else [])})
        headers = ["Monomer"] + [f"{s} dE" for s in solvent_names] + \
                  [f"{s} BSSE-dE" for s in solvent_names]
        rows = []
        for mono, solvs in sorted(dft_data.items()):
            if not isinstance(solvs, dict):
                continue
            row = [mono]
            for s in solvent_names:
                v = solvs.get(s, {})
                de = v.get("raw_dE", v.get("dE", "N/A"))
                row.append(f"{de:+.3f}" if isinstance(de, (int, float)) else str(de))
            for s in solvent_names:
                v = solvs.get(s, {})
                bsse = v.get("bsse_dE", "N/A")
                row.append(f"{bsse:+.3f}" if isinstance(bsse, (int, float)) else str(bsse))
            rows.append(row)
        body_parts.append(_table(headers, rows, "DFT Binding Energies (kcal/mol)"))
    elif isinstance(dft_data, list):
        # Flat list format
        if dft_data and isinstance(dft_data[0], dict):
            headers = list(dft_data[0].keys())
            rows = []
            for entry in dft_data:
                row = []
                for h in headers:
                    v = entry.get(h, "")
                    if isinstance(v, float):
                        v = f"{v:+.3f}"
                    row.append(str(v))
                rows.append(row)
            body_parts.append(_table(headers, rows, "DFT Binding Energies"))

    return _section("Stage 2 — DFT Binding Energy", "\n".join(body_parts))


def _build_stage3(out: pathlib.Path) -> str:
    """Section 4 — Selectivity table + scatter + bar chart."""
    csv_rows = _load_csv(out / "stage3" / "stage3_selectivity.csv")
    scatter_uri = _embed_image(out / "stage3" / "stage3_scatter.png")
    bar_uri = _embed_image(out / "stage3" / "stage3_bar.png")

    if csv_rows is None and scatter_uri is None and bar_uri is None:
        return _section("Stage 3 — Selectivity Analysis", _not_executed())

    body_parts = []

    if csv_rows:
        headers = list(csv_rows[0].keys())
        rows = []
        for entry in csv_rows:
            row = []
            for h in headers:
                v = entry.get(h, "")
                # Try formatting numeric strings
                try:
                    fv = float(v)
                    v = f"{fv:+.3f}" if "energy" in h.lower() or "dE" in h or "log" in h.lower() else f"{fv:.3f}"
                except (ValueError, TypeError):
                    pass
                row.append(str(v))
            rows.append(row)
        body_parts.append(_table(headers, rows, "Selectivity Results"))

    if scatter_uri:
        body_parts.append(
            f'<p><b>Scatter Plot:</b></p><img class="plot" src="{scatter_uri}" '
            f'alt="Stage 3 Scatter Plot"/>')
    if bar_uri:
        body_parts.append(
            f'<p><b>Bar Chart:</b></p><img class="plot" src="{bar_uri}" '
            f'alt="Stage 3 Bar Chart"/>')

    return _section("Stage 3 — Selectivity Analysis", "\n".join(body_parts))


def _build_stage4(out: pathlib.Path) -> str:
    """Section 5 — MD results: EBN / H-bond table + RDF graph."""
    md_data = _load_json(out / "stage5" / "stage5_md.json")
    rdf_uri = _embed_image(out / "stage5" / "stage5_rdf.png")

    if md_data is None and rdf_uri is None:
        return _section("Stage 4 — MD Validation", _not_executed())

    body_parts = []

    if isinstance(md_data, list) and md_data:
        # Expected keys: monomer, EBN, n_hbonds_stable, ratio, ...
        sample_keys = list(md_data[0].keys())
        display_headers = []
        for k in sample_keys:
            display_headers.append(k.replace("_", " ").title())
        rows = []
        for entry in md_data:
            row = []
            for k in sample_keys:
                v = entry.get(k, "")
                if isinstance(v, float):
                    v = f"{v:.3f}"
                row.append(str(v))
            rows.append(row)
        body_parts.append(_table(display_headers, rows, "MD Simulation Results"))
    elif isinstance(md_data, dict):
        headers = ["Monomer", "EBN", "Stable H-bonds", "Ratio"]
        rows = []
        for mono, info in md_data.items():
            if isinstance(info, dict):
                ebn = info.get("EBN", "N/A")
                hb = info.get("n_hbonds_mean", info.get("n_hbonds_stable", "N/A"))
                ratio = info.get("ratio", "N/A")
                if isinstance(ebn, float):
                    ebn = f"{ebn:.3f}"
                rows.append([mono, ebn, hb, ratio])
        if rows:
            body_parts.append(_table(headers, rows, "MD Simulation Results"))

    if rdf_uri:
        body_parts.append(
            f'<p><b>Radial Distribution Function (RDF):</b></p>'
            f'<img class="plot" src="{rdf_uri}" alt="RDF Plot"/>')

    # XVG inline plots for each monomer's MD trajectory
    stage4_dir = out / "stage5"
    if stage4_dir.is_dir():
        xvg_plots = []
        for mono_dir in sorted(stage4_dir.iterdir()):
            if not mono_dir.is_dir():
                continue
            m_name = mono_dir.name
            for xvg_name, title, xlabel, ylabel in [
                ("md.edr", f"{m_name} Energy", "Time (ps)", "Energy (kJ/mol)"),
            ]:
                # Try RMSD xvg if available
                rmsd_xvg = mono_dir / "rmsd.xvg"
                if rmsd_xvg.exists():
                    xvg_plots.append(_xvg_to_inline_plot(
                        rmsd_xvg, f"{m_name} RMSD", "Time (ps)", "RMSD (nm)"))
        if xvg_plots:
            body_parts.append("<h3>MD Trajectory Plots</h3>")
            body_parts.extend(xvg_plots)

    return _section("Stage 4 — MD Validation", "\n".join(body_parts))


def _build_crosslinker(out: pathlib.Path) -> str:
    """Section 6 — Cross-linker screening results (optional)."""
    cl_data = _load_json(out / "features" / "crosslinker.json")
    if cl_data is None:
        return ""  # completely omit section if not run

    body_parts = []

    if isinstance(cl_data, list) and cl_data:
        sample_keys = list(cl_data[0].keys())
        headers = [k.replace("_", " ").title() for k in sample_keys]
        rows = []
        for entry in cl_data:
            row = []
            for k in sample_keys:
                v = entry.get(k, "")
                if isinstance(v, float):
                    v = f"{v:+.3f}"
                row.append(str(v))
            rows.append(row)
        body_parts.append(_table(headers, rows, "Cross-linker Screening"))
    elif isinstance(cl_data, dict):
        headers = ["Cross-linker", "Binding Energy (kcal/mol)"]
        rows = []
        for name, val in cl_data.items():
            if isinstance(val, dict):
                be = val.get("bsse_dE", val.get("raw_dE", val.get("binding_energy", val.get("dE", "N/A"))))
            else:
                be = val
            if isinstance(be, (int, float)):
                be = f"{be:+.3f}"
            rows.append([name, be])
        body_parts.append(_table(headers, rows, "Cross-linker Screening"))

    return _section("Cross-linker Results", "\n".join(body_parts))


def _build_esp_gallery(out: pathlib.Path) -> str:
    """Section 7 — ESP map gallery (optional)."""
    pattern = str(out / "stage2" / "stage2_esp_*.png")
    esp_files = sorted(glob.glob(pattern))
    if not esp_files:
        return ""  # omit section entirely

    imgs = []
    for fp in esp_files:
        p = pathlib.Path(fp)
        uri = _embed_image(p)
        if uri:
            label = p.stem.replace("stage2_esp_", "").replace("_", " ")
            imgs.append(
                f'<div><img src="{uri}" alt="{label}" '
                f'style="max-width:320px; border:1px solid #ddd;"/>'
                f'<p style="text-align:center; font-size:13px;">{label}</p></div>'
            )

    body = f'<div class="gallery">\n' + "\n".join(imgs) + "\n</div>"
    return _section("ESP Map Gallery", body)


def _build_recommendations(out: pathlib.Path) -> str:
    """Section 8 — Final recommendations."""
    dft_data = _load_json(out / "stage2" / "stage2_dft.json")
    md_data = _load_json(out / "stage5" / "stage5_md.json")
    cl_data = _load_json(out / "features" / "crosslinker.json")
    csv_rows = _load_csv(out / "stage3" / "stage3_selectivity.csv")

    if dft_data is None and md_data is None:
        return _section("Final Recommendations", _not_executed())

    body_parts = ['<div class="recommend-box">']

    # --- TOP-3 monomers ---
    # Priority: Stage 3 selectivity > Stage 2 binding energy
    top_names = []
    stage3_top = _load_json(out / "stage3" / "stage3_top.json")
    if stage3_top and isinstance(stage3_top, list):
        top_names = stage3_top[:3]
    elif csv_rows:
        # Fallback: sort by avg_log_S from selectivity CSV
        scored = [(r.get("monomer", ""), float(r.get("avg_log_S", 0)))
                  for r in csv_rows if r.get("avg_log_S")]
        scored.sort(key=lambda x: -x[1])  # higher log_S = more selective
        top_names = [name for name, _ in scored[:3]]
    elif isinstance(dft_data, dict):
        # Last fallback: Stage 2 binding energy
        avg_be = {}
        for m, solvs in dft_data.items():
            if isinstance(solvs, dict):
                energies = [v.get("bsse_dE", v.get("raw_dE", 0))
                            for v in solvs.values() if isinstance(v, dict)]
                avg_be[m] = sum(energies) / len(energies) if energies else 0.0
        top_names = sorted(avg_be, key=avg_be.get)[:3]

    if top_names:
        body_parts.append("<h3>TOP-3 Recommended Monomers</h3><ol>")
        for name in top_names:
            smiles = MONOMER_LIBRARY.get(name, "")
            body_parts.append(f"<li><b>{name}</b> &mdash; <code>{smiles}</code></li>")
        body_parts.append("</ol>")
    else:
        body_parts.append("<p>Insufficient data to rank monomers.</p>")

    # --- Recommended cross-linker ---
    if cl_data is not None:
        best_cl = None
        best_be = float("inf")
        if isinstance(cl_data, list):
            for entry in cl_data:
                be = entry.get("dE_kcal", entry.get("binding_energy", entry.get("dE", 0)))
                if isinstance(be, (int, float)) and be < best_be:
                    best_be = be
                    best_cl = entry.get("crosslinker", entry.get("name", ""))
        elif isinstance(cl_data, dict):
            for name, val in cl_data.items():
                if isinstance(val, dict):
                    be = val.get("bsse_dE", val.get("raw_dE", val.get("dE", 0)))
                else:
                    be = val if isinstance(val, (int, float)) else 0
                if be < best_be:
                    best_be = be
                    best_cl = name
        if best_cl:
            body_parts.append(
                f"<p><b>Recommended Cross-linker:</b> {best_cl} "
                f"({best_be:+.3f} kcal/mol)</p>")

    # --- Recommended template:monomer ratio ---
    if isinstance(md_data, list) and md_data:
        best_entry = min(md_data, key=lambda x: x.get("EBN", float("inf")))
        ratio = best_entry.get("ratio", "N/A")
        mono = best_entry.get("monomer", "N/A")
        ebn = best_entry.get("EBN", "N/A")
        if isinstance(ebn, float):
            ebn = f"{ebn:.3f}"
        body_parts.append(
            f"<p><b>Recommended Template:Monomer Ratio:</b> 1:{ratio} "
            f"(best EBN = {ebn}, monomer = {mono})</p>")

    body_parts.append("</div>")
    return _section("Final Recommendations", "\n".join(body_parts))


def _build_references() -> str:
    """Section 9 — Reference papers."""
    refs = [
        "Haupt, K.; Mosbach, K. <i>Chem. Rev.</i> <b>2000</b>, 100, 2495-2504.",
        "Mukasa, R. et al. <i>ACS Appl. Polym. Mater.</i> <b>2023</b>, 5, 6388-6397.",
        "Bannwarth, C.; Ehlert, S.; Grimme, S. <i>J. Chem. Theory Comput.</i> <b>2019</b>, 15, 1652-1671. (GFN2-xTB)",
        "Becke, A. D. <i>J. Chem. Phys.</i> <b>1993</b>, 98, 5648-5652. (B3LYP)",
        "Grimme, S. et al. <i>J. Comput. Chem.</i> <b>2011</b>, 32, 1456-1465. (D3-BJ dispersion)",
        "Lipparini, F. et al. <i>J. Chem. Theory Comput.</i> <b>2013</b>, 9, 3637-3648. (ddCOSMO)",
        "Eastman, P. et al. <i>PLoS Comput. Biol.</i> <b>2017</b>, 13, e1005659. (OpenMM)",
    ]
    items = "\n".join(f"<li>{r}</li>" for r in refs)
    return _section("References", f"<ol>{items}</ol>")


# ---------------------------------------------------------------------------
# XVG inline plot utility (ported from Bio project)
# ---------------------------------------------------------------------------

def _xvg_to_inline_plot(xvg_path, title, xlabel, ylabel):
    """Convert GROMACS XVG file to inline base64 plot for HTML embedding."""
    xvg_path = pathlib.Path(xvg_path)
    if not xvg_path.exists():
        return f"<p><em>{title}: data not available</em></p>"
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        import io

        data = []
        for line in xvg_path.read_text().split("\n"):
            if line.startswith(("#", "@")) or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    data.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    pass
        if not data:
            return f"<p><em>{title}: no data</em></p>"

        arr = np.array(data)
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(arr[:, 0], arr[:, 1], linewidth=0.8, color='#2196F3')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        img_data = base64.b64encode(buf.getvalue()).decode()
        return f'<img src="data:image/png;base64,{img_data}" style="max-width:100%;" />'
    except Exception as e:
        return f"<p><em>{title}: plot error ({e})</em></p>"


def _build_selectivity_matrix(out):
    """Build interferent selectivity matrix as HTML table."""
    s3_path = out / "stage3" / "stage3_interferent_dft.json"
    s2_path = out / "stage2" / "stage2_dft.json"
    if not s3_path.exists() or not s2_path.exists():
        return ""

    s2 = json.loads(s2_path.read_text())
    s3 = json.loads(s3_path.read_text())

    # Get template binding energy per monomer
    tmpl_be = {}
    for m, solvents in s2.items():
        for solv, data in solvents.items():
            if isinstance(data, dict):
                tmpl_be[m] = data.get("bsse_dE", 0)
                break

    # Get interferent binding energies
    interf_names = list(s3.keys())
    if not interf_names or not tmpl_be:
        return ""

    # Build matrix
    header = "<th>Monomer</th><th>Template</th>" + "".join(
        f"<th>{i}</th>" for i in interf_names)

    rows = ""
    for m in sorted(tmpl_be.keys()):
        t_be = tmpl_be[m]
        cells = f"<td style='font-weight:bold;background:#e8f5e9;'>{t_be:.2f}</td>"
        for iname in interf_names:
            i_data = s3.get(iname, {}).get(m, {})
            i_be = list(i_data.values())[0] if isinstance(i_data, dict) and i_data else None
            if i_be is not None:
                color = "green" if abs(i_be) < abs(t_be) else "red"
                cells += f"<td style='color:{color};'>{i_be:.2f}</td>"
            else:
                cells += "<td>N/A</td>"
        rows += f"<tr><td><strong>{m}</strong></td>{cells}</tr>"

    return f"""
    <h2>Selectivity Matrix (Binding Energy, kcal/mol)</h2>
    <div style="overflow-x:auto;">
        <p>Template column (green) vs interferents. <span style="color:red;">Red</span> = interferent binds stronger (poor selectivity).</p>
        <table><tr>{header}</tr>{rows}</table>
    </div>"""


def _build_stage5_section(out):
    """Build Stage 5 VIP rebinding section for HTML report."""
    vip_path = out / "stage6" / "stage6_vip.json"
    if not vip_path.exists():
        return ""

    vip = json.loads(vip_path.read_text())
    rows = ""
    for m, data in sorted(vip.items(), key=lambda x: -x[1].get("vip_score", 0)
                           if isinstance(x[1], dict) else 0):
        if not isinstance(data, dict):
            continue
        rows += f"""<tr>
            <td>{m}</td>
            <td>{data.get('rebind_rate', 0):.0%}</td>
            <td>{data.get('removal_rate', 0):.0%}</td>
            <td>{data.get('selectivity_index', 0):.2f}</td>
            <td><strong>{data.get('vip_score', 0):.3f}</strong></td>
        </tr>"""

    return f"""
    <h2>Stage 5: VIP Cavity Rebinding</h2>
    <table>
        <tr><th>Monomer</th><th>Rebind Rate</th><th>Removal Rate</th>
            <th>Selectivity Index</th><th>VIP Score</th></tr>
        {rows}
    </table>"""


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_report(output_dir: str = OUTPUT_DIR) -> str:
    """Generate a self-contained HTML report and return the file path."""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    safe_name = TEMPLATE_NAME.replace(" ", "_")
    filename = f"mip_report_{safe_name}_{timestamp}.html"
    report_path = out / filename

    # Assemble HTML
    sections = [
        _build_header(now),
        _build_stage1(out),
        _build_stage2(out),
        _build_stage3(out),
        _build_selectivity_matrix(out),
        _build_stage4(out),
        _build_stage5_section(out),
        _build_crosslinker(out),
        _build_esp_gallery(out),
        _build_recommendations(out),
        _build_references(),
    ]

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>MIP Report — {TEMPLATE_NAME}</title>
<style>
{INLINE_CSS}
</style>
</head>
<body>
{"".join(sections)}
<hr/>
<p style="font-size:12px; color:#999; text-align:center;">
  Generated by MIP Screening Pipeline &mdash; {now.strftime("%Y-%m-%d %H:%M:%S")}
</p>
</body>
</html>"""

    report_path.write_text(html, encoding="utf-8")
    print(f"Report saved: {report_path}")
    return str(report_path)


if __name__ == "__main__":
    generate_report()
