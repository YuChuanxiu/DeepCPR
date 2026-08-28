# -*- coding: utf-8 -*-
"""
DeepCPR local web interface (Gradio).
Run:
	python app.py        (or: gradio app.py)
A browser window opens at http://127.0.0.1:7860 automatically.
All computation runs on the local machine; no external server is involved.
"""
import os
import sys
import gc
import glob
import time
import shutil
import traceback
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # headless backend: no figure windows pop up at runtime
import gradio as gr
import pandas as pd
from PIL import Image

# 兼容直接运行（python app.py）与包模块方式运行（python -m DeepCPR.app）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DeepCPR import data_resolution  # same import style as workflow.py
try:
    from DeepCPR.csv_merge import peaktable  # used to build the merged peak table
    HAS_PEAKTABLE = True
except Exception:
    HAS_PEAKTABLE = False

theme = gr.themes.Default(
	font=[gr.themes.LocalFont("Microsoft YaHei"),
		  gr.themes.LocalFont("Segoe UI"),
		  "Arial", "sans-serif"],                                  # Interface text: Microsoft YaHei
	font_mono=[gr.themes.LocalFont("Consolas"),
			   gr.themes.LocalFont("Courier New"),
			   "monospace"],                                       # Code/path box: monospaced font
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEEPCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example", "DeepCS.h5")
DEFAULT_DEEPCPR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example", "DeepCPR.h5")
RUNS_DIR = os.path.join(REPO_ROOT, "gradio_runs")  # all results are saved here
CDF_SUFFIXES = (".cdf", ".nc", ".netcdf")
def _collect_table(folder):
	"""Read all CSV files in a folder and merge them into one DataFrame."""
	frames = []
	for c in sorted(glob.glob(os.path.join(folder, "*.csv"))):
		df = pd.read_csv(c)
		df.insert(0, "file", os.path.basename(c)[:-4])
		frames.append(df)
	return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
def _tif_to_png(save_dir):
	"""Convert the .tif figures produced by the core code to .png.
	The core code itself is not modified.
	"""
	pngs = []
	pattern = os.path.join(save_dir, "figure", "**", "*.tif")
	for tif in glob.glob(pattern, recursive=True):
		png = os.path.splitext(tif)[0] + ".png"
		try:
			Image.open(tif).convert("RGB").save(png)
			os.remove(tif)
			pngs.append(png)
		except Exception as e:
			print("[warning] tif->png conversion failed:", tif, e)
	return pngs
def resolve(files, deepcs_path, deepcpr_path, adaptive,
			max_iter, max_comp, gen_image, progress=gr.Progress()):
	"""Run DeepCPR resolution on the uploaded CDF files and collect outputs."""
	t0 = time.time()
	# ---- input validation ----
	if not files:
		raise gr.Error("Please upload at least one GC-MS data file first.")
	for p, name in ((deepcs_path, "DeepCS model"), (deepcpr_path, "DeepCPR model")):
		if not p or not os.path.isfile(p):
			raise gr.Error(
				f"{name} file not found: {p} "
				"(both .h5 and .onnx are supported; please check the path)")
	cdfs = [f.name for f in files
			if os.path.splitext(f.name)[1].lower() in CDF_SUFFIXES]
	if not cdfs:
		raise gr.Error(
			"No CDF files found in the upload "
			"(supported extensions: .cdf / .nc / .netcdf)")
	# ---- create the output folder for this run ----
	run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
	save_dir = os.path.join(RUNS_DIR, run_id)
	tmp_in = os.path.join(save_dir, "_tmp_input")
	os.makedirs(tmp_in, exist_ok=True)
	figures, per_file_time = [], []
	n = len(cdfs)
	try:
		# process the files one by one so the progress bar can update
		for i, src in enumerate(cdfs):
			fname = os.path.basename(src)
			progress(i / n, desc=f"[{i+1}/{n}] Resolving {fname} "
						"(each file takes several minutes; detailed progress "
						"is printed to the terminal)")
			t_file = time.time()
			# keep only the current file in the temporary input folder
			for old in os.listdir(tmp_in):
				os.remove(os.path.join(tmp_in, old))
			shutil.copy2(src, os.path.join(tmp_in, fname))
			data_resolution(
				tmp_in, deepcs_path, deepcpr_path, save_dir,
				True if gen_image else None,
				adaptive=adaptive,
				adaptive_kwargs=({"max_iterations": int(max_iter),
									"max_components": int(max_comp)}
									if adaptive else None),
			)
			if gen_image:
				figures += _tif_to_png(save_dir)
			per_file_time.append(f"{fname}: {(time.time()-t_file)/60:.1f} min")
			gc.collect()
		progress(0.97, desc="Collecting results...")
		# merge the peak tables (same as workflow.py; a failure here
		# does not affect the resolution results themselves)
		merged_note = ""
		if HAS_PEAKTABLE:
			try:
				peaktable(os.path.join(save_dir, "single"), save_dir)
				merged_note = "\n- `peak_area_table.csv` (merged peak table)"
			except Exception as e:
				print("[warning] peaktable merge failed:", e)
		peak_df = _collect_table(os.path.join(save_dir, "single")).round(4)
		seg_df = _collect_table(os.path.join(save_dir, "seg")).round(4)
		# pack everything into a zip archive for download
		progress(0.99, desc="Packing download archive...")
		zip_path = shutil.make_archive(
			os.path.join(RUNS_DIR, f"DeepCPR_results_{run_id}"), "zip", save_dir)
		n_peaks = len(peak_df)
		status = (
			f"✅ **Done**: {n} file(s), **{n_peaks}** peaks resolved in total, "
			f"total time {(time.time()-t0)/60:.1f} min\n\n"
			f"📁 Results folder: `{save_dir}`{merged_note}\n"
			"- `single/*.csv` peak tables | `seg/*.csv` segment info | "
			"`ms/**/*.msp` mass spectra (NIST compatible) | "
			"`figure/**/*.png` figures\n\n"
			"⏱ Time per file: " + "; ".join(per_file_time)
		)
		gallery = [(p, os.path.relpath(p, save_dir)) for p in figures]
		return status, peak_df, seg_df, gallery, zip_path
	except Exception as e:
		traceback.print_exc()
		raise gr.Error(f"Resolution failed: {type(e).__name__}: {e}")
with gr.Blocks(title="DeepCPR", 
			   theme=theme,
			   css="""
					/* ---- tighten Gradio default spacing ---- */
					.column { gap:5px !important; }
					.html-container { padding:0 !important; }
					/* ---- color-coded section titles ---- */
					.sec { display:flex; align-items:center; gap:8px;
							font-size:1.02em; font-weight:700;
							margin:4px 0 0 0; padding:8px 12px;
							border-radius:9px; }
					.sec .dot { width:10px; height:10px; border-radius:50%;
								flex:0 0 auto; }
					.sec-blue   { background:#e7f0fb; color:#1f5aa8; }
					.sec-blue   .dot { background:#2f7bd1; }
					.sec-purple { background:#f1ebfa; color:#66349e; }
					.sec-purple .dot { background:#8a5cd6; }
					.sec-green  { background:#e7f6ee; color:#1c7a4c; }
					.sec-green  .dot { background:#2fa564; }
					.sec-orange { background:#fdf0e1; color:#a85d10; }
					.sec-orange .dot { background:#e8822a; }
					/* ---- color-coded panels ---- */
					.panel { border-radius:12px; padding:12px 14px;
							 margin:0 0 12px 0; }
					.panel-blue   { background:#f3f7fc; border:1px solid #cfdff2; }
					.panel-purple { background:#f8f4fc; border:1px solid #e0d3f0; }
					.panel-green  { background:#f2faf5; border:1px solid #cde8d8; }
					.panel-orange { background:#fdf8f0; border:1px solid #efdcc2; }
					/* ---- accordion label coloring ---- */
					#acc-adaptive > .label-wrap { background:#f1ebfa;
							border:1px solid #dcc9f0; border-radius:8px; }
					#acc-adaptive > .label-wrap:hover { background:#e8ddf5; }
					#acc-segment  > .label-wrap { background:#fdf3e5;
							border:1px solid #efd9ba; border-radius:8px; }
					#acc-segment  > .label-wrap:hover { background:#faead2; }
					/* ---- run button ---- */
					#run_btn { background:linear-gradient(135deg,#2f7bd1,#4aa0e6);
							   color:#fff; border:none; }
					#run_btn:hover { filter:brightness(1.08); }
					""",
				) as demo:
		gr.HTML("""
		<div style="font-size:2.4em;font-weight:800;line-height:1.2;">DeepCPR</div>
		<div style="font-size:1.2em;color:#5a6572;margin-top:4px;">
			Deep learning-based Chromatographic Profile Resolution —
			runs locally, your data never leaves this machine
		</div>
		""")
		with gr.Row():
			with gr.Column(scale=2):
				gr.HTML("""<div class="sec sec-blue"><span class="dot"></span>📥 Data input</div>""")
				with gr.Group(elem_classes=["panel", "panel-blue"]):
					files_in = gr.File(label="GC-MS data files (multiple allowed)",
										file_count="multiple",
										file_types=[".cdf", ".nc", ".netcdf"])
					deepcs_in = gr.Textbox(value=DEFAULT_DEEPCS,
											label="DeepCS model path (.h5 or .onnx)")
					deepcpr_in = gr.Textbox(value=DEFAULT_DEEPCPR,
											label="DeepCPR model path (.h5 or .onnx)")
				gr.HTML("""<div class="sec sec-purple"><span class="dot"></span>⚙️ Resolution options</div>""")
				with gr.Group(elem_classes=["panel", "panel-purple"]):
					adaptive_in = gr.Checkbox(
						False,
						label="Adaptive mode (use when a segment contains "
								"more than five co-eluting components)")
					with gr.Accordion("Adaptive mode parameters", open=False,
										elem_id="acc-adaptive"):
						max_iter_in = gr.Slider(1, 16, value=4, step=1,
												label="max_iterations")
						max_comp_in = gr.Slider(6, 32, value=32, step=1,
												label="max_components")
					figs_in = gr.Checkbox(
						False, label="Generate figures (PNG; increases runtime)")
				run_btn = gr.Button("▶ Start resolution", variant="primary",
									elem_id="run_btn")
			with gr.Column(scale=3):
				gr.HTML("""<div class="sec sec-green"><span class="dot"></span>📊 Status</div>""")
				with gr.Group(elem_classes=["panel", "panel-green"]):
					status_out = gr.Markdown()
				gr.HTML("""<div class="sec sec-orange"><span class="dot"></span>📋 Results</div>""")
				with gr.Group(elem_classes=["panel", "panel-orange"]):
					peak_out = gr.Dataframe(label="Peak table (all files)",
											interactive=False, max_height=380)
					with gr.Accordion("Segment summary", open=False,
										elem_id="acc-segment"):
						seg_out = gr.Dataframe(label="Segment info",
												interactive=False, max_height=260)
					gallery_out = gr.Gallery(
						label="Resolution figures (shown when figure generation "
								"is enabled)", columns=3, height="auto")
					zip_out = gr.File(label="Download all results (.zip)")
			run_btn.click(
				resolve,
				[files_in, deepcs_in, deepcpr_in, adaptive_in,
					max_iter_in, max_comp_in, figs_in],
				[status_out, peak_out, seg_out, gallery_out, zip_out],
			)
if __name__ == "__main__":
	demo.queue(default_concurrency_limit=1)  # one job at a time to avoid OOM
	demo.launch(inbrowser=True, show_error=True)