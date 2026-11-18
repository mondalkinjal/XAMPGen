"""Reproduce cached manuscript analyses without model weights or a GPU.
Original numerical definitions are executed unchanged; outputs stay in --output.
"""
import argparse, contextlib, importlib.util, io, json, os, runpy, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('reproduced'));a=p.parse_args()
root=Path(__file__).resolve().parents[1];src=root/'sequence_gen';out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
work=out/'sequence_gen';work.mkdir(exist_ok=True)
(work.parent/'69c005365ec28e7b5ef2bc4c/Figures').mkdir(parents=True,exist_ok=True)
for folder in ['probing_multiseed','top10_layers']:shutil.copytree(src/folder,work/folder,dirs_exist_ok=True)
sys.path.insert(0,str(src));os.chdir(work)
import helicity_layer_profile as h
from motif_common import sym
cache=np.load('top10_layers/attn_all_layers.npz')
scores=np.array([[h.score(sym(cache[name][layer])) for name,_,_ in h.PEPTIDES] for layer in range(36)])
saved=np.load('top10_layers/helicity_scores_all_layers.npy')
np.testing.assert_allclose(scores,saved,rtol=1e-6,atol=1e-8)
y=np.array([v for _,_,v in h.PEPTIDES]);r=np.array([pearsonr(x,y)[0] for x in scores]);best=int(r.argmax())
assert best==22,(best,r[best]);assert abs(r[22]-0.695)<0.001,r[22]
np.save(out/'recomputed_helicity_scores.npy',scores);h.OUT=str(out);h.main()
runpy.run_path(str(src/'probing_multiseed_summary.py'),run_name='__main__')
pd.testing.assert_frame_equal(pd.read_csv(work/'probing_multiseed/layer_summary.csv'),pd.read_csv(src/'probing_multiseed/layer_summary.csv'),check_exact=False,rtol=1e-6,atol=1e-8)
runpy.run_path(str(src/'probing_4panel.py'),run_name='__main__')
novelty=src/'novelty_analysis';os.chdir(novelty)
spec=importlib.util.spec_from_file_location('align_tables',novelty/'align_tables.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for mode in ['train','sp']:
 buf=io.StringIO()
 with contextlib.redirect_stdout(buf):m.main(mode)
 text=buf.getvalue();assert sum('&' in line for line in text.splitlines())==18
 (out/f'novelty_{mode}.tex').write_text(text)
report={'helicity_scores_match_saved':True,'best_layer':best,'pearson_r':float(r[best]),'probe_summary_matches_saved':True,'novelty_rows_per_table':18,'reproduced_from':'saved attention maps, probe results and BLAST alignments','not_tested':['training from scratch','checkpoint inference','new BLAST database searches','full filtering pipeline']}
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
