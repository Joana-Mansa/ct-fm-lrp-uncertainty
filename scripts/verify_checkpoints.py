"""Independent checkpoint and recorded-metric checks; does not overwrite experiments."""
import sys, os, json, hashlib, importlib.metadata, inspect
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata,spearmanr
R=Path(__file__).resolve().parents[1]; OUT=R / "verification_run";OUT.mkdir(parents=True,exist_ok=True)
os.chdir(R);sys.path.insert(0,str(R/'src'))
from model import NoduleClassifier
from data import Nodules
from uncertainty import expected_calibration_error
from explain import lrp_attribution,gradcam_attribution
from faithfulness import deletion_curve,insertion_curve
from torch.utils.data import DataLoader

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def auc(y,p):
 pos=y==1;n=pos.sum();m=(~pos).sum();return float((rankdata(p)[pos].sum()-n*(n+1)/2)/(n*m))
def metrics(y,p):
 pred=p.argmax(1);return {'accuracy':float((pred==y).mean()),'balanced_accuracy':float(np.mean([(pred[y==c]==c).mean() for c in [0,1]])),'auc':auc(y,p[:,1])}
torch.set_num_threads(4);torch.manual_seed(0);dev='cuda' if torch.cuda.is_available() else 'cpu';ds=Nodules('test');dl=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0)
y=ds.labels;result={'test_n':len(ds),'test_class_counts':np.bincount(y).tolist(),'checkpoints':{},'source_hashes':{p.name:digest(p) for p in (R/'src').glob('*.py')}}
predictions={}
for required in ['ctfm_pretrained.pt', 'ctfm_scratch.pt']:
 if not (R/'weights'/required).exists():
  raise FileNotFoundError(f'Missing {required}; run scripts/download_checkpoints.py first')
for path in sorted((R/'weights').glob('ctfm_*.pt')):
 model=NoduleClassifier(pretrained=True).to(dev);ckpt=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(ckpt['state_dict'],strict=True);model.eval();probs=[]
 with torch.inference_mode():
  for x,_ in dl:probs.append(model(x.to(dev)).softmax(1).cpu().numpy())
 p=np.concatenate(probs);predictions[path.stem]=p
 result['checkpoints'][path.name]={'sha256':digest(path),'metrics':metrics(y,p),'parameters':sum(v.numel() for v in model.parameters())}
 print(path.name,result['checkpoints'][path.name],flush=True)
 if path.name=='ctfm_pretrained.pt':
  result['module_types']={type(m).__name__:sum(type(z)==type(m) for z in model.modules()) for m in model.modules()}
  x=torch.stack([ds[i][0] for i in [0,3]]).to(dev)
  with torch.no_grad():target=model(x).argmax(1);logits=model(x).cpu().numpy();feat=model.bottleneck(x)
  result['bottleneck_shape']=list(feat.shape)
  result['attribution_checks']={}
  for method in ['epsilon_plus_flat','epsilon_gamma_box','epsilon_alpha2_beta1','gradcam']:
   arr=gradcam_attribution(model,x,target) if method=='gradcam' else lrp_attribution(model,x,target,method)
   d=deletion_curve(model,x,target,arr,steps=4);ins=insertion_curve(model,x,target,arr,steps=4)
   result['attribution_checks'][method]={'shape':list(arr.shape),'finite':bool(np.isfinite(arr).all()),'signed_sum':arr.reshape(len(arr),-1).sum(1).tolist(),'selected_logit':logits[np.arange(2),target.cpu().numpy()].tolist(),'deletion_start':d[:,0].tolist(),'insertion_end':ins[:,-1].tolist()}
 del model;torch.cuda.empty_cache()
np.savez_compressed(OUT/'checkpoint_predictions.npz',labels=y,**predictions)
saved=json.loads((R/'results/ensemble.json').read_text())
if all(Path(p).stem in predictions for p in saved['members']):
 members=[predictions[Path(p).stem] for p in saved['members']];mean=np.mean(members,axis=0)
 result['ensemble_recomputed']={**metrics(y,mean),'ece':expected_calibration_error(mean,y)[0]}
else:
 result['ensemble_recomputed']={'status':'skipped: all four ensemble checkpoints are required'}
result['ensemble_recorded']=saved['ensemble']
per=np.load(R/'results/per_sample.npz');ex=json.loads((R/'results/explanations.json').read_text())
result['saved_array_checks']={'ece_recomputed':expected_calibration_error(per['probs'],per['labels'])[0],'ece_recorded':ex['calibration']['ece'],'correlations':{}}
for key in per.files:
 if key.startswith('aopc_'):
  res=spearmanr(per['entropy'],per[key]);method=key[5:];result['saved_array_checks']['correlations'][method]={'rho':float(res.statistic),'p_value':float(res.pvalue),'aopc':float(per[key].mean()),'recorded_rho':ex['faithfulness'][method]['uncertainty_vs_aopc_spearman']}
# Paired stratified bootstrap for seed-0 AUC difference, no fitting or threshold tuning.
rng=np.random.default_rng(2026);groups=[np.flatnonzero(y==c) for c in [0,1]];diff=[]
for _ in range(2000):
 ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);diff.append(auc(y[ix],predictions['ctfm_pretrained'][ix,1])-auc(y[ix],predictions['ctfm_scratch'][ix,1]))
result['seed0_auc_difference_bootstrap']={'replicates':2000,'seed':2026,'unit':'benchmark patch; patient identifiers not provided','ci95':np.quantile(diff,[.025,.975]).tolist()}
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',OUT,flush=True)
