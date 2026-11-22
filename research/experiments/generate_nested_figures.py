from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('research/results/final/nested_cv_gpu'); out=Path('figures'); out.mkdir(exist_ok=True)
labels={'direct_gru':'Direct GRU','ridge_direct':'Ridge','oc_full':'Operating-conditioned','persistence':'Persistence'}
order=['direct_gru','ridge_direct','oc_full','persistence']
summary=pd.read_csv(root/'stats_cv/summary_table.csv').set_index('exp')
fig,ax=plt.subplots(figsize=(7.2,4.2))
for i,k in enumerate(order):
 m=summary.loc[k,'MAE']; lo=m-summary.loc[k,'MAE_lo']; hi=summary.loc[k,'MAE_hi']-m
 ax.errorbar(m,i,xerr=[[lo],[hi]],fmt='o',color=['#245b8f','#6d879b','#6d879b','#888888'][i],capsize=3,lw=1.7,ms=6)
ax.set_yticks(range(len(order)),[labels[k] for k in order]); ax.invert_yaxis(); ax.set_xlabel('Pooled 20-step MAE (°C), 95% profile-cluster bootstrap CI'); ax.set_title('Nested profile-grouped evaluation'); ax.grid(axis='x',alpha=.25); fig.tight_layout(); fig.savefig(out/'nested_cv_mae.png',dpi=220,bbox_inches='tight'); plt.close(fig)
raw=pd.read_csv(root/'profile_metrics.csv')
prof=raw.groupby(['exp','split','profile'],as_index=False).mean(numeric_only=True)
rng=np.random.RandomState(2026); rows=[]
for k in order:
 d=prof[prof.exp==k].sort_values(['split','profile']); w=d.n_windows.to_numpy()
 for step in range(1,21):
  v=d[f'mae_h{step:02d}'].to_numpy(); val=np.sum(v*w)/np.sum(w)
  bi=rng.randint(0,len(d),size=(10000,len(d))); bm=np.sum(v[bi]*w[bi],axis=1)/np.sum(w[bi],axis=1)
  rows.append(dict(exp=k,step=step,MAE=val,lo=np.percentile(bm,2.5),hi=np.percentile(bm,97.5)))
h=pd.DataFrame(rows); h.to_csv(root/'stats_cv/horizon_all_models.csv',index=False)
fig,ax=plt.subplots(figsize=(7.2,4.4))
for k in order:
 d=h[h.exp==k]; ax.plot(d.step,d.MAE,label=labels[k],lw=2); ax.fill_between(d.step,d.lo,d.hi,alpha=.10)
ax.set_xlabel('Forecast step'); ax.set_ylabel('MAE (°C)'); ax.set_title('Forecast error by horizon'); ax.grid(alpha=.25); ax.legend(frameon=False,ncol=2); fig.tight_layout(); fig.savefig(out/'nested_cv_horizon.png',dpi=220,bbox_inches='tight'); plt.close(fig)
print(h[h.step.isin([1,10,20])].to_string(index=False))
