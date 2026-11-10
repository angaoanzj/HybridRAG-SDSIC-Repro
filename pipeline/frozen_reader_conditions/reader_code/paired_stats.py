"""Frozen paired statistics, independent of all retrieval code."""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np

SEED = 2026110602
B = 5000

def support_components(labels,ordered_ids):
    parent=list(range(len(ordered_ids)))
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    owners={}
    for i,qid in enumerate(ordered_ids):
        for title in set(labels[qid]['gold_titles']):
            if title in owners:
                a,b=find(i),find(owners[title]);parent[a]=b
            else:owners[title]=i
    groups=defaultdict(list)
    for i in range(len(parent)):groups[find(i)].append(i)
    return sorted(groups.values(),key=lambda x:x[0])

def paired_interval(a,b,components,seed=SEED,resamples=B):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or not len(a):raise ValueError('Invalid paired arrays')
    diff=a-b;n=len(diff);rng=np.random.default_rng(seed)
    boot=diff[rng.integers(0,n,size=(resamples,n))].mean(axis=1)
    sums=np.asarray([diff[c].sum() for c in components]);sizes=np.asarray([len(c) for c in components])
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(components),size=(resamples,len(components)))
    cluster=sums[indices].sum(axis=1)/sizes[indices].sum(axis=1)
    return {'n_pairs':n,'difference':float(diff.mean()),
            'question_ci95':np.percentile(boot,[2.5,97.5],method='linear').tolist(),
            'cluster_ci95':np.percentile(cluster,[2.5,97.5],method='linear').tolist(),
            'wins':int(np.sum(diff>0)),'losses':int(np.sum(diff<0)),'ties':int(np.sum(diff==0)),
            'seed':seed,'bootstrap_resamples':resamples,'interval':'paired percentile bootstrap; pointwise unadjusted',
            'n_components':len(components),'largest_component':max(map(len,components)),
            'cluster_estimand':'Pooled question-weighted mean difference after resampling whole exact-support-title components.',
            'limits':'Conditions on frozen sample/corpus/protocol; support-title components may miss broader dependence.'}

def exact_mcnemar(wins,losses):
    """Two-sided exact conditional binomial test, p=.5, on discordant pairs."""
    n=int(wins)+int(losses)
    if n==0:return 1.0
    tail=sum(math.comb(n,k) for k in range(min(int(wins),int(losses))+1))
    return min(1.0,2*tail/(2**n))

def holm_adjust(pvalues):
    """Holm step-down adjusted p-values, preserving input order."""
    m=len(pvalues);result=[None]*m;previous=0.0
    for rank,i in enumerate(sorted(range(m),key=lambda i:(pvalues[i],i))):
        previous=max(previous,min(1.0,(m-rank)*float(pvalues[i])))
        result[i]=previous
    return result

def timing_ratio(a,b,seed=SEED,resamples=B):
    """Arrays contain each paired question's median of its five observations."""
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if len(a)!=len(b) or min(a.min(),b.min())<=0:raise ValueError('Invalid timing pairs')
    rng=np.random.default_rng(seed);ix=rng.integers(0,len(a),size=(resamples,len(a)))
    ratio=np.median(a[ix],axis=1)/np.median(b[ix],axis=1)
    return {'n_question_pairs':len(a),'ratio_of_across_question_medians':float(np.median(a)/np.median(b)),
            'ratio_ci95':np.percentile(ratio,[2.5,97.5],method='linear').tolist(),
            'seed':seed,'bootstrap_resamples':resamples,
            'scope':'Warm total_seconds, each question reduced to its median of five measured observations; paired questions resampled.',
            'multiplicity':'Secondary pointwise timing interval, outside the four primary coverage hypotheses.'}
