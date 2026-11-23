"""Adapt Ingre et al. (2017), DOI 10.1007/978-3-319-63645-0_23, to
SentinelNet's three-class task. Run with the existing project dataset:
  python compare_published_baseline.py --data "path/to/data/nsl_kdd"
No app changes, no test-set model selection, no claim of exact replication.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

COLS='duration protocol_type service flag src_bytes dst_bytes land wrong_fragment urgent hot num_failed_logins logged_in num_compromised root_shell su_attempted num_root num_file_creations num_shells num_access_files num_outbound_cmds is_host_login is_guest_login count srv_count serror_rate srv_serror_rate rerror_rate srv_rerror_rate same_srv_rate diff_srv_rate srv_diff_host_rate dst_host_count dst_host_srv_count dst_host_same_srv_rate dst_host_diff_srv_rate dst_host_same_src_port_rate dst_host_srv_diff_host_rate dst_host_serror_rate dst_host_srv_serror_rate dst_host_rerror_rate dst_host_srv_rerror_rate label difficulty'.split()
TARGET=['Normal','DoS','Probe']
DOS=set('back land neptune pod smurf teardrop mailbomb apache2 processtable udpstorm worm'.split())
PROBE=set('ipsweep nmap portsweep satan mscan saint'.split())
BASIC=['src_bytes','duration','dst_bytes','protocol_type']
ENHANCED=['src_bytes','dst_bytes','duration','count','srv_count','same_srv_rate','serror_rate','dst_host_diff_srv_rate','protocol_type','flag']
# Ingre et al., Table 2: exact selected raw feature names.
CFS14=['duration','protocol_type','service','flag','src_bytes','dst_bytes','logged_in','serror_rate','srv_serror_rate','same_srv_rate','diff_srv_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate']
CATEGORICAL={'protocol_type','service','flag','logged_in'}
FLAGS='SF S0 REJ RSTR RSTO S1 SH S2 RSTOS0 S3 OTH'.split()

def label(s):
    s=str(s).strip().rstrip('.').lower()
    return 'Normal' if s=='normal' else 'DoS' if s in DOS else 'Probe' if s in PROBE else None

def read(path):
    df=pd.read_csv(path,header=None,names=COLS)
    df['category']=df['label'].map(label)
    return df[df.category.notna()].reset_index(drop=True)

def fit_tree_features(train, df):
    out=df[CFS14].copy()
    for c in ['protocol_type','service','flag']:
        # Frequency-ranked numbers; mappings only from fitting data.
        order=train[c].value_counts().index.tolist()
        out[c]=out[c].map({v:i+1 for i,v in enumerate(order)}).fillna(0)
    return out.astype(float)

def rf_features(df, mode):
    nums=BASIC[:3] if mode=='Basic' else ENHANCED[:8]
    out=df[nums].copy().astype(float)
    for p in ['TCP','UDP','ICMP']:
        out['proto_'+p]=(df.protocol_type.str.upper()==p).astype(int)
    if mode=='Enhanced':
        for fl in FLAGS: out['flag_'+fl]=(df.flag.str.upper()==fl).astype(int)
    return out,nums

def fit_rf(train,test,mode):
    a,nums=rf_features(train,mode);b,_=rf_features(test,mode)
    sc=StandardScaler();a[nums]=sc.fit_transform(a[nums]);b[nums]=sc.transform(b[nums])
    rf=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1).fit(a,train.category)
    return rf.predict(b)

def metrics(y,p):
    rec=recall_score(y,p,labels=TARGET,average=None,zero_division=0)
    return {'accuracy':accuracy_score(y,p),'weighted_f1':f1_score(y,p,average='weighted',zero_division=0),'macro_f1':f1_score(y,p,average='macro',zero_division=0),'normal_recall':rec[0],'dos_recall':rec[1],'probe_recall':rec[2]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True,type=Path);parser.add_argument('--out',type=Path,default=Path('published_comparison_results'));args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    train_path=args.data/'KDDTrain+.txt'; test_path=args.data/'KDDTest+.txt'; train,test=read(train_path),read(test_path);y=test.category.to_numpy()
    assert len(train)==124926 and len(test)==19592
    fit,val=train_test_split(train,test_size=.2,random_state=42,stratify=train.category)
    a=fit_tree_features(fit,fit);b=fit_tree_features(fit,val)
    # Fixed grid before looking at KDDTest+, select based on validation accuracy.
    # Use full three-class training data rather than unpublished sampling counts.
    grid=[0,1e-6,1e-5,1e-4,1e-3,1e-2]
    selection=[]
    for alpha in grid:
        tree=DecisionTreeClassifier(criterion='gini',ccp_alpha=alpha,random_state=42).fit(a,fit.category)
        selection.append({'ccp_alpha':alpha,'validation_accuracy':accuracy_score(val.category,tree.predict(b)),'leaves':tree.get_n_leaves()})
    best=max(selection,key=lambda r:(r['validation_accuracy'],r['ccp_alpha']))
    tree=DecisionTreeClassifier(criterion='gini',ccp_alpha=best['ccp_alpha'],random_state=42).fit(fit_tree_features(train,train),train.category)
    paper=tree.predict(fit_tree_features(train,test))
    basic=fit_rf(train,test,'Basic');enhanced=fit_rf(train,test,'Enhanced')
    complete=[]
    for name,n,p in [('Ingre-inspired CFS14 CART',14,paper),('SentinelNet Basic RF',4,basic),('SentinelNet Enhanced RF',10,enhanced)]:
        complete.append({'model':name,'raw_features':n,**metrics(y,p),'confusion_matrix':confusion_matrix(y,p,labels=TARGET).tolist()})
    # Simulate records carrying only the common four Basic fields. The static
    # CART receives training-only median/mode imputation for its missing ten.
    degraded=test.copy()
    for c in set(CFS14)-set(BASIC):
        degraded[c]=train[c].mode().iloc[0] if c in CATEGORICAL else train[c].median()
    paper_degraded=tree.predict(fit_tree_features(train,degraded))
    rows=[]
    for seed in [42,43,44,45,46]:
        u=np.random.default_rng(seed).random(len(test))
        for rate in [0,.25,.5,.75,1]:
            mask=u<rate
            for name,pred in [('Ingre-inspired fixed CART with imputation',np.where(mask,paper_degraded,paper)),('Adaptive SentinelNet',np.where(mask,basic,enhanced))]:
                rows.append({'seed':seed,'missing_target_pct':rate*100,'missing_actual_pct':mask.mean()*100,'model':name,**metrics(y,pred)})
    results=pd.DataFrame(rows);summary=results.groupby(['missing_target_pct','model'],as_index=False).agg(accuracy_mean=('accuracy','mean'),accuracy_min=('accuracy','min'),accuracy_max=('accuracy','max'),macro_f1_mean=('macro_f1','mean'),probe_recall_mean=('probe_recall','mean'))
    pd.DataFrame(selection).to_csv(args.out/'validation_pruning.csv',index=False)
    pd.DataFrame(complete).drop(columns='confusion_matrix').to_csv(args.out/'complete_data_comparison.csv',index=False)
    results.to_csv(args.out/'missingness_all_runs.csv',index=False);summary.to_csv(args.out/'missingness_summary.csv',index=False)
    meta={'reference':'Ingre, Yadav & Soni (2017), DOI 10.1007/978-3-319-63645-0_23','scope':'three-class adaptation; not exact reproduction','train':len(train),'test':len(test),'train_counts':train.category.value_counts().to_dict(),'test_counts':test.category.value_counts().to_dict(),'sklearn_version':sklearn.__version__,'train_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),'test_sha256':hashlib.sha256(test_path.read_bytes()).hexdigest(),'paper_features':CFS14,'selected_pruning':best,'class_order':TARGET,'complete':complete,'notes':'Original paper uses binary/five-class tasks, sampling, MATLAB CART, WEKA CFS; this uses their published fixed feature subset, full training data, scikit-learn CART and train-validation pruning. The missing-data treatment is our explicit experimental baseline assumption, not a method claimed by the paper. Six additional SentinelNet or ten additional CART fields hidden on exactly the same rows.'}
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
    print(pd.DataFrame(complete).drop(columns='confusion_matrix').to_string(index=False));print(summary.to_string(index=False));print('Selected pruning:',best)

if __name__=='__main__':main()
