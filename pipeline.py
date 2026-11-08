"""Reproducible offline content-risk classification and launch planning.

Only tweet text enters the model. Annotator counts and original class labels
are evaluation targets, never predictors. Aggregate outputs contain no text.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import io
import json
import math
import re
import sqlite3
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
SOURCE_URL = 'https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/cb50f7e9eb855a99abc51423bf40df0e26d6fe30/data/labeled_data.csv'
SOURCE_SHA256 = 'fcb8bc7c68120ae4af04a5b9acd58585513ede11e1548ebf36a5c2040b6f6281'
SEED = 731

def normalize(text):
    text = html.unescape(text).casefold()
    text = re.sub(r'https?://\S+', ' urltoken ', text)
    text = re.sub(r'@\w+', ' usertoken ', text)
    return re.sub(r'\s+', ' ', text).strip()

def prepare(rows):
    groups = defaultdict(list)
    for row in rows:
        text = normalize(row['tweet'])
        if not text:
            raise ValueError('Empty normalized text')
        label = int(row['class'])
        if label not in (0, 1, 2):
            raise ValueError('Unexpected annotation class')
        digest = hashlib.sha256(text.encode()).hexdigest()
        groups[digest].append((text, label))
    clean, conflict_rows, duplicate_rows = [], 0, 0
    for digest, group in sorted(groups.items()):
        if len({x[1] for x in group}) != 1:
            conflict_rows += len(group)
            continue
        text, label = group[0]
        clean.append({'text_hash': digest, 'text': text, 'original_class': label,
                      'target': int(label == 0), 'source_rows': len(group)})
        duplicate_rows += len(group) - 1
    assert len(clean) + duplicate_rows + conflict_rows == len(rows)
    return clean, {'raw_rows': len(rows), 'unique_eligible_rows': len(clean),
                   'duplicate_rows_removed': duplicate_rows,
                   'conflicting_annotation_rows_quarantined': conflict_rows}

def partition(records):
    idx = np.arange(len(records))
    labels = [x['original_class'] for x in records]
    train, remainder = train_test_split(idx, test_size=.4, stratify=labels, random_state=SEED)
    valid, test = train_test_split(remainder, test_size=.5,
        stratify=[labels[i] for i in remainder], random_state=SEED)
    return {name: [records[i] for i in sorted(indices)]
            for name, indices in [('train', train), ('validation', valid), ('test', test)]}

def wilson(k, n, z=1.95996398454):
    if n == 0:
        return None
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [center-half, center+half]

def operating_metrics(labels, scores, threshold):
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0 or len(y) != len(s) or not np.isfinite(s).all():
        raise ValueError('Nonempty finite aligned labels and scores required')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Binary labels required')
    pred = s >= threshold
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    return {'n': len(y), 'positive_labels': int(y.sum()),
        'flagged': int(pred.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': tp/(tp+fp) if tp+fp else None,
        'recall': tp/(tp+fn) if tp+fn else None,
        'false_positive_rate': fp/(fp+tn) if fp+tn else None,
        'review_rate': float(pred.mean()),
        'precision_wilson_95': wilson(tp, tp+fp),
        'recall_wilson_95': wilson(tp, tp+fn),
        'false_positive_rate_wilson_95': wilson(fp, fp+tn)}

def choose_threshold(labels, scores, max_review=.2, max_fpr=.05):
    if not (0 <= max_review <= 1 and 0 <= max_fpr <= 1):
        raise ValueError('Constraints must be rates between zero and one')
    if len(set(labels)) != 2:
        raise ValueError('Both label classes required for threshold selection')
    candidates = []
    for threshold in np.append(np.linspace(0, 1, 201), 1.000001):
        m = operating_metrics(labels, scores, threshold)
        if m['review_rate'] <= max_review and m['false_positive_rate'] <= max_fpr:
            candidates.append((m['recall'], -m['false_positive_rate'], float(threshold), m))
    return max(candidates, key=lambda x: x[:3])

def launch_sample_size(baseline=.01, absolute_mde=.002, alpha=.05, power=.8):
    if not (0 < baseline < 1 and 0 < absolute_mde < baseline and 0 < alpha < 1 and .5 < power < 1):
        raise ValueError('Invalid experiment inputs')
    p1 = baseline; p2 = baseline-absolute_mde; pooled=(p1+p2)/2
    n = ((norm.ppf(1-alpha/2)*math.sqrt(2*pooled*(1-pooled)) +
          norm.ppf(power)*math.sqrt(p1*(1-p1)+p2*(1-p2))) / absolute_mde)**2
    return int(math.ceil(n))

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator=chr(10))
        writer.writeheader(); writer.writerows(rows)

def run(source_path=None):
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    cache=ROOT/'data'; cache.mkdir(exist_ok=True)
    source_path=Path(source_path) if source_path else cache/'labeled_data.csv'
    if not source_path.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            source_path.write_bytes(response.read())
    raw=source_path.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('Source hash changed')
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    records, quality=prepare(rows)
    splits=partition(records)
    x={name:[r['text'] for r in rr] for name,rr in splits.items()}
    y={name:np.array([r['target'] for r in rr]) for name,rr in splits.items()}
    vec=TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000, sublinear_tf=True)
    train_x=vec.fit_transform(x['train'])
    valid_x=vec.transform(x['validation'])
    sweep=[]; best=None
    for c in [.25, 1., 4.]:
        model=LogisticRegression(C=c, class_weight='balanced', solver='liblinear', max_iter=500, random_state=SEED)
        model.fit(train_x,y['train'])
        score=model.predict_proba(valid_x)[:,1]
        ap=float(average_precision_score(y['validation'],score))
        sweep.append({'C':c,'validation_pr_auc':ap})
        if best is None or ap>best[0]: best=(ap,c,model,score)
    _,selected_c,model,valid_score=best
    _,_,threshold,valid_op=choose_threshold(y['validation'],valid_score)
    # Test labels are first consulted after model and operating threshold are fixed.
    test_score=model.predict_proba(vec.transform(x['test']))[:,1]
    test_op=operating_metrics(y['test'],test_score,threshold)
    ap=float(average_precision_score(y['test'],test_score))
    auc=float(roc_auc_score(y['test'],test_score))
    test_op.update({'pr_auc':ap,'roc_auc':auc,'constant_score_pr_auc':float(y['test'].mean()),
                    'pr_auc_lift_over_prevalence':ap/float(y['test'].mean())})
    frontier=[]
    for cap in [.05,.1,.2,.4]:
        _,_,t,vm=choose_threshold(y['validation'],valid_score,max_review=cap,max_fpr=1)
        tm=operating_metrics(y['test'],test_score,t)
        frontier.append({'validation_capacity_cap':cap,'threshold':t,
            'validation_review_rate':vm['review_rate'],'test_review_rate':tm['review_rate'],
            'test_precision':tm['precision'],'test_recall':tm['recall']})
    slices=[]
    for slice_name,predicate in [
        ('contains_url',lambda r:'urltoken' in r['text']),
        ('no_url',lambda r:'urltoken' not in r['text']),
        ('contains_mention',lambda r:'usertoken' in r['text']),
        ('no_mention',lambda r:'usertoken' not in r['text']),
        ('short_text',lambda r:len(r['text'])<80),
        ('long_text',lambda r:len(r['text'])>=80)]:
        mask=np.array([predicate(r) for r in splits['test']])
        if mask.sum():
            m=operating_metrics(y['test'][mask],test_score[mask],threshold)
            slices.append({'slice':slice_name,**{k:m[k] for k in ['n','positive_labels','flagged','tp','fp','fn','tn','precision','recall','false_positive_rate']}})
    # SQLite stores hashes and aggregate-compatible outcomes, never source text.
    db=sqlite3.connect(cache/'safety.sqlite')
    db.executescript('DROP TABLE IF EXISTS evaluation; CREATE TABLE evaluation (text_hash TEXT PRIMARY KEY, split TEXT NOT NULL, original_class INTEGER NOT NULL, target INTEGER NOT NULL, score REAL, flagged INTEGER);')
    for name,rr in splits.items():
        score=test_score if name=='test' else (valid_score if name=='validation' else [None]*len(rr))
        db.executemany('INSERT INTO evaluation VALUES (?,?,?,?,?,?)',[
            (r['text_hash'],name,r['original_class'],r['target'],None if s is None else float(s),None if s is None else int(s>=threshold))
            for r,s in zip(rr,score)])
    db.commit()
    sql=(ROOT/'sql/metrics.sql').read_text()
    sql_rows=db.execute(sql).fetchall()
    test_sql=next(r for r in sql_rows if r[0]=='test')
    sets={n:{r['text_hash'] for r in rr} for n,rr in splits.items()}
    checks={
      'raw_row_conservation':quality['raw_rows']==quality['unique_eligible_rows']+quality['duplicate_rows_removed']+quality['conflicting_annotation_rows_quarantined'],
      'partition_row_conservation':sum(map(len,splits.values()))==len(records),
      'train_validation_disjoint':not(sets['train']&sets['validation']),
      'train_test_disjoint':not(sets['train']&sets['test']),
      'validation_test_disjoint':not(sets['validation']&sets['test']),
      'both_classes_each_partition':all(len(set(v))==2 for v in y.values()),
      'sqlite_integrity':db.execute('PRAGMA integrity_check').fetchone()[0]=='ok',
      'sql_test_count_parity':test_sql[1]==test_op['n'],
      'sql_true_positive_parity':test_sql[3]==test_op['tp'],
      'sql_false_positive_parity':test_sql[4]==test_op['fp'],
      'validation_review_constraint':valid_op['review_rate']<=.2,
      'validation_fpr_constraint':valid_op['false_positive_rate']<=.05}
    db.close()
    if not all(checks.values()): raise AssertionError(checks)
    report={'source':{'url':SOURCE_URL,'sha256':digest,'original_class_counts':dict(Counter(r['class'] for r in rows))},
      'quality':quality,'split_sizes':{k:len(v) for k,v in splits.items()},
      'model':{'type':'TF-IDF word unigram/bigram plus class-weighted logistic regression','C':selected_c,
               'vocabulary_size':len(vec.vocabulary_),'seed':SEED,'selection':'validation PR-AUC',
               'threshold_selection':'maximize validation recall at review rate <=20% and FPR <=5%',
               'threshold':threshold,'features':['normalized tweet text only'],
               'scores_are_calibrated_probabilities':False},
      'validation':valid_op,'test':test_op,
      'launch_plan':{'assumed_baseline_harm_rate':.01,'absolute_mde':.002,'alpha':.05,'power':.8,
                      'required_users_per_arm':launch_sample_size()},
      'checks':checks,'checks_passed':sum(checks.values())}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    write_csv(out/'model_selection.csv',sweep)
    write_csv(out/'capacity_frontier.csv',frontier)
    write_csv(out/'slice_metrics.csv',slices)
    write_csv(out/'sql_metrics.csv',[dict(zip(['split','n','positives','tp','fp','flagged'],r)) for r in sql_rows])
    print(json.dumps({'quality':quality,'test':test_op,'threshold':threshold,'checks_passed':sum(checks.values()),'launch':report['launch_plan']},indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path)
    run(parser.parse_args().source)
