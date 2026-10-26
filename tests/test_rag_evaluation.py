"""Métricas e avaliação humana com dados sintéticos; não atribui notas reais."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from ri_project.rag_evaluation import (
    CRITERIA, RAG_QUERY_IDS, RANKING_FILES, aggregate_human_evaluation,
    agreement_statistics, context_metrics, load_evaluation_artifacts,
    prepare_rag_evaluation, read_human_ratings,
)
from ri_project.rag_analysis import finalize_failure_analysis, select_analysis_cases
from ri_project.rag_experiment import export_standardized_csv


class RAGEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.data = root / 'data'; self.data.mkdir()
        (self.data / 'qrels').mkdir()
        self.experiment = root / 'experiment'; self.experiment.mkdir()
        self.rankings = root / 'rankings'; self.rankings.mkdir()
        self.output = self.experiment / 'evaluation'
        documents = [{'_id': f'D{i:03}', 'title': f'Title {i}', 'text': f'Original evidence D{i:03}.'}
                     for i in range(101)]
        self.docs = {d['_id']: d for d in documents}
        (self.data / 'corpus.jsonl').write_text('\n'.join(json.dumps(d) for d in documents))
        (self.data / 'queries.jsonl').write_text('\n'.join(json.dumps({'_id': q, 'text': q + '?'}) for q in RAG_QUERY_IDS))
        for split in ('train', 'dev', 'test'):
            with (self.data / 'qrels' / f'{split}.tsv').open('w', newline='') as file:
                w = csv.writer(file, delimiter='\t');w.writerow(['query-id','corpus-id','score'])
                if split == 'test':
                    w.writerows((q,d,grade) for q in RAG_QUERY_IDS for d,grade in [('D000',2),('D005',1),('D100',1)])
        before = [f'D{i:03}' for i in range(100)]
        after = list(before);after[0],after[5] = after[5],after[0]
        self.answers = [];cases = []
        rank_rows = {k: [] for k in RANKING_FILES}
        for qid in RAG_QUERY_IDS:
            contexts = [{'doc_id': d, 'title': self.docs[d]['title'], 'text': self.docs[d]['text']}
                        for d in after[:5]]
            cases.append({'query_id':qid,'query':qid+'?','contexts':contexts,
                          'reranked_top100':[{'doc_id':d,'score':1.0} for d in after]})
            for stage in rank_rows:
                rank_rows[stage].extend({'query_id':qid,'doc_id':d,'rank':i,'score':1.0}
                                       for i,d in enumerate(after if stage=='reranking' else before,1))
            for condition in ('A_without_rag','B_with_rag'):
                sent = condition.startswith('B')
                self.answers.append({'query_id':qid,'query':qid+'?','condition':condition,
                                     'contexts':contexts if sent else [],'external_context_provided':sent,
                                     'manifest':{'model':'synthetic'},'generation_kwargs':{'do_sample':False},
                                     'experiment_signature':'fixture','prompt':condition+qid,
                                     'answer':'Synthetic answer '+qid+condition})
        bundle = {'query_ids':list(RAG_QUERY_IDS),'cases':cases,'retrieval':{'rrf_k':10},'dataset_sha256':{
            name:hashlib.sha256((self.data/name).read_bytes()).hexdigest()
            for name in ('corpus.jsonl','queries.jsonl','qrels/dev.tsv','qrels/test.tsv')}}
        (self.experiment/'contexts.json').write_text(json.dumps(bundle))
        (self.experiment/'answers.jsonl').write_text('\n'.join(json.dumps(a) for a in self.answers))
        (self.experiment/'run_status.json').write_text(json.dumps({
            'status':'complete','completed_responses':20,'experiment_signature':'fixture'}))
        export_standardized_csv(self.answers,self.experiment/'rag_experiment.csv')
        for stage,filename in RANKING_FILES.items(): pd.DataFrame(rank_rows[stage]).to_csv(self.rankings/filename,index=False)

    def prepare(self):
        return prepare_rag_evaluation(self.experiment,self.data,self.rankings,self.output)

    def fill(self,number=1,uniform=False):
        path=self.output/f'ratings_reviewer_{number}.csv'
        with path.open() as f: reader=csv.DictReader(f);rows=list(reader);fields=reader.fieldnames
        key=json.loads((self.output/'blind_key.json').read_text())
        lookup={r['answer_id']:r for r in key['items']}
        for row in rows:
            item=lookup[row['answer_id']];value=1
            if not uniform and item['query_id']==RAG_QUERY_IDS[0]: value=1 if item['condition']=='A' else 2
            if not uniform and item['query_id']==RAG_QUERY_IDS[1]: value=2 if item['condition']=='A' else 0
            for c in CRITERIA:
                row[c]=str(value);row[f'justificativa_{c}']='Synthetic justification; fixture only.'
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        return path,rows,fields

    def test_context_metrics_use_total_relevant_and_reject_duplicates(self):
        m=context_metrics(['A','B','C','D','E'],{'A':2,'D':1,'X':1,'Y':1})
        self.assertEqual((m['P5'],m['R5'],m['Hit5']),(.4,.5,1))
        self.assertEqual(context_metrics(['A','B','C','D','E'],{})['R5'],0)
        with self.assertRaises(ValueError):context_metrics(['A']*5,{})

    def test_prepare_binds_sources_and_preserves_human_work(self):
        result=self.prepare()
        self.assertAlmostEqual(result['context_summary'].iloc[0].R5,1/3)
        docs=result['diagnostics']['documents']
        missing=docs[(docs.query_id==RAG_QUERY_IDS[0])&(docs.doc_id=='D100')].iloc[0]
        self.assertTrue(pd.isna(missing.lexical_rank));self.assertFalse(missing.in_context)
        path,_,_=self.fill();before=path.read_bytes();self.prepare()
        self.assertEqual(path.read_bytes(),before)
        with path.open() as f: fields=csv.DictReader(f).fieldnames
        self.assertNotIn('condition',fields);self.assertNotIn('query_id',fields)
        text=(self.output/'human_review.md').read_text()
        self.assertEqual(text.count('Original evidence D005.'),20)
        ranking_path=self.rankings/'results_hybrid.csv';original=ranking_path.read_bytes()
        bad=pd.read_csv(ranking_path)
        bad.loc[0,'doc_id'],bad.loc[1,'doc_id']=bad.loc[1,'doc_id'],bad.loc[0,'doc_id']
        bad.to_csv(ranking_path,index=False)
        with self.assertRaisesRegex(ValueError,'RRF e baselines'):self.prepare()
        ranking_path.write_bytes(original)
        changed=list(self.answers);changed[0]={**changed[0],'answer':'Different output'}
        (self.experiment/'answers.jsonl').write_text('\n'.join(json.dumps(r) for r in changed))
        with self.assertRaisesRegex(ValueError,'mudaram'):self.prepare()

    def test_pending_notes_are_not_zero_or_final_cases(self):
        self.prepare()
        result=aggregate_human_evaluation([self.output/'ratings_reviewer_1.csv'],self.experiment,self.data,self.output)
        self.assertEqual(result['status']['status'],'pending_human_ratings')
        self.assertTrue(result['summary']['mean'].isna().all())
        self.assertEqual(result['summary'].rated_count.sum(),0)
        p=finalize_failure_analysis(self.experiment,self.data,self.output)
        self.assertEqual(json.loads(p.read_text())['status'],'pending_human_ratings')
        with (self.experiment/'rag_experiment.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertTrue(all(not r['fidelidade'] for r in rows))

    def test_invalid_scores_justifications_and_na_are_rejected(self):
        self.prepare();path,rows,fields=self.fill()
        key=json.loads((self.output/'blind_key.json').read_text())
        b=next(i for i,r in enumerate(rows) if next(x for x in key['items'] if x['answer_id']==r['answer_id'])['condition']=='B')
        for grade,reason in [('3','why'),('1',''),('N/A','why')]:
            changed=[dict(r) for r in rows];changed[b]['fidelidade']=grade;changed[b]['justificativa_fidelidade']=reason
            with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(changed)
            with self.assertRaises(ValueError):read_human_ratings([path],key)

    def test_human_consolidation_excludes_na_and_keeps_integer_notes(self):
        self.prepare();path,_,_=self.fill()
        result=aggregate_human_evaluation([path],self.experiment,self.data,self.output)
        self.assertEqual(result['status']['status'],'complete')
        na=result['summary'][(result['summary'].condition=='A')&(result['summary'].criterion=='rastreabilidade')].iloc[0]
        self.assertFalse(na.applicable);self.assertEqual(na.rated_count,0);self.assertTrue(pd.isna(na['mean']))
        with (self.experiment/'rag_experiment.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertTrue(all(r['rastreabilidade']=='N/A' for r in rows if r['condition']=='A'))
        self.assertTrue(all(r['fidelidade'] in ('0','1','2') for r in rows))
        with (self.output/'human_scores_per_answer.csv').open() as f:raw=list(csv.DictReader(f))
        self.assertTrue(all(r['rastreabilidade']=='N/A' for r in raw if r['condition']=='A'))
        cases=json.loads((self.output/'failure_case_selection.json').read_text())
        self.assertEqual(len({r['query_id'] for r in cases['cases']}),3)
        # Repeated consolidation must also work when a CSV has integer columns.
        aggregate_human_evaluation([path],self.experiment,self.data,self.output)

    def test_two_reviewers_require_explicit_final_choice_and_agreement(self):
        self.prepare();p1,_,_=self.fill();p2,_,_=self.fill(2,uniform=True)
        r=aggregate_human_evaluation([p1,p2],self.experiment,self.data,self.output)
        self.assertEqual(r['status']['status'],'needs_final_reviewer')
        self.assertEqual(len(r['agreement']),7)
        r=aggregate_human_evaluation([p1,p2],self.experiment,self.data,self.output,'avaliador_1')
        self.assertEqual(r['status']['status'],'complete')
        a=agreement_statistics([0,2],[2,0]);self.assertEqual(a['quadratic_kappa'],-1)
        self.assertIsNone(agreement_statistics([1,1],[1,1])['quadratic_kappa'])

    def test_final_case_analysis_requires_verified_quotes(self):
        self.prepare();path,_,_=self.fill()
        aggregate_human_evaluation([path],self.experiment,self.data,self.output)
        review_path=self.output/'case_reviews.json';review=json.loads(review_path.read_text())
        review['reviewer_id']='synthetic reviewer'
        for r in review['cases']:
            r.update(selection_reason='Synthetic selection',comparison='Synthetic comparison',
                     information_in_context='yes',primary_stage='generation',stage_reason='Synthetic reason',
                     evidence=[{'doc_id':'D005','quote':'Original evidence D005.'}])
        review_path.write_text(json.dumps(review))
        final=finalize_failure_analysis(self.experiment,self.data,self.output)
        self.assertEqual(final.name,'failure_analysis.md')
        self.assertIn('Original evidence D005.',final.read_text())
        review['cases'][0]['evidence'][0]['quote']='Fabricated text'
        review_path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError,'Trecho inexistente'):finalize_failure_analysis(self.experiment,self.data,self.output)

    def test_case_selection_does_not_invent_harm_when_all_improve(self):
        self.prepare();path,rows,fields=self.fill(uniform=True)
        key=json.loads((self.output/'blind_key.json').read_text());lookup={x['answer_id']:x for x in key['items']}
        for r in rows:
            for c in CRITERIA:r[c]='0' if lookup[r['answer_id']]['condition']=='A' else '2'
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        selection=select_analysis_cases(read_human_ratings([path],key))
        self.assertEqual(selection['status'],'categories_not_observed')
        self.assertIn('nao_ajudou',selection['missing_categories'])

    def test_opposing_large_changes_are_not_treated_as_little_difference(self):
        records=[]
        for i,qid in enumerate(RAG_QUERY_IDS):
            for condition in ('A','B'):
                grade=1
                if i==0:grade=1 if condition=='A' else 2
                if i==1:grade=2 if condition=='A' else 0
                row={'query_id':qid,'condition':condition,
                     'fidelidade':grade,'completude':grade,'utilidade':grade}
                if i==2:
                    row['fidelidade']=0 if condition=='A' else 2
                    row['utilidade']=2 if condition=='A' else 0
                records.append(row)
        selection=select_analysis_cases(records)
        neutral=next(r for r in selection['cases'] if r['category']=='pouca_diferenca')
        self.assertNotEqual(neutral['query_id'],RAG_QUERY_IDS[2])


if __name__=='__main__':unittest.main()
