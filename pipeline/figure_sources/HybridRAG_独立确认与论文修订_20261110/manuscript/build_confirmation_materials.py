from pathlib import Path
import csv, hashlib, json, shutil
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
SOURCE=WORK/'HybridRAG_预冻结确认评估_20261110'
DEST=ROOT/'LaTeX'
DATA=DEST/'data/confirmation';DATA.mkdir(parents=True,exist_ok=True)
AUDIT=ROOT.parent/'analysis/confirmation_reanalysis.json'
audit=json.loads(AUDIT.read_text())
assert audit['status']=='PASS'
for name in ['results/summary.json','frozen/protocol.json']:
    p=SOURCE/name
    key='source_summary_sha256' if name.startswith('results') else 'source_protocol_sha256'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==audit[key]
    shutil.copy2(p,DATA/p.name)
shutil.copy2(AUDIT,DATA/AUDIT.name)
summary=json.loads((DATA/'summary.json').read_text())
DS=[('2wiki','2Wiki'),('hotpotqa','HotpotQA'),('musique','MuSiQue')]

# Six-row primary table; all figures are derived from the independently audited summary.
head=r'''\begin{table*}[t]
\caption{\rev{Prespecified Top5 confirmation outcomes, $N=1{,}000$ per corpus. $P$ is complete-pool coverage; before and after are complete Top5 coverage. $W$ and $L$ count rescued failures and newly lost successes. All percentages use all 1,000 questions; $\Delta$ and its descriptive 95\% paired-bootstrap interval are percentage points. A's after column is C. The two new-question cohorts retain the existing models; MuSiQue uses fresh training with group-isolated internal confirmation.}}
\label{tab_confirmation}
\centering\small
\setlength{\tabcolsep}{5pt}
{\color{revisionink}
\begin{tabular}{llrrrrrl}
\toprule
Corpus & Pool & $P$ (\%) & Before (\%) & After (\%) & $W$ & $L$ & $\Delta$ [95\% interval]\\
\midrule
'''
lines=[]
rows=[]
for ds,label in DS:
    for pol in ['S2','A']:
        o=summary[ds][pol];v=o['K5'];ci=[100*x for x in o['K5_bootstrap_95']['delta_rate']]
        assert v['n']==1000 and v['after']-v['before']==v['wins']-v['losses']
        values=[100*v[k] for k in ['pool_rate','before_rate','after_rate','delta_rate']]
        lines.append(f"{label} & {pol} & {values[0]:.1f} & {values[1]:.1f} & {values[2]:.1f} & {v['wins']} & {v['losses']} & $+{values[3]:.1f}$ [{ci[0]:.2f}, {ci[1]:.2f}]\\\\")
        rows.append({'dataset':ds,'pool':pol,**v,'delta_ci_low_pp':ci[0],'delta_ci_high_pp':ci[1]})
(DEST/'table_confirmation.tex').write_text(head+'\n'.join(lines)+r'''
\bottomrule
\end{tabular}}
\end{table*}
''')
with (DATA/'primary_K5.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

secondary=[2,10,20,30,40,50]
app=r'''\section{Confirmation Outcomes at All Prespecified Secondary Cutoffs}
\label{app_confirmation}
\rev{These tables use the separately fixed 1,000-question confirmation cohorts, not the earlier 300-question development cohorts. Table~\ref{tab_confirmation_secondary} reports all six prespecified secondary cutoffs. The full $K=1$--50 per-question and aggregate records accompany the source package; $K=5$ is the primary endpoint in Table~\ref{tab_confirmation}.}

\begin{table}[H]
\caption{\rev{Complete-support coverage before $\rightarrow$ after reranking (\%), $N=1{,}000$ in every row, at every prespecified secondary cutoff. A after reranking is C. Each comparison retains its own identical pool; values between the listed cutoffs appear in Figure~\ref{fig_confirmation}.}}
\label{tab_confirmation_secondary}
\centering\small
\setlength{\tabcolsep}{4pt}
{\color{revisionink}
\begin{tabular}{llrrrrrr}
\toprule
Corpus & Pool & $K=2$ & $K=10$ & $K=20$ & $K=30$ & $K=40$ & $K=50$\\
\midrule
'''
for ds,label in DS:
    for pol in ['S2','A']:
        idx={v['K']:v for v in summary[ds][pol]['K_grid']}
        cells=[f"${100*idx[k]['before_rate']:.1f}\\!\\rightarrow\\!{100*idx[k]['after_rate']:.1f}$" for k in secondary]
        app+=label+' & '+pol+' & '+' & '.join(cells)+'\\\\\n'
app+=r'''\bottomrule
\end{tabular}}
\end{table}

\begin{table}[H]
\caption{\rev{Confirmation Top5 counts split by whether supplementary retrieval was executed. $n_P$ counts complete pools; $f_0,f_1$ count complete lists before and after reranking. Undefined rates in empty strata are not replaced by zero rates. This table reports counts, retaining the small skipped strata without treating them as stable subgroup estimates.}}
\label{tab_confirmation_execution}
\centering\small
\setlength{\tabcolsep}{6pt}
{\color{revisionink}
\begin{tabular}{lllrrrrrr}
\toprule
Corpus & Pool & Supplement & $n$ & $n_P$ & $f_0$ & $f_1$ & $W$ & $L$\\
\midrule
'''
for ds,label in DS:
    for pol in ['S2','A']:
        for status in ['executed','skipped']:
            v=summary[ds][pol]['execution_strata'][status]
            app+=label+' & '+pol+' & '+status+' & '+' & '.join(str(v[k]) for k in ['n','pool_complete','before','after','wins','losses'])+'\\\\\n'
app+=r'''\bottomrule
\end{tabular}}
\end{table}

\rev{All six before/after comparisons have positive net differences at the prespecified secondary cutoffs. The complete curve retains the one intervening negative result: MuSiQue A/C has one fewer complete question at $K=39$, a difference of $-0.1$ percentage points. This observation prevents interpreting positive Top5 confirmation as uniform improvement at every output capacity.}
'''
(DEST/'appendix_confirmation.tex').write_text(app)

# Vector figure, final canvas sized to the ACM double-column width.
font='/System/Library/Fonts/Supplemental/Times New Roman.ttf'
pdfmetrics.registerFont(TTFont('FigureSerif',font))
W,H=510,340
out=DEST/'figures/confirmation_fixed_pool_K1_50.pdf'
c=canvas.Canvas(str(out),pagesize=(W,H),invariant=1,initialFontName='FigureSerif')
c.setTitle('Confirmation complete-support coverage over output capacity')
c.setAuthor('')
ink=HexColor('#18242f');grid=HexColor('#dce3e9');axis=HexColor('#64727d')
blue=HexColor('#0072b2');orange=HexColor('#d55e00');green=HexColor('#009e73')
styles=[('Before',blue,[]),('Frozen reranking',orange,[4,2]),('Shared pool ceiling',green,[1.5,2])]
for x,(label,color,dash) in zip([58,209,371],styles):
    c.setStrokeColor(color);c.setLineWidth(1.2);c.setDash(dash);c.line(x,326,x+20,326);c.setDash();c.setFont('FigureSerif',9);c.setFillColor(ink);c.drawString(x+25,323,label)
plotdata={'source_summary_sha256':hashlib.sha256((DATA/'summary.json').read_bytes()).hexdigest(),'K_grid':list(range(1,51)),'x_scale':'linear','y_limits':[0,100],'primary_K':5,'panels':[]}
curve_rows=[]
for row,pol in enumerate(['S2','A']):
    y0,y1=(198,282) if row==0 else (42,126)
    c.setFont('FigureSerif',9.5);c.setFillColor(ink);c.drawString(44,y1+29,'(a) S2 candidate pool' if pol=='S2' else '(b) A candidate pool; reranked A is C')
    c.saveState();c.translate(11,(y0+y1)/2);c.rotate(90);c.setFont('FigureSerif',9);c.drawCentredString(0,0,'Complete-support coverage (%)');c.restoreState()
    for col,(ds,label) in enumerate(DS):
        x0=44+col*163;x1=x0+130
        xx=lambda k:x0+(k-1)/49*130
        yy=lambda v:y0+v/100*(y1-y0)
        c.setFillColor(ink);c.setFont('FigureSerif',9.5);c.drawCentredString((x0+x1)/2,y1+13,label)
        for y in range(0,101,20):
            c.setStrokeColor(grid);c.setLineWidth(.35);c.line(x0,yy(y),x1,yy(y));c.setFillColor(axis);c.setFont('FigureSerif',8.5);c.drawRightString(x0-5,yy(y)-2.8,str(y))
        c.setStrokeColor(axis);c.setLineWidth(.5);c.line(x0,y0,x1,y0)
        for k in [1,5,10,20,30,40,50]:
            c.line(xx(k),y0,xx(k),y0-2.5);c.setFillColor(axis);c.setFont('FigureSerif',8.5);c.drawCentredString(xx(k),y0-13,str(k))
        c.setFont('FigureSerif',9);c.setFillColor(ink);c.drawCentredString((x0+x1)/2,y0-27,'Output capacity K')
        c.setStrokeColor(HexColor('#9aa5ae'));c.setDash([2,3]);c.setLineWidth(.5);c.line(xx(5),y0,xx(5),y1);c.setDash()
        vals=summary[ds][pol]['K_grid'];assert [v['K'] for v in vals]==list(range(1,51))
        pool=100*summary[ds][pol]['K5']['pool_rate']
        c.setStrokeColor(green);c.setDash([1.5,2]);c.setLineWidth(1.1);c.line(x0,yy(pool),x1,yy(pool));c.setDash()
        for key,color,dash in [('before_rate',blue,[]),('after_rate',orange,[4,2])]:
            c.setStrokeColor(color);c.setLineWidth(1.3);c.setDash(dash);p=c.beginPath()
            for j,v in enumerate(vals):
                (p.moveTo if j==0 else p.lineTo)(xx(v['K']),yy(100*v[key]))
            c.drawPath(p);c.setDash();c.setFillColor(color);c.circle(xx(5),yy(100*vals[4][key]),1.8,stroke=0,fill=1)
        plotdata['panels'].append({'dataset':ds,'pool':pol,'pool_pct':pool,'before_pct':[100*v['before_rate'] for v in vals],'after_pct':[100*v['after_rate'] for v in vals]})
        curve_rows.extend({'dataset':ds,'pool':pol,**v} for v in vals)
c.showPage();c.save()
(DATA/'figure_plot_data.json').write_text(json.dumps(plotdata,ensure_ascii=False,indent=2)+'\n')
with (DATA/'all_K1_50.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(curve_rows[0]));w.writeheader();w.writerows(curve_rows)
manifest={'source':str(SOURCE),'source_protocol_sha256':audit['source_protocol_sha256'],'source_summary_sha256':audit['source_summary_sha256'],'independent_reanalysis':str(AUDIT),'new_figure':str(out.relative_to(ROOT)),'pdf_vector':True,'font_min_pt':8.5,'canvas_points':[W,H],'all_y_limits':[0,100],'new_tables':['table_confirmation.tex','appendix_confirmation.tex'],'primary_K':5,'new_K_range':[1,50]}
(ROOT/'confirmation_materials_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'BUILT','figure':str(out),'primary_rows':len(rows),'curve_rows':len(curve_rows)}))
