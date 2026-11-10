"""Publication-width vector figures from previously verified aggregate results."""
from pathlib import Path
import csv,hashlib,json,math,shutil
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from pypdf import PdfReader
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'HybridRAG_K全池延伸_20261109'
OUT=ROOT/'figures';OUT.mkdir(parents=True,exist_ok=True)
DEST=ROOT/'manuscript/LaTeX/figures';DEST.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('PaperFigure','/System/Library/Fonts/Supplemental/Times New Roman.ttf'))
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
srcs=[PRIOR/'primary/results/all_methods_K1_400.csv',PRIOR/'primary/results/paired_contrasts_K1_400.csv',PRIOR/'development/results/curves_K1_400.csv',PRIOR/'development/results/pairs_K1_400.csv']
m={(r['dataset'],r['method'],int(r['K'])):r for r in read(srcs[0])}
p={(r['dataset'],r['method'],r['baseline'],int(r['K'])):r for r in read(srcs[1])}
d={(r['dataset'],r['pool_method'],r['variant'],int(r['K'])):r for r in read(srcs[2]) if r['stratum']=='all'}
q={(r['dataset'],r['pool_method'],int(r['K'])):r for r in read(srcs[3]) if r['stratum']=='all'}
DS=[('2wiki','2Wiki',12376),('hotpotqa','HotpotQA',7204),('musique','MuSiQue',2311)]
S='no_second_branch';I='idf_m3_head180_w8';R='hybrid_rrf';T='repeat_q0_s2mask'
COLORS=['#0072B2','#D55E00','#009E73'];DASH=[[],[4,2],[1.4,1.7]]
figs=[]
def row(label,kind,items,ks,limits,log=True):
    panels=[];ks=list(ks)
    for j,(ds,title,n) in enumerate(DS):
        ss=[]
        for item in items:
            if kind=='primary_pair':
                vals=[float(p[ds,item[0],item[1],k]['delta_pp']) for k in ks]
            elif kind=='primary_absolute':
                vals=[float(m[ds,item[0],k]['support_complete_pct']) for k in ks]
            elif kind=='development_pair':
                vals=[float(q[ds,item[0],k]['delta_pct_points']) for k in ks]
            else:
                if item[1]=='pool':vals=[100*int(d[ds,item[0],'before',k]['pool_complete_n'])/300 for k in ks]
                else:vals=[float(d[ds,item[0],item[1],k]['pct']) for k in ks]
            ss.append({'source_key':list(item),'values':vals})
        panels.append({'dataset':ds,'title':title,'N':300 if kind.startswith('development') else n,'ylim':limits[j][0],'yticks':limits[j][1],'series':ss})
    return {'label':label,'kind':kind,'K':ks,'log_x':log,'panels':panels}
net_limits=[((-10,45),[-10,0,10,20,30,40]),((-15,15),[-15,-10,-5,0,5,10,15]),((-5,20),[-5,0,5,10,15,20])]
full=list(range(1,401))
def add(name,labels,rows,ylabel):figs.append({'name':name,'labels':labels,'rows':rows,'ylabel':ylabel})
add('fullk_net_gain',['S2 - RRF','IDF8 - RRF','IDF8 - S2'],[
    row('(a) Small-capacity window: K = 1-50','primary_pair',[(S,R),(I,R),(I,S)],range(1,51),net_limits),
    row('(b) Complete-pool range: K = 1-400','primary_pair',[(S,R),(I,R),(I,S)],full,net_limits)],'Complete-coverage difference (pp)')
add('matched_query_fullk',['S2 - matched Repeat','Matched Repeat - RRF'],[
    row('','primary_pair',[(S,T),(T,R)],full,[((-10,45),[-10,0,10,20,30,40]),((-15,10),[-15,-10,-5,0,5,10]),((-10,15),[-10,-5,0,5,10,15])])],'Complete-coverage difference (pp)')
add('rm_depth_fullk',['RM8 m20 - m3'],[
    row('','primary_pair',[('rm_m20_head180_w8','rm_m3_head180_w8')],full,[((-14,1),[-12,-9,-6,-3,0])]*3)],'Complete-coverage difference (pp)')
covlimits=[((0,100),[0,20,40,60,80,100])]*3
add('fixed_pool_coverage_fullk',['Before','Frozen reranking','Shared pool ceiling'],[
    row('(a) S2 candidate pool','development_absolute',[('S2','before'),('S2','after'),('S2','pool')],full,covlimits),
    row('(b) A candidate pool; reranked A is C','development_absolute',[('A','before'),('A','after'),('A','pool')],full,covlimits)],'Complete support coverage (%)')
add('absolute_coverage_fullk',['RRF','S2','IDF8'],[
    row('','primary_absolute',[(R,),(S,),(I,)],full,covlimits)],'Complete support coverage (%)')
add('fixed_pool_net_and_tail',['S2: after - before','A/C: after - before'],[
    row('(a) Complete-pool range: K = 1-400','development_pair',[('S2',),('A',)],full,[((-1,30),[0,5,10,15,20,25,30]),((-1,30),[0,5,10,15,20,25,30]),((-1.2,6),[-1,0,1,2,3,4,5,6])]),
    row('(b) Later cutoffs: K = 51-400','development_pair',[('S2',),('A',)],range(51,401),[((-1,4),[-1,0,1,2,3,4])]*3,False)],'Complete-coverage difference (pp)')

W=510;audit={'figures':[],'source_tables':{str(s.relative_to(ROOT.parent)):hashlib.sha256(s.read_bytes()).hexdigest() for s in srcs}}
for sp in figs:
    double=len(sp['rows'])==2;H=430 if double else 228
    path=OUT/(sp['name']+'.pdf')
    c=canvas.Canvas(str(path),pagesize=(W,H),invariant=1,initialFontName='PaperFigure')
    c.setTitle(sp['name']);c.setAuthor('')
    ink,axis,grid=[HexColor(x) for x in ['#17222D','#56616B','#DDE2E7']]
    ly=416 if double else 216
    lx=[180] if len(sp['labels'])==1 else ([56,290] if len(sp['labels'])==2 else [44,204,359])
    for i,(label,x) in enumerate(zip(sp['labels'],lx)):
        c.setStrokeColor(HexColor(COLORS[i]));c.setLineWidth(1.25);c.setDash(DASH[i]);c.line(x,ly,x+17,ly);c.setDash()
        c.setFillColor(ink);c.setFont('PaperFigure',8.8);c.drawString(x+22,ly-2.8,label)
    count=0
    for ri,rr in enumerate(sp['rows']):
        if double:
            top,bottom,header,ptitle=(360,235,393,377) if ri==0 else (162,37,195,179)
            c.setFillColor(ink);c.setFont('PaperFigure',9.4);c.drawString(43,header,rr['label'])
        else:top,bottom,ptitle=178,35,195
        ks=rr['K']
        for pa,x0 in zip(rr['panels'],[43,205,367]):
            x1=x0+128;low,high=pa['ylim']
            if rr['log_x']:xx=lambda k:x0+math.log(k/ks[0])/math.log(ks[-1]/ks[0])*128
            else:xx=lambda k:x0+(k-ks[0])/(ks[-1]-ks[0])*128
            yy=lambda v:bottom+(v-low)/(high-low)*(top-bottom)
            c.setFillColor(ink);c.setFont('PaperFigure',9.5);c.drawCentredString((x0+x1)/2,ptitle,pa['title'])
            c.setFont('PaperFigure',8.4)
            for t in pa['yticks']:
                c.setStrokeColor(axis if t==0 else grid);c.setLineWidth(.65 if t==0 else .35);c.line(x0,yy(t),x1,yy(t))
                c.setFillColor(axis);c.drawRightString(x0-4,yy(t)-2.7,str(t))
            c.setStrokeColor(axis);c.setLineWidth(.5);c.line(x0,bottom,x1,bottom)
            if not rr['log_x']:ticks=[51,100,200,300,400]
            elif ks[-1]==50:ticks=[1,2,5,10,20,50]
            else:ticks=[1,2,5,10,50,100,400]
            for k in ticks:
                c.line(xx(k),bottom,xx(k),bottom-3);c.setFillColor(axis);c.drawCentredString(xx(k),bottom-13,str(k))
            if ks[-1]==400 and rr['log_x']:
                c.setStrokeColor(HexColor('#A7AFB6'));c.setDash([2,3]);c.line(xx(50),bottom,xx(50),top);c.setDash()
            for si,s in enumerate(pa['series']):
                vs=s['values'];assert len(vs)==len(ks) and all(math.isfinite(v) and low<=v<=high for v in vs)
                c.setStrokeColor(HexColor(COLORS[si]));c.setLineWidth(1.2);c.setDash(DASH[si])
                line=c.beginPath();line.moveTo(xx(ks[0]),yy(vs[0]))
                for k,v in zip(ks[1:],vs[1:]):line.lineTo(xx(k),yy(v))
                c.drawPath(line,stroke=1,fill=0);c.setDash();count+=len(vs)
            c.setFillColor(ink);c.setFont('PaperFigure',8.6);c.drawCentredString((x0+x1)/2,bottom-29,'K (log scale)' if rr['log_x'] else 'Output cutoff K')
        c.saveState();c.translate(10,(bottom+top)/2);c.rotate(90);c.setFillColor(ink);c.setFont('PaperFigure',8.5);c.drawCentredString(0,0,sp['ylabel']);c.restoreState()
    c.save()
    pdf=PdfReader(path);res=pdf.pages[0]['/Resources']
    assert not [v for v in res.get('/XObject',{}).values() if v.get_object().get('/Subtype')=='/Image']
    imagepdf=pdfium.PdfDocument(str(path));page=imagepdf[0];page.render(scale=2.5).to_pil().save(path.with_suffix('.png'));page.close();imagepdf.close()
    shutil.copy2(path,DEST/path.name)
    audit['figures'].append({'name':sp['name'],'width_pt':W,'height_pt':H,'minimum_font_pt':8.4,'plotted_points':count,'vector':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'visual_review':'pending'})
(OUT/'plot_data.json').write_text(json.dumps({'sources':audit['source_tables'],'figures':figs},ensure_ascii=False,indent=2)+'\n')
(OUT/'figure_QA.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'figures':len(figs),'points':sum(f['plotted_points'] for f in audit['figures']),'copied_to':str(DEST)}))
