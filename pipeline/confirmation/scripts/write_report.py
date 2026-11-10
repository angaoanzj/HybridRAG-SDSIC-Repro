from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'results/summary.json').read_text())
audit=json.loads((ROOT/'verification/independent_recomputation.json').read_text())
assert audit['status']=='PASS'
names={'2wiki':'2Wiki','hotpotqa':'HotpotQA','musique':'MuSiQue'}
def pct(x):return '—' if x is None else f'{100*x:.2f}%'
def pp(x):return '—' if x is None else f'{100*x:+.2f}'
def interval(x):return f'[{100*x[0]:+.2f}, {100*x[1]:+.2f}]'
lines=['# 三库预冻结确认评估结果','',
'状态：三库各 1,000 题、四个条件均已完成，独立排序及指标复算通过。2Wiki 和 HotpotQA 沿用原模型；MuSiQue 排除确认关联组及开发集关联缓冲题后，从头拟合新文档排序器。','',
'## 划分与训练','',
'| 数据集 | 确认集 | 训练与选参 | 结果范围 |','|---|---:|---|---|',
'| 2Wiki | 1,000 题 | 原模型不变，权重 0.5 | 经历史筛查、固定语料支持可达的新题 |',
'| HotpotQA | 1,000 题 | 原模型不变，权重 0.5 | 经历史筛查、固定语料支持可达的 hard 新题 |',
'| MuSiQue | 213 个完整关联组，1,000 题 | 新训练 20,990 题；原 300 题选权重；缓冲排除 3 题 | 重训后按组隔离的内部确认 |','',
'冻结种子 2026111001。前两库在题型及支持文档数层内按比例抽取，MuSiQue 按种子产生关联组优先级、整体选入至恰好 1,000 题。名单与协议在本次训练/评估前保存，未根据结果重抽或改参。MuSiQue 确认题具有历史训练暴露，因此不称项目从未见过的独立外部测试集。','',
'确认集支持文档数构成为：2Wiki 两篇 710 题、四篇 290 题；HotpotQA 全部为两篇；MuSiQue 两篇 504 题、三篇 256 题、四篇 240 题。MuSiQue 未为匹配旧开发集的支持文档数分布而拆组或再抽样。','',
'## K=5 主结果','',
'S2 为纯标题桥接 `no_second_branch`；A 为 P1.1 固定规则 `mix_ranked16`。S2＋重排与 S2 共用候选池，C 与 A 共用候选池；每库两池使用同一冻结排序器和权重。每行分母均为同一批 1,000 题。','',
'| 数据集 | 固定池 | 池完整 P | 原 Top5 完整 | 重排后 Top5 完整 | 净增幅（百分点） | 配对 95% 区间（百分点） |',
'|---|---|---:|---:|---:|---:|---|']
for ds in names:
 for pool in ['S2','A']:
  v=data[ds][pool];s=v['K5'];ci=v['K5_bootstrap_95']['delta_rate']
  lines.append(f"| {names[ds]} | {pool} | {s['pool_complete']}（{pct(s['pool_rate'])}） | {s['before']}（{pct(s['before_rate'])}） | {s['after']}（{pct(s['after_rate'])}） | {pp(s['delta_rate'])} | {interval(ci)} |")
lines += ['', '## 救回、损失与池内保留', '',
'救回率 = 原来失败、重排后成功的题数 /（池完整题数 − 原 Top5 完整题数）。分母排除了池中缺证据的题。损失率以原 Top5 成功题为分母。池内条件保留率分别为重排前后 Top5 完整题数 / 池完整题数。','',
'| 数据集 | 池 | 救回题数 / 可救回分母 | 救回率 | 损失题数 / 原成功 | 损失率 | 池内保留率：前 → 后 |',
'|---|---|---:|---:|---:|---:|---|']
for ds in names:
 for pool in ['S2','A']:
  s=data[ds][pool]['K5']
  lines.append(f"| {names[ds]} | {pool} | {s['wins']}/{s['pool_complete']-s['before']} | {pct(s['rescue_rate'])} | {s['losses']}/{s['before']} | {pct(s['harm_rate'])} | {pct(s['before_retention'])} → {pct(s['after_retention'])} |")
lines += ['', '## K 的变化', '',
'下表全部来自相同完整排序的截断；没有为每个 K 重新检索、重新训练或选择权重。每格为“原排序 → 重排后”的严格完整率。完整 K=1–50 表见 `results/K_grid.csv`。','',
'| 数据集 | 池 | K=5 | K=10 | K=20 | K=50 |','|---|---|---|---|---|---|']
for ds in names:
 for pool in ['S2','A']:
  cells=[]
  for k in [5,10,20,50]:
   s=data[ds][pool]['K_grid'][k-1];cells.append(f"{pct(s['before_rate'])} → {pct(s['after_rate'])}")
  lines.append(f"| {names[ds]} | {pool} | "+' | '.join(cells)+' |')
lines += ['', '## 补查执行与跳过分层', '',
'各库分母分别为本次 1,000 题，且 S2 和 A 各自按实际补查掩码分层。这里的跳过题数与历史主家族 1,108 题、原开发 300 题的跳过数无关。','',
'| 数据集 | 池 | 分层 | 题数 | 池完整 | Top5 前 → 后 | 救回 / 原池内失败 | 损失 |','|---|---|---|---:|---:|---|---|---:|']
for ds in names:
 for pool in ['S2','A']:
  for label,zh in [('executed','执行'),('skipped','跳过')]:
   s=data[ds][pool]['execution_strata'][label]
   lines.append(f"| {names[ds]} | {pool} | {zh} | {s['n']} | {s['pool_complete']} | {s['before']} → {s['after']} | {s['wins']}/{s['pool_complete']-s['before']} | {s['losses']} |")
lines += ['', '## 按原失败深度分层', '',
'只取“池完整、原 Top5 失败”的题，按原排序首次支持齐全位次分桶。下表为救回题数 / 桶内失败题数（救回率）。它描述本次冻结排序器，不是其他排序方法的能力上限。','',
'| 数据集 | 池 | 6–10 | 11–20 | 21–50 | >50 |','|---|---|---|---|---|---|']
for ds in names:
 for pool in ['S2','A']:
  cells=[f"{s['wins']}/{s['n']}（{pct(s['rescue_rate'])}）" for s in data[ds][pool]['failure_depth_strata']]
  lines.append(f"| {names[ds]} | {pool} | "+' | '.join(cells)+' |')
lines += ['', '## 对已有结论的影响', '',
'1. **固定池中存在可修复的排序损失，得到新确认题支持。** S2→S2＋重排在 2Wiki、HotpotQA 新题上的 Top5 完整率分别增加 26.4、20.7 个百分点。MuSiQue 在重新训练、隔离关联组后增加 4.2 个百分点。A→C 同样三库均有正的 K=5 净增益。',
'2. **库间差异仍存在，但不能只由排序器强弱解释。** S2 池内失败救回率为 60.00%、58.73%、9.75%；MuSiQue 同时有较低池完整率和较低池内保留率。该结果刻画此模型、权重与样本组成，不隔离所有造成差异的因素。',
'3. **旧开发样本中的深位零救回不能泛化。** 新 2Wiki 确认集的 S2 共有 82 道池完整、首次齐全位次大于 50 的失败题，其中 13 道被救回 Top5（15.85%）；A 池对应为 12/106（11.32%）。HotpotQA 与 MuSiQue 的该桶仍为零。这直接反驳了把旧样本零救回泛化为“深位失败无法救回”的说法。',
'4. **保留完整 K 曲线，不写成所有 K 都改善。** 六组比较在预先列明的 K=5/10/20/50 均有净提升；K=1–50 全网格中，MuSiQue 的 A→C 在 K=39 为 455→454，下降 1 题（0.1 个百分点）。其余格没有负净变化，但可能为零。',
'5. **此次补强直接支持重排干预的确认性结论。** 它没有重新完成主家族的 repeat-q0 查询替换或 IDF8 反馈消融，不能拿这批确认集结果替代那些实验的证据与预算说明。','',
'## 核验与证据边界', '',
'- 全部 3,000 题使用同一严格有效题目掩码；候选池与 Top-K 指标口径一致。2Wiki/HotpotQA 要求每个金标题至少有一个返回版本包含全部标注支持句，不跨版本拼接；MuSiQue 要求金支持段落 ID 全部命中。',
'- 2Wiki 和 HotpotQA 模型文件与原模型逐字节哈希一致。MuSiQue 新建并拟合分类器；训练缓存恢复逐题行映射后精确排除 1,003 题，新分片保存每行题号。独立检查确认全部保留特征与原缓存对应行完全一致，确认题、缓冲题和开发题均无训练行。',
'- 原第一批 64 题、192 个采样方案块重新检索生成，与原缓存逐字节一致。模型拟合使用 20,990 题、1,638,385 行特征，其中正例 127,105 行。',
'- MuSiQue 权重只在原 300 题开发集上选取。原权重网格 0/0.25/0.5 的开发集六个 K 平均完整率分别为 32.9444%/33.6111%/32.2222%，因此冻结为 0.25。确认集没有参与选值。',
'- 独立复算直接读取原始返回、固定语料、文档特征与冻结模型，重建融合顺序、重排顺序、支持齐全位置和所有 K 截断。核验结果见 `verification/independent_recomputation.json`。',
'- 95% 区间使用预先指定的 2,000 次配对 Bootstrap：2Wiki/HotpotQA 在原分层内按题重采样，MuSiQue 按完整关联组重采样。区间为描述性结果，没有进行多重比较校正，不包含重新训练的随机性。',
'- 新确认集与历史 300 题在题目、来源和支持文档数分布上不同。不能把新旧分数差直接归因为旧实验泄漏，也不能把某个冻结排序器的救回率写成所有排序方法的上限。',
'- S2 与 A 的查询方案不同；需要作纯排序干预结论时，使用 S2 → S2＋重排或 A → C 的池内配对结果。排序干预没有新增检索调用。','',
'## 文件索引','',
'- `frozen/protocol.json`：预先冻结的抽样、评估与选参规则。',
'- `frozen/*_selection.json`、`frozen/musique_selected_groups.json`：确认题与关联组名单。',
'- `frozen/musique_ids.json`：新训练、开发、确认及缓冲题号。',
'- `models/musique/`：新模型及选参冻结回执；其余两库目录保存原模型副本。',
'- `results/summary.json`：全部统计、分层与主指标区间。',
'- `results/*_per_question.jsonl.gz`：逐题池完整及重排前后首次齐全位置。',
'- `results/K_grid.csv`：全部 K=1–50 的严格配对结果。',
'- `cache/`：原始各路返回、完整排序、特征和重排结果。',
'- `training/`：精确剔除后的训练分片、逐行题号与原题号—行边界映射。',
'- `verification/`：来源划分、原采样再生成和独立复算核验。',
'- `delivery_manifest.json`：本地接收文件的 SHA256 校验。','']
(ROOT/'确认评估结果.md').write_text('\n'.join(lines))
print(ROOT/'确认评估结果.md')
