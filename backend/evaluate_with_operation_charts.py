import os
import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

matplotlib.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
})

COLORS = {
    'Template':    '#2171B5',
    'Grammar':     '#238B45',
    'Seq2Seq':     '#CB181D',
    'Transformer': '#6A3D9A',
    'mT5':         '#D94801',
    'PE Zero-shot':'#636363',
    'PE Few-shot': '#969696',
    'PE CoT':      '#BDBDBD',
}

CATEGORIES = ['Filtrare', 'Agregare', 'JOIN simplu', 'Triple JOIN']

def classify_sql(sql: str) -> str:
    sql_up = sql.upper()
    n_join = sql_up.count('JOIN')
    if n_join >= 2:
        return 'Triple JOIN'
    if n_join == 1:
        return 'JOIN simplu'
    if any(f in sql_up for f in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
        return 'Agregare'
    if 'WHERE' in sql_up:
        return 'Filtrare'
    if 'ORDER BY' in sql_up:
        return 'Sortare'
    return 'Listare'

def load_standard_method(path: str, method_key: str) -> list:
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    key = list(d.keys())[0]
    return d[key]['results']


def load_prompt_method(path: str, technique: str) -> list:
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return d[technique]['results']


def compute_global(results: list) -> dict:
    n     = len(results)
    em_ok = sum(1 for r in results if r['exact_match'])
    ea_ok = sum(1 for r in results if r.get('execution_accuracy'))
    ves_sum = sum(r.get('ves') or 0.0 for r in results)
    return {
        'em':  em_ok / n * 100 if n else 0.0,
        'ea':  ea_ok / n * 100 if n else 0.0,
        'ves': ves_sum / n      if n else 0.0,
        'n':   n,
    }


def compute_by_category(results: list) -> dict:
    cat = defaultdict(lambda: {'em_ok': 0, 'ea_ok': 0, 'ves_sum': 0.0, 'total': 0})
    for r in results:
        c = classify_sql(r['gold'])
        cat[c]['total'] += 1
        if r['exact_match']:
            cat[c]['em_ok'] += 1
        if r.get('execution_accuracy'):
            cat[c]['ea_ok'] += 1
        cat[c]['ves_sum'] += r.get('ves') or 0.0
    return {
        c: {
            'em':  v['em_ok'] / v['total'] * 100 if v['total'] else 0.0,
            'ea':  v['ea_ok'] / v['total'] * 100 if v['total'] else 0.0,
            'ves': v['ves_sum'] / v['total'] if v['total'] else 0.0,
            'n':   v['total'],
        }
        for c, v in cat.items()
    }


def load_all(json_dir: str) -> tuple[dict, dict]:
    files = {
        'Template':    'evaluation_template.json',
        'Grammar':     'evaluation_grammar.json',
        'Seq2Seq':     'evaluation_seq2seq.json',
        'Transformer': 'evaluation_transformer.json',
        'mT5':         'evaluation_mt5.json',
    }
    prompt_file = os.path.join(json_dir, 'prompt_manual_results.json')
    prompt_techniques = {
        'PE Zero-shot': 'zero_shot',
        'PE Few-shot':  'few_shot',
        'PE CoT':       'cot',
    }

    global_results   = {}
    category_results = {}

    for method, fname in files.items():
        path = os.path.join(json_dir, fname)
        if not os.path.exists(path):
            print(f'  [WARN] Lipsă: {path}')
            continue
        results = load_standard_method(path, method.lower())
        global_results[method]   = compute_global(results)
        category_results[method] = compute_by_category(results)
        print(f'  {method}: {len(results)} exemple, '
              f'EM={global_results[method]["em"]:.1f}%, '
              f'EA={global_results[method]["ea"]:.1f}%')

    if os.path.exists(prompt_file):
        with open(prompt_file, encoding='utf-8') as f:
            pd = json.load(f)
        for method, tech_key in prompt_techniques.items():
            if tech_key not in pd:
                continue
            results = pd[tech_key]['results']
            global_results[method]   = compute_global(results)
            category_results[method] = compute_by_category(results)
            print(f'  {method}: {len(results)} exemple, '
                  f'EM={global_results[method]["em"]:.1f}%, '
                  f'EA={global_results[method]["ea"]:.1f}%')
    else:
        print(f'  [WARN] Lipsă: {prompt_file}')

    return global_results, category_results

def save(fig, filename, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'  Salvat: {path}')
    plt.close(fig)

def plot_em_ea_comparison(global_results, out_dir):
    methods = [m for m in [
        'Template', 'Grammar', 'Seq2Seq', 'Transformer', 'mT5',
        'PE Zero-shot', 'PE Few-shot', 'PE CoT'
    ] if m in global_results]

    em_vals = [global_results[m]['em'] for m in methods]
    ea_vals = [global_results[m]['ea'] for m in methods]

    x     = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))

    bars_em = ax.bar(x - width/2, em_vals, width,
                     color='#2171B5', label='Exact Match',
                     alpha=0.85, edgecolor='#084594', linewidth=0.7)
    bars_ea = ax.bar(x + width/2, ea_vals, width,
                     color='#238B45', label='Execution Accuracy',
                     alpha=0.85, edgecolor='#005A32', linewidth=0.7,
                     hatch='//')

    for bar in bars_em:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f'{h:.1f}', ha='center', va='bottom', fontsize=8,
                color='#2171B5')
    for bar in bars_ea:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f'{h:.1f}', ha='center', va='bottom', fontsize=8,
                color='#238B45')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('Acuratețe (%)')
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(loc='upper left', framealpha=0.9)

    if 'PE Zero-shot' in methods:
        sep = methods.index('PE Zero-shot') - 0.5
        ax.axvline(x=sep, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax.text(sep + 0.1, 108, 'Prompt\nEngineering',
                fontsize=8, color='gray', va='top')

    fig.tight_layout()
    save(fig, 'em_ea_comparison.png', out_dir)


def plot_category_em(global_results, category_results, out_dir):
    methods   = [m for m in ['Template', 'Grammar', 'Seq2Seq',
                              'Transformer', 'mT5']
                 if m in category_results]
    n_methods = len(methods)
    x         = np.arange(len(CATEGORIES))
    width     = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, method in enumerate(methods):
        vals   = [category_results[method].get(cat, {}).get('em', 0)
                  for cat in CATEGORIES]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width,
               label=method,
               color=COLORS.get(method, f'C{i}'),
               alpha=0.85, edgecolor='white', linewidth=0.5)

    labels_x = []
    for cat in CATEGORIES:

        n = next(
            (category_results[m].get(cat, {}).get('n', '')
             for m in methods if m in category_results),
            ''
        )
        labels_x.append(f'{cat}\n(n={n})')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.set_ylabel('Exact Match (%)')
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)

    fig.tight_layout()
    save(fig, 'category_em.png', out_dir)


def plot_category_ea(global_results, category_results, out_dir):
    methods   = [m for m in ['Template', 'Grammar', 'Seq2Seq',
                              'Transformer', 'mT5']
                 if m in category_results]
    n_methods = len(methods)
    x         = np.arange(len(CATEGORIES))
    width     = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, method in enumerate(methods):
        vals   = [category_results[method].get(cat, {}).get('ea', 0)
                  for cat in CATEGORIES]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width,
               label=method,
               color=COLORS.get(method, f'C{i}'),
               alpha=0.85, edgecolor='white', linewidth=0.5,
               hatch='//' if i % 2 == 1 else '')

    labels_x = []
    for cat in CATEGORIES:
        n = next(
            (category_results[m].get(cat, {}).get('n', '')
             for m in methods if m in category_results),
            ''
        )
        labels_x.append(f'{cat}\n(n={n})')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.set_ylabel('Execution Accuracy (%)')
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)

    fig.tight_layout()
    save(fig, 'category_ea.png', out_dir)


def plot_em_ea_gap(global_results, out_dir):
    methods = [m for m in [
        'Template', 'Grammar', 'Seq2Seq', 'Transformer', 'mT5',
        'PE Zero-shot', 'PE Few-shot', 'PE CoT'
    ] if m in global_results]

    gaps = [global_results[m]['ea'] - global_results[m]['em']
            for m in methods]
    colors = ['#CB181D' if g > 10 else '#2171B5' if g > 3 else '#238B45'
              for g in gaps]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    x    = np.arange(len(methods))
    bars = ax.bar(x, gaps, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.5)

    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{gap:.1f}pp', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('EA − EM (puncte procentuale)')
    ax.set_ylim(0, max(gaps) + 10)
    ax.axhline(y=0, color='black', linewidth=0.8)

    legend_elements = [
        mpatches.Patch(color='#CB181D', alpha=0.85, label='Diferență mare (>10pp)'),
        mpatches.Patch(color='#2171B5', alpha=0.85, label='Diferență medie (3–10pp)'),
        mpatches.Patch(color='#238B45', alpha=0.85, label='Diferență mică (<3pp)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    if 'PE Zero-shot' in methods:
        sep = methods.index('PE Zero-shot') - 0.5
        ax.axvline(x=sep, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax.text(sep + 0.1, max(gaps) + 6, 'PE', fontsize=8, color='gray')

    fig.tight_layout()
    save(fig, 'em_ea_gap.png', out_dir)


def plot_ves_comparison(global_results, out_dir):
    methods  = [m for m in [
        'Template', 'Grammar', 'Seq2Seq', 'Transformer', 'mT5',
        'PE Zero-shot', 'PE Few-shot', 'PE CoT'
    ] if m in global_results]

    ves_vals = [global_results[m]['ves'] for m in methods]
    colors   = [COLORS.get(m, 'C0') for m in methods]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    x    = np.arange(len(methods))
    bars = ax.bar(x, ves_vals, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.5)

    for bar, v in zip(bars, ves_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.008,
                f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('VES adaptat')
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(y=1.0, color='gray', linestyle='--',
               linewidth=1, alpha=0.6, label='VES maxim = 1.0')
    ax.legend(framealpha=0.9)

    if 'PE Zero-shot' in methods:
        sep = methods.index('PE Zero-shot') - 0.5
        ax.axvline(x=sep, color='gray', linestyle=':', linewidth=1, alpha=0.6)

    fig.tight_layout()
    save(fig, 'ves_comparison.png', out_dir)


def plot_radar(global_results, out_dir):
    methods  = [m for m in ['Template', 'Grammar', 'Seq2Seq',
                             'Transformer', 'mT5']
                if m in global_results]
    metrics  = ['EM (%)', 'EA (%)', 'VES×100']
    angles   = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles  += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for method in methods:
        g    = global_results[method]
        vals = [g['em'], g['ea'], g['ves'] * 100] + [g['em']]
        ax.plot(angles, vals, 'o-', linewidth=2,
                color=COLORS[method], label=method, markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=COLORS[method])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), framealpha=0.9)

    fig.tight_layout()
    save(fig, 'radar_chart.png', out_dir)


def plot_heatmap(global_results, category_results, out_dir):
    methods = [m for m in ['Template', 'Grammar', 'Seq2Seq',
                            'Transformer', 'mT5']
               if m in category_results]

    matrix = np.array([
        [category_results[m].get(cat, {}).get('em', 0)
         for cat in CATEGORIES]
        for m in methods
    ])

    col_labels = []
    for cat in CATEGORIES:
        n = next(
            (category_results[m].get(cat, {}).get('n', '')
             for m in methods if m in category_results),
            ''
        )
        col_labels.append(f'{cat}\n(n={n})')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels(col_labels, rotation=15, ha='right')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)

    for i in range(len(methods)):
        for j in range(len(CATEGORIES)):
            val   = matrix[i, j]
            color = 'white' if val < 30 or val > 80 else 'black'
            ax.text(j, i, f'{val:.1f}%',
                    ha='center', va='center',
                    fontsize=10, color=color, fontweight='500')

    plt.colorbar(im, ax=ax, label='Exact Match (%)',
                 fraction=0.04, pad=0.04)

    fig.tight_layout()
    save(fig, 'heatmap_em.png', out_dir)


def plot_operation_metric_charts(global_results, category_results, out_dir):
    methods = [m for m in [
        'Template', 'Grammar', 'Seq2Seq', 'Transformer', 'mT5',
        'PE Zero-shot', 'PE Few-shot', 'PE CoT'
    ] if m in category_results]

    metric_labels = ['EM', 'EA', 'VES × 100']
    metric_keys = ['em', 'ea', 'ves']
    metric_colors = ['#2171B5', '#238B45', '#D94801']

    filename_map = {
        'Filtrare': 'operation_filtrare_em_ea_ves.png',
        'Agregare': 'operation_agregare_em_ea_ves.png',
        'JOIN simplu': 'operation_join_simplu_em_ea_ves.png',
        'Triple JOIN': 'operation_triple_join_em_ea_ves.png',
    }

    for category in CATEGORIES:
        x = np.arange(len(methods))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 5))

        for i, (label, key, color) in enumerate(zip(metric_labels, metric_keys, metric_colors)):
            vals = []
            for method in methods:
                value = category_results[method].get(category, {}).get(key, 0)
                if key == 'ves':
                    value *= 100
                vals.append(value)

            bars = ax.bar(
                x + (i - 1) * width,
                vals,
                width,
                label=label,
                color=color,
                alpha=0.85,
                edgecolor='white',
                linewidth=0.6,
                hatch='//' if key == 'ea' else ''
            )

            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f'{v:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=0
                )

        n = next(
            (category_results[m].get(category, {}).get('n', '')
             for m in methods if category in category_results[m]),
            ''
        )

        ax.set_title(f'{category}: comparație EM, EA și VES (n={n})')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.set_ylabel('Scor (%)')
        ax.set_ylim(0, 115)
        ax.set_yticks(range(0, 101, 20))
        ax.legend(loc='upper right', framealpha=0.9, ncol=3)
        ax.grid(True, alpha=0.3, linestyle='--')

        fig.tight_layout()
        save(fig, filename_map[category], out_dir)

def plot_prompt_engineering(global_results, out_dir):
    techniques = [m for m in ['PE Zero-shot', 'PE Few-shot', 'PE CoT']
                  if m in global_results]
    labels     = [t.replace('PE ', '') for t in techniques]

    em_vals  = [global_results[t]['em']  for t in techniques]
    ea_vals  = [global_results[t]['ea']  for t in techniques]
    ves_vals = [global_results[t]['ves'] for t in techniques]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    for ax, vals, ylabel, title, bar_colors in [
        (axes[0], em_vals,  'Exact Match (%)',        'Exact Match',
         ['#636363', '#969696', '#BDBDBD']),
        (axes[1], ea_vals,  'Execution Accuracy (%)', 'Execution Accuracy',
         ['#2171B5', '#4292C6', '#6BAED6']),
        (axes[2], ves_vals, 'VES adaptat',            'VES adaptat',
         ['#238B45', '#41AB5D', '#74C476']),
    ]:
        bars = ax.bar(labels, vals,
                      color=bar_colors[:len(labels)],
                      edgecolor='#333', linewidth=0.7, alpha=0.9)
        for bar, v in zip(bars, vals):
            fmt = f'{v:.1f}%' if ylabel != 'VES adaptat' else f'{v:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.5 if ylabel != 'VES adaptat' else 0.008),
                    fmt, ha='center', va='bottom', fontsize=10)
        ax.set_ylabel(ylabel)
        ylim = (0, max(vals) * 1.25 + 5) if vals else (0, 100)
        ax.set_ylim(ylim)
        ax.set_title(title)

    n_total = global_results.get('PE Zero-shot', {}).get('n', 20)
    fig.suptitle(f'Comparație tehnici Prompt Engineering (n={n_total})',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, 'prompt_engineering_comparison.png', out_dir)

def plot_category_profile(global_results, category_results, out_dir):
    methods = [m for m in ['Template', 'Grammar', 'Seq2Seq',
                            'Transformer', 'mT5']
               if m in category_results]

    cats = ['Filtrare', 'Agregare', 'JOIN simplu', 'Triple JOIN']

    first_m = methods[0]
    labels_x = []
    for cat in cats:
        n = category_results[first_m].get(cat, {}).get('n', '')
        labels_x.append(f'{cat}\n(n={n})')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, metric, ylabel in [
        (axes[0], 'em', 'Exact Match (%)'),
        (axes[1], 'ea', 'Execution Accuracy (%)'),
    ]:
        for method in methods:
            vals = [category_results[method].get(cat, {}).get(metric, 0)
                    for cat in cats]
            ax.plot(range(len(cats)), vals,
                    'o-', linewidth=2, markersize=7,
                    color=COLORS.get(method, 'C0'),
                    label=method)
            for i, v in enumerate(vals):
                ax.annotate(f'{v:.0f}',
                            xy=(i, v),
                            xytext=(0, 7),
                            textcoords='offset points',
                            ha='center', fontsize=8,
                            color=COLORS.get(method, 'C0'))

        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(labels_x, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-5, 115)
        ax.set_yticks(range(0, 101, 20))
        ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    fig.tight_layout()
    save(fig, 'category_profile_lines.png', out_dir)

def plot_complexity_tradeoff(global_results, out_dir):
    order = ['Template', 'Grammar', 'Seq2Seq', 'Transformer', 'mT5']
    methods = [m for m in order if m in global_results]

    em_vals  = [global_results[m]['em']  for m in methods]
    ea_vals  = [global_results[m]['ea']  for m in methods]
    ves_vals = [global_results[m]['ves'] * 100 for m in methods]

    x = range(len(methods))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, em_vals,  'o-', linewidth=2.5, markersize=8,
            color='#2171B5', label='Exact Match (%)', zorder=3)
    ax.plot(x, ea_vals,  's--', linewidth=2.5, markersize=8,
            color='#238B45', label='Execution Accuracy (%)', zorder=3)
    ax.plot(x, ves_vals, '^:', linewidth=2, markersize=8,
            color='#D94801', label='VES × 100', zorder=3)

    ax.fill_between(x, em_vals, ea_vals,
                    alpha=0.08, color='#238B45',
                    label='Gap EM–EA')

    for i, (em, ea, ves) in enumerate(zip(em_vals, ea_vals, ves_vals)):
        ax.annotate(f'{em:.1f}', xy=(i, em), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=8, color='#2171B5')
        ax.annotate(f'{ea:.1f}', xy=(i, ea), xytext=(0, -16),
                    textcoords='offset points', ha='center',
                    fontsize=8, color='#238B45')

    ax.annotate('', xy=(len(methods)-1, -8), xytext=(0, -8),
                arrowprops=dict(arrowstyle='->', color='gray',
                                lw=1.5),
    )
    ax.text(len(methods)/2 - 0.5, -12,
            'Complexitate arhitecturală crescătoare →',
            ha='center', fontsize=9, color='gray',
)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Scor (%)')
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.tight_layout()
    save(fig, 'complexity_tradeoff.png', out_dir)


def plot_em_ea_shaded(global_results, out_dir):
    methods = [m for m in [
        'Template', 'Grammar', 'Seq2Seq', 'Transformer', 'mT5',
        'PE Zero-shot', 'PE Few-shot', 'PE CoT'
    ] if m in global_results]

    em_vals  = [global_results[m]['em']  for m in methods]
    ea_vals  = [global_results[m]['ea']  for m in methods]
    x        = range(len(methods))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x, ea_vals, 'o-', linewidth=2.5, markersize=8,
            color='#238B45', label='Execution Accuracy', zorder=4)
    ax.plot(x, em_vals, 's--', linewidth=2.5, markersize=8,
            color='#2171B5', label='Exact Match', zorder=4)

    ax.fill_between(x, em_vals, ea_vals,
                    alpha=0.15, color='#CB181D',
                    label='Gap semantic (EA − EM)')

    for i, (em, ea) in enumerate(zip(em_vals, ea_vals)):
        gap = ea - em
        if gap > 1:
            mid = (em + ea) / 2
            ax.text(i, mid, f'+{gap:.1f}pp',
                    ha='center', va='center',
                    fontsize=7.5, color='#CB181D',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.7,
                              edgecolor='none'))

    if 'PE Zero-shot' in methods:
        sep = methods.index('PE Zero-shot') - 0.5
        ax.axvline(x=sep, color='gray', linestyle=':',
                   linewidth=1.2, alpha=0.7)
        ax.text(sep + 0.1, 108, 'Prompt\nEngineering',
                fontsize=8, color='gray', va='top')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('Acuratețe (%)')
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.tight_layout()
    save(fig, 'em_ea_shaded.png', out_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Generează grafice comparative NL2SQL din fișiere JSON'
    )
    parser.add_argument(
        '--json-dir', required=True,
        help='Folder cu fișierele JSON de evaluare'
    )
    parser.add_argument(
        '--out', default='figures',
        help='Folder output pentru grafice (default: figures/)'
    )
    parser.add_argument(
        '--chart', default='all',
        choices=['all', 'em_ea', 'category_em', 'category_ea',
                 'gap', 'ves', 'radar', 'heatmap', 'prompt',
                 'cat_profile', 'tradeoff', 'em_ea_shaded', 'operation_metrics'],
        help='Graficul de generat (default: all)'
    )
    args = parser.parse_args()

    print(f'\nÎncărcare date din: {args.json_dir}\n')
    global_results, category_results = load_all(args.json_dir)

    if not global_results:
        print('Eroare: niciun fișier JSON găsit.')
        return

    print(f'\nGenerare grafice în: {args.out}/\n')

    charts = {
        'em_ea':           lambda: plot_em_ea_comparison(global_results, args.out),
        'category_em':     lambda: plot_category_em(global_results, category_results, args.out),
        'category_ea':     lambda: plot_category_ea(global_results, category_results, args.out),
        'gap':             lambda: plot_em_ea_gap(global_results, args.out),
        'ves':             lambda: plot_ves_comparison(global_results, args.out),
        'radar':           lambda: plot_radar(global_results, args.out),
        'heatmap':         lambda: plot_heatmap(global_results, category_results, args.out),
        'prompt':          lambda: plot_prompt_engineering(global_results, args.out),
        'cat_profile':     lambda: plot_category_profile(global_results, category_results, args.out),
        'tradeoff':        lambda: plot_complexity_tradeoff(global_results, args.out),
        'em_ea_shaded':    lambda: plot_em_ea_shaded(global_results, args.out),
        'operation_metrics': lambda: plot_operation_metric_charts(global_results, category_results, args.out),
    }

    if args.chart == 'all':
        for name, fn in charts.items():
            print(f'  [{name}]')
            fn()
    else:
        print(f'  [{args.chart}]')
        charts[args.chart]()


if __name__ == '__main__':
    main()