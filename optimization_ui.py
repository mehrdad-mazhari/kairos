#!/usr/bin/env python3
"""Desktop control panel for the existing transmon optimizer (Tk, no web server)."""
import copy
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font as tkfont
from datetime import datetime

from optimization_inputs import INPUTS, parse_physical_input

PROJECT = Path(__file__).resolve().parent
LABELS = {
    'detuning_mhz': 'Detuning Δ / 2π', 'alpha_mhz': 'Anharmonicity α / 2π',
    't1_1_ns': 'Relaxation T1(1)', 't1_2_ns': 'Relaxation T1(2)',
    'tphi_1_ns': 'Dephasing Tφ(1)', 'tphi_2_ns': 'Dephasing Tφ(2)',
    'tg_ns': 'Gate duration Tg',
}
from pulse_methods import METHOD_LABELS, parameter_rows, selected_spec_from_manifest


def configured_python():
    if sys.prefix == sys.base_prefix:
        for path in (PROJECT/'.venv/bin/python', PROJECT.parent/'.venv/bin/python'):
            if path.is_file():
                return str(path)
    return sys.executable


def build_config(base, inputs, nsga, grape):
    """One validation path for desktop and CLI; never alter fixed physical inputs."""
    from robust_optimization import validate_config
    config = copy.deepcopy(base)
    for name in INPUTS:
        value = parse_physical_input(inputs[name], name)
        if name == 'tg_ns':
            config['gate_duration_ns'] = value
        else:
            config['device'][name] = value
    config['nsga_enabled'] = bool(nsga)
    config.setdefault('grape', {})['enabled'] = bool(grape)
    validate_config(config)
    return config


def ui_font(root):
    """Choose a readable sans-serif for the main application."""
    families = {name.casefold(): name for name in tkfont.families(root)}
    for name in ('Noto Sans', 'DejaVu Sans', 'Arial'):
        if name.casefold() in families:
            return families[name.casefold()]
    return 'TkDefaultFont'


def load_logo(root, size):
    source = tk.PhotoImage(master=root, file=str(PROJECT/'assets'/'kairos-logo.png'))
    factor = max(1, (max(source.width(), source.height())+size-1)//size)
    return source.subsample(factor, factor)


class OptimizerApp:
    # A shared palette keeps widgets, charts and states visually consistent.
    BG = '#0b0d10'
    INK = '#eee9df'
    MUTED = '#a4a7ae'
    ACCENT = '#d6af70'
    LINE = '#303238'
    FONT = 'Noto Sans'

    def __init__(self, root):
        self.root = root
        self.FONT = ui_font(root)
        root.title('Kairos')
        root.geometry('1360x900')
        root.minsize(1180, 760)
        root.configure(bg=self.BG)
        self.base = json.loads((PROJECT/'challenge_config.json').read_text())
        self.events = queue.Queue()
        self.process = None
        self.running = False
        self.output = None
        self.inputs = {}
        self.input_entries = {}
        self.nsga = tk.BooleanVar(value=False)
        self.grape = tk.BooleanVar(value=self.base.get('grape', {}).get('enabled', True))
        self.quick = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value='Ready. Set your device parameters and run a comparison.')
        self.winner = tk.StringVar(value='Your next optimal pulse starts here.')
        self.winner_caption = tk.StringVar(value='Compare Gaussian, DRAG and your selected optimization methods.')
        self.quality = tk.StringVar(value='AWAITING RESULTS')
        self.metric_values = [tk.StringVar(value='—') for _ in range(3)]
        self.chart_data = []
        self.detail = tk.StringVar(value='Best means the best eligible candidate found in this run. Independent validation checks the selected pulse.')
        self._configure_styles()

        # Compact navigation keeps file actions separate from the experiment.
        header = tk.Frame(root, bg='#090b0e', padx=26, pady=16)
        header.pack(fill='x')
        self.logo = load_logo(root, 64)
        root.iconphoto(True, self.logo)
        mark = tk.Label(header, image=self.logo, bg='#000000', bd=0)
        mark.pack(side='left', padx=(0, 14))
        tk.Label(header, text='Kairos', font=self._font(17, True),
                 bg='#090b0e', fg='#eee9df').pack(side='left')
        tk.Label(header, text=' /  PULSE LAB', font=self._font(9),
                 bg='#090b0e', fg='#a4a7ae').pack(side='left', padx=10)
        ttk.Button(header, text='Open results', style='Header.TButton',
                   command=self.load_results).pack(side='right', padx=(10, 0))
        ttk.Button(header, text='Load configuration', style='Header.TButton',
                   command=self.load_config).pack(side='right')

        # Pin status and the Run button outside scrolling content.
        footer = tk.Frame(root, bg=self.BG, padx=26, pady=12)
        footer.pack(side='bottom', fill='x')
        self.progress = ttk.Progressbar(footer, mode='indeterminate', style='Accent.Horizontal.TProgressbar')
        self.progress.pack(fill='x', pady=(0, 8))
        tk.Label(footer, textvariable=self.status, font=self._font(9), bg=self.BG,
                 fg=self.MUTED, anchor='w').pack(fill='x')
        body = tk.Frame(root, bg=self.BG, padx=24, pady=20)
        body.pack(fill='both', expand=True)
        sidebar = tk.Frame(body, bg=self.BG, width=408)
        sidebar.pack(side='left', fill='y', padx=(0, 22))
        sidebar.pack_propagate(False)
        actions = tk.Frame(sidebar, bg=self.BG)
        actions.pack(side='bottom', fill='x', pady=(12, 0))
        self.cancel_button = ttk.Button(actions, text='Stop', command=self.cancel, state='disabled', style='Secondary.TButton')
        self.cancel_button.pack(side='right', padx=(8, 0))
        self.run_button = ttk.Button(actions, text='Run optimization  →', style='Run.TButton', command=self.start)
        self.run_button.pack(side='left', fill='x', expand=True)
        left = self._scrollable(sidebar, self.BG)

        device_card = self._card(left)
        self._label(device_card, '01  /  DEVICE', size=9, color=self.ACCENT, bold=True).pack(anchor='w')
        self._label(device_card, 'Physical parameters', size=16, bold=True).pack(anchor='w', pady=(5, 2))
        self._label(device_card, 'Your fixed device. Your gate duration.', color=self.MUTED, size=9).pack(anchor='w', pady=(0, 8))
        fields = tk.Frame(device_card, bg='#14181e')
        fields.pack(fill='x')
        fields.columnconfigure(0, weight=1)
        fields.columnconfigure(1, weight=0, minsize=170)
        for row, name in enumerate(INPUTS):
            unit = 'MHz' if name in ('detuning_mhz', 'alpha_mhz') else 'ns'
            self._label(fields, LABELS[name], size=10).grid(row=row, column=0, sticky='w', padx=(0, 8), pady=5)
            # The visible bordered box, value and unit share one aligned row.
            box = tk.Frame(fields, bg='#0d1014', highlightbackground='#363941',
                           highlightcolor=self.ACCENT, highlightthickness=1, bd=0)
            box.grid(row=row, column=1, sticky='ew', pady=3)
            value = self.base['gate_duration_ns'] if name == 'tg_ns' else self.base['device'][name]
            self.inputs[name] = tk.StringVar(value=f'{value:g}')
            entry = tk.Entry(box, textvariable=self.inputs[name], width=11, bd=0,
                             relief='flat', font=self._font(11), bg='#0d1014', fg=self.INK,
                             insertbackground=self.ACCENT, selectbackground='#59462d',
                             selectforeground=self.INK, highlightthickness=0)
            entry.pack(side='left', fill='x', expand=True, padx=(10, 4), pady=4)
            tk.Label(box, text=unit, font=self._font(9), bg='#0d1014', fg=self.MUTED,
                     width=4).pack(side='right', padx=(0, 6))
            entry.bind('<FocusIn>', lambda event, box=box: box.configure(highlightbackground=self.ACCENT))
            entry.bind('<FocusOut>', lambda event, box=box: box.configure(highlightbackground='#363941'))
            self.input_entries[name] = entry
        self._label(device_card, 'Explicit units accepted, e.g. 30 us or -0.22 GHz.\nTg: 10–60 ns in steps of 0.2 ns.',
                    size=9, color=self.MUTED).pack(anchor='w', pady=(12, 0))

        methods = self._card(left, pady=(12, 0))
        self._label(methods, '02  /  METHODS', size=9, color=self.ACCENT, bold=True).pack(anchor='w', pady=(0, 9))
        self._label(methods, 'Six reference pulse families', bold=True).pack(anchor='w')
        self._label(methods, 'Gaussian → DRAG → Higher-order DRAG\n→ Wah-Wah → BB1 → CORPSE\nComposites show their longer actual durations.', size=9, color=self.MUTED).pack(anchor='w', pady=(2, 10))
        ttk.Checkbutton(methods, text='NSGA-III numerical search', variable=self.nsga,
                        style='Method.TCheckbutton').pack(anchor='w', pady=5)
        ttk.Checkbutton(methods, text='GRAPE refinement', variable=self.grape,
                        style='Method.TCheckbutton').pack(anchor='w', pady=5)
        tk.Frame(methods, bg=self.LINE, height=1).pack(fill='x', pady=10)
        ttk.Checkbutton(methods, text='Quick check · reduced budget', variable=self.quick,
                        style='Method.TCheckbutton').pack(anchor='w')
        self._label(methods, 'Monte Carlo runs after the selected methods.', size=9,
                    color=self.MUTED).pack(anchor='w', pady=(10, 0))

        right = tk.Frame(body, bg=self.BG)
        right.pack(side='left', fill='both', expand=True)
        overview = tk.Frame(right, bg=self.BG)
        overview.pack(fill='x', pady=(0, 8))
        self._label(overview, 'EXPERIMENT OVERVIEW', bg=self.BG, size=9,
                    color=self.MUTED, bold=True).pack(anchor='w')
        self._label(overview, textvariable=self.winner, bg=self.BG, size=20,
                    bold=True, wraplength=750).pack(anchor='w', pady=(6, 4))
        self._label(overview, textvariable=self.winner_caption, bg=self.BG,
                    color=self.MUTED, size=10, wraplength=750).pack(anchor='w')
        metrics = tk.Frame(right, bg=self.BG)
        metrics.pack(fill='x', pady=(0, 16))
        for index, (title, subtitle) in enumerate((('NOMINAL FIDELITY', 'Six-state gate average'),
                                                   ('MC MEAN FIDELITY', 'Held-out device samples'),
                                                   ('MC P05 FIDELITY', 'Lower-tail performance'))):
            metrics.columnconfigure(index, weight=1, uniform='metric')
            card = tk.Frame(metrics, bg='#14181e', padx=16, pady=14,
                            highlightbackground=self.LINE, highlightthickness=1)
            card.grid(row=0, column=index, sticky='nsew', padx=(0 if index == 0 else 6, 0 if index == 2 else 6))
            self._label(card, title, color=self.MUTED, size=8, bold=True).pack(anchor='w')
            self._label(card, textvariable=self.metric_values[index], size=23, bold=True,
                        color=self.ACCENT if index == 0 else self.INK).pack(anchor='w', pady=(4, 2))
            self._label(card, subtitle, size=8, color=self.MUTED).pack(anchor='w')

        tabs = ttk.Notebook(right, style='Studio.TNotebook')
        tabs.pack(fill='both', expand=True)
        result_host = tk.Frame(tabs, bg='#14181e')
        parameters_host = tk.Frame(tabs, bg='#14181e', padx=18, pady=18)
        logs = tk.Frame(tabs, bg='#090b0e', padx=14, pady=14)
        tabs.add(result_host, text='  Comparison  ')
        tabs.add(parameters_host, text='  Selected pulse  ')
        tabs.add(logs, text='  Activity log  ')
        self._label(parameters_host, 'Selected pulse parameters', size=15, bold=True).pack(anchor='w')
        self._label(parameters_host, 'Amplitude, shape and method-specific controls. Values include units.',
                    color=self.MUTED, size=9).pack(anchor='w', pady=(4, 12))
        ttk.Button(parameters_host, text='Copy pulse parameters', style='Secondary.TButton',
                   command=self.copy_parameters).pack(side='bottom', anchor='w', pady=(12, 0))
        param_body = tk.Frame(parameters_host, bg='#14181e')
        param_body.pack(fill='both', expand=True)
        self.parameter_text = tk.Text(param_body, bg='#0e1115', fg=self.INK, relief='flat', bd=0,
                                      padx=14, pady=14, wrap='word', height=1, width=1, font=self._font(10))
        param_scroll = ttk.Scrollbar(param_body, command=self.parameter_text.yview, style='Studio.Vertical.TScrollbar')
        self.parameter_text.configure(yscrollcommand=param_scroll.set)
        param_scroll.pack(side='right', fill='y')
        self.parameter_text.pack(fill='both', expand=True)
        self.parameter_text.insert('end', 'Run a comparison or open saved results to see the selected pulse parameters.')
        self.parameter_text.configure(state='disabled')
        results = self._scrollable(result_host, '#14181e')
        results.configure(padx=18, pady=18)
        result_heading = tk.Frame(results, bg='#14181e')
        result_heading.pack(fill='x', pady=(0, 12))
        self._label(result_heading, 'Method performance', size=13, bold=True).pack(side='left')
        self.quality_badge = self._label(result_heading, textvariable=self.quality, size=8, bold=True,
                                         color=self.MUTED, bg='#20242b', padx=10, pady=5)
        self.quality_badge.pack(side='right')
        columns = ('method', 'duration', 'nominal', 'mean', 'p05', 'leakage', 'eligibility')
        self.table = ttk.Treeview(results, columns=columns, show='headings', height=1, style='Studio.Treeview')
        for key, label in zip(columns, ('METHOD', 'TIME ns', 'NOMINAL %', 'MC MEAN %', 'MC P05 %', 'MAX LEAK %', 'ELIGIBILITY')):
            self.table.heading(key, text=label)
            self.table.column(key, width=150 if key == 'method' else 96,
                              minwidth=130 if key == 'method' else 74, anchor='w' if key == 'method' else 'e')
        self.table.tag_configure('even', background='#191d23')
        self.table.tag_configure('final', background='#332b20', foreground='#edc989', font=self._font(10, True))
        self.table.pack(fill='x')
        self.selection_detail = tk.StringVar(value='Select a method row to see its eligibility and any rejection reasons.')
        self.method_details = {}
        self.table.bind('<<TreeviewSelect>>', self.show_method_detail)
        self._label(results, textvariable=self.selection_detail, size=9, color=self.MUTED,
                    wraplength=720).pack(anchor='w', fill='x', pady=(8, 0))
        self.chart = tk.Canvas(results, background='#14181e', height=260, highlightthickness=0)
        self.chart.pack(fill='x', pady=(16, 10))
        self.chart.bind('<Configure>', lambda event: self.draw_chart())
        self._label(results, textvariable=self.detail, color=self.MUTED, size=9,
                    wraplength=720).pack(anchor='w', fill='x')
        ttk.Button(results, text='Open report & files  ↗', style='Secondary.TButton',
                   command=self.open_output).pack(anchor='w', pady=(12, 0))
        self.log = tk.Text(logs, wrap='word', font=self._font(10), bd=0,
                           padx=8, pady=8, bg='#090b0e', fg='#e6e0d5', state='disabled')
        scroll = ttk.Scrollbar(logs, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.log.pack(fill='both', expand=True)
        root.bind('<Control-Return>', lambda event: self.start())
        for event in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            root.bind(event, self._scroll_wheel, add='+')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self.poll)
        self.draw_chart()

    def _font(self, size, bold=False):
        # Pixel fonts match the pixel layout, avoiding Xft's extra point-size
        # scaling on HiDPI desktops (which can otherwise double text size).
        return (self.FONT, -round(size * 4 / 3), 'bold' if bold else 'normal')

    def _label(self, parent, text='', *, size=10, color=None, bg='#14181e', bold=False, **kwargs):
        return tk.Label(parent, text=text, bg=bg, fg=color or self.INK,
                        font=self._font(size, bold),
                        anchor='w', justify='left', bd=0, **kwargs)

    def _card(self, parent, pady=0):
        card = tk.Frame(parent, bg='#14181e', padx=18, pady=18,
                        highlightbackground=self.LINE, highlightthickness=1)
        card.pack(fill='x', pady=pady)
        return card

    def _scrollable(self, parent, bg):
        host = tk.Frame(parent, bg=bg)
        host.pack(fill='both', expand=True)
        canvas = tk.Canvas(host, bg=bg, highlightthickness=0, width=1, height=1)
        scrollbar = ttk.Scrollbar(host, orient='vertical', command=canvas.yview, style='Studio.Vertical.TScrollbar')
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        content = tk.Frame(canvas, bg=bg)
        content._scroll_canvas = canvas
        window = canvas.create_window((0, 0), window=content, anchor='nw')
        content.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda event: canvas.itemconfigure(window, width=event.width))
        return content

    def _scroll_wheel(self, event):
        widget = event.widget
        # Text and table widgets retain their own scrolling behavior.
        if isinstance(widget, (tk.Text, ttk.Treeview)):
            return
        while widget is not None:
            if hasattr(widget, '_scroll_canvas'):
                canvas = widget._scroll_canvas
                bounds = canvas.bbox('all')
                if bounds and bounds[3] > canvas.winfo_height():
                    direction = -1 if getattr(event, 'num', None) == 4 or getattr(event, 'delta', 0) > 0 else 1
                    canvas.yview_scroll(direction * 3, 'units')
                return
            widget = getattr(widget, 'master', None)

    def _configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', font=self._font(10))
        for name, color, foreground in [('Run', self.ACCENT, '#101216'),
                                         ('Secondary', '#252a32', self.INK),
                                         ('Header', '#242830', '#eee9df')]:
            style.configure(name+'.TButton', background=color, foreground=foreground,
                            borderwidth=0, relief='flat', padding=(16, 11), font=self._font(10, True))
            style.map(name+'.TButton', background=[('disabled', '#25282e'), ('active', '#ebc98f' if name == 'Run' else '#343b45' if name == 'Secondary' else '#363d48')],
                      foreground=[('disabled', '#737a85')])
        style.configure('Method.TCheckbutton', background='#14181e', foreground=self.INK,
                        padding=(0, 3), indicatorsize=18, indicatormargin=(0, 0, 9, 0))
        style.map('Method.TCheckbutton', background=[('active', '#14181e')],
                  indicatorbackground=[('selected', self.ACCENT), ('!selected', '#0b0d10')],
                  indicatorforeground=[('selected', '#14181e')])
        style.configure('Studio.TNotebook', background=self.BG, bordercolor='#14181e', lightcolor='#14181e', darkcolor='#14181e', borderwidth=0, tabmargins=(0, 0, 0, 8))
        style.layout('Studio.TNotebook.Tab', [('Notebook.padding', {'sticky': 'nswe',
                     'children': [('Notebook.label', {'sticky': 'nswe'})]})])
        style.configure('Studio.TNotebook.Tab', padding=(16, 10), background=self.BG,
                        foreground=self.MUTED, borderwidth=0, font=self._font(10, True))
        style.map('Studio.TNotebook.Tab', background=[('selected', '#14181e')], foreground=[('selected', self.ACCENT)])
        style.configure('Studio.Treeview', background='#14181e', fieldbackground='#14181e', foreground=self.INK,
                        rowheight=35, bordercolor=self.LINE, lightcolor='#14181e', darkcolor='#14181e', borderwidth=0, relief='flat', font=self._font(10))
        style.configure('Studio.Treeview.Heading', background='#0b0d10', foreground=self.MUTED,
                        borderwidth=0, relief='flat', padding=(10, 9), font=self._font(8, True))
        style.map('Studio.Treeview', background=[('selected', '#463923')], foreground=[('selected', '#edc989')])
        style.layout('Studio.Vertical.TScrollbar', [('Vertical.Scrollbar.trough', {'sticky': 'ns',
                     'children': [('Vertical.Scrollbar.thumb', {'expand': '1', 'sticky': 'nswe'})]})])
        style.configure('Studio.Vertical.TScrollbar', background='#40464f', troughcolor=self.BG,
                        bordercolor=self.BG, lightcolor=self.BG, darkcolor=self.BG, borderwidth=0, arrowsize=11, width=9, relief='flat')
        style.configure('Accent.Horizontal.TProgressbar', background=self.ACCENT,
                        troughcolor='#303238', bordercolor=self.BG, lightcolor=self.BG, darkcolor=self.BG, borderwidth=0, thickness=3)

    def start(self):
        if self.running:
            return
        try:
            config = build_config(self.base, {k: v.get() for k, v in self.inputs.items()}, self.nsga.get(), self.grape.get())
        except (ValueError, KeyError, TypeError) as error:
            messagebox.showerror('Check inputs', str(error), parent=self.root)
            return
        self.output = PROJECT/'results'/datetime.now().strftime('desktop_%Y%m%d_%H%M%S_%f')
        # Keep the submitted configuration outside the worker's initially empty output.
        submitted = self.output.with_suffix('.json')
        submitted.parent.mkdir(parents=True, exist_ok=True)
        submitted.write_text(json.dumps(config, indent=2)+'\n')
        command = [configured_python(), '-u', str(PROJECT/'robust_optimization.py'), '--config', str(submitted),
                   '--output', str(self.output), '--non-interactive']
        if self.quick.get():
            command.append('--quick')
        env = os.environ.copy()
        env.update(MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1', MPLCONFIGDIR='/tmp/kairos-mpl')
        try:
            self.process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True)
        except OSError as error:
            messagebox.showerror('Could not start', str(error), parent=self.root)
            return
        self.running = True
        self.run_button.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        self.progress.start(12)
        self.chart_data = []
        self.table.delete(*self.table.get_children())
        self.table.configure(height=1)
        self.method_details = {}
        self.selection_detail.set('Selection and eligibility will appear after the run.')
        self.parameter_text.configure(state='normal')
        self.parameter_text.delete('1.0', 'end')
        self.parameter_text.insert('end', 'Run in progress. The selected pulse is not yet known.')
        self.parameter_text.configure(state='disabled')
        self.draw_chart()
        self.winner.set('Running method comparison…')
        self.winner_caption.set('Optimizing selected methods, then checking manufacturing robustness.')
        for value in self.metric_values:
            value.set('—')
        self.quality.set('RUNNING')
        self.quality_badge.configure(bg='#332b20', fg=self.ACCENT)
        self.detail.set('Quick check: not convergence evidence.' if self.quick.get() else 'Full configured optimization budget.')
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.configure(state='disabled')
        self.status.set('Starting • baselines → optional NSGA → optional GRAPE → Monte Carlo')
        process = self.process
        def read_worker():
            for line in process.stdout:
                self.events.put(('log', line))
            self.events.put(('done', process.wait()))
        threading.Thread(target=read_worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'log':
                    self.log.configure(state='normal')
                    self.log.insert('end', value)
                    self.log.see('end')
                    self.log.configure(state='disabled')
                    if value.startswith('Stage '):
                        self.status.set(value.strip())
                else:
                    self.running = False
                    self.progress.stop()
                    self.run_button.configure(state='normal')
                    self.cancel_button.configure(state='disabled')
                    summary = self.output/'summary.json'
                    if summary.exists() and value in (0, 2):
                        self.show_results(summary)
                    else:
                        self.winner.set('Run stopped — no completed result')
                        self.winner_caption.set('See the activity log for details.')
                        self.quality.set('STOPPED')
                        self.quality_badge.configure(bg='#382a1c', fg='#f0bc79')
                        self.status.set(f'Worker exit {value}. See Run log for details. Partial files: {self.output}')
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def show_results(self, path):
        manifest = json.loads(Path(path).read_text())
        self.output = Path(path).parent
        final = manifest['variants']['robust_selected']
        method = manifest.get('selected_method', 'Selected waveform (legacy result)')
        valid = manifest['quality_passed']
        eligible = manifest.get('mc_selection', {}).get('selection_feasible', False)
        self.winner.set(f'{"Best eligible method" if eligible else "Best diagnostic candidate"}: {method}')
        self.winner_caption.set('Selected by final Monte Carlo ranking. Evaluated on independent device samples.'
                                if eligible else 'No tested candidate met the selection constraints. This is not an accepted design.')
        for variable, value in zip(self.metric_values, (final['nominal']['fidelity'],
                                    final['monte_carlo']['mean_fidelity'], final['monte_carlo']['p05_fidelity'])):
            variable.set(f'{100*value:.4f}%')
        self.quality.set('QUALITY PASS' if valid else 'QUALITY FAIL · DIAGNOSTIC')
        self.quality_badge.configure(bg='#18372e' if valid else '#382a1c', fg='#8ad2b1' if valid else '#f0bc79')
        self.table.delete(*self.table.get_children())
        self.table.configure(height=len(manifest['variants']))
        self.method_details = {}
        self.chart_data = []
        for index, (name, row) in enumerate(manifest['variants'].items()):
            mc = row['monte_carlo']
            label = METHOD_LABELS.get(name, name)
            values = (row['nominal']['fidelity'], mc['mean_fidelity'], mc['p05_fidelity'], mc['worst_sample_leakage'])
            selection = row.get('selection')
            eligibility = ('Eligible' if selection['selection_feasible'] else 'Excluded') if selection else ('Not shortlisted' if manifest.get('selected_pulse_spec') else 'Not recorded')
            item = self.table.insert('', 'end', values=(label, f"{row['hardware']['duration_ns']:.3g}", *(f'{100*x:.5f}' for x in values), eligibility),
                                     tags=('final' if name == 'robust_selected' else 'even' if index % 2 == 0 else 'odd',))
            self.method_details[item] = label + ': ' + (
                '; '.join(selection.get('rejection_reasons', [])) or 'Passed the selection constraints.'
                if selection else 'Selection reasons are unavailable in this older result or the candidate was not shortlisted.')
            self.chart_data.append((label, values[0], values[1]))
        self.draw_chart()
        self.detail.set(f"Quality: {'PASS' if valid else 'FAIL — diagnostic candidate; not an accepted design'}. "
                        f"P05 95% lower bound: {100*final['monte_carlo']['p05_fidelity_lcb95']:.5f}%. "
                        f"Validation samples: {final['monte_carlo']['n_samples']}. "
                        'Best among tested eligible candidates under the configured MC criterion; no global-optimum claim.')
        spec = selected_spec_from_manifest(manifest)
        text = '\n\n'.join(f'{label}\n{value}' for label, value in parameter_rows(spec))
        self.parameter_text.configure(state='normal')
        self.parameter_text.delete('1.0', 'end')
        self.parameter_text.insert('end', text)
        self.parameter_text.configure(state='disabled')
        self.selection_detail.set('Select a method row for eligibility details. Reproduction values are in the Selected pulse tab.')
        self.status.set(f'Completed • results saved in {self.output}')

    def copy_parameters(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.parameter_text.get('1.0', 'end-1c'))

    def show_method_detail(self, event=None):
        selection = self.table.selection()
        if selection:
            self.selection_detail.set(self.method_details.get(selection[0], ''))

    def draw_chart(self):
        self.chart.delete('all')
        width = max(self.chart.winfo_width(), 620)
        if not self.chart_data:
            self.chart.configure(height=230)
            self.chart.create_rectangle(0, 0, width, 230, fill='#0e1115', outline='')
            # Quiet pulse motif for the empty state, not simulated data.
            import math
            points = []
            for x in range(100):
                t = (x-50)/16
                points.extend((width/2-100+x*2, 63-21*math.exp(-t*t)*math.cos(t*3)))
            self.chart.create_line(*points, fill='#8c7350', width=3, smooth=True)
            self.chart.create_text(width/2, 120, text='Ready when you are', fill=self.INK,
                                   font=self._font(14, True))
            self.chart.create_text(width/2, 151, text='Enter your parameters, choose methods, then run optimization.',
                                   fill=self.MUTED, font=self._font(10))
            self.chart.create_text(width/2, 183, text='Fidelity comparisons will appear here.',
                                   fill=self.MUTED, font=self._font(9))
            return
        height = 74+len(self.chart_data)*42
        self.chart.configure(height=height)
        left, span = 192, max(240, width-280)
        self.chart.create_text(0, 14, anchor='w', text='Fidelity comparison', fill=self.INK, font=self._font(11, True))
        for index, (label, nominal, mean) in enumerate(self.chart_data):
            y = 43+index*42
            self.chart.create_text(0, y+10, anchor='w', text=label, fill=self.INK, font=self._font(9))
            for dy, value, color in ((0, nominal, self.ACCENT), (13, mean, '#8b7757')):
                self.chart.create_rectangle(left, y+dy, left+span, y+dy+8, fill='#252a32', outline='')
                self.chart.create_rectangle(left, y+dy, left+span*value, y+dy+8, fill=color, outline='')
            self.chart.create_text(left+span+8, y+10, anchor='w', text=f'{100*mean:.4f}%', fill=self.INK, font=self._font(9))
        self.chart.create_text(0, height-8, anchor='w', text='DARK  Nominal     LIGHT  MC mean     /     Full scale 0–100%',
                               fill=self.MUTED, font=self._font(8))

    def cancel(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.status.set('Stopping the worker…')
            self.cancel_button.configure(state='disabled')

    def close(self):
        self.cancel()
        self.root.destroy()

    def load_config(self):
        if self.running:
            return
        path = filedialog.askopenfilename(initialdir=PROJECT, filetypes=[('JSON configuration', '*.json')])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
            values = {k: str(data['gate_duration_ns'] if k == 'tg_ns' else data['device'][k]) for k in INPUTS}
            build_config(data, values, data.get('nsga_enabled', False), data.get('grape', {}).get('enabled', True))
            self.base = data
            for key, value in values.items():
                self.inputs[key].set(value)
            self.nsga.set(data.get('nsga_enabled', False))
            self.grape.set(data.get('grape', {}).get('enabled', True))
            self.status.set(f'Configuration loaded: {path}')
        except (OSError, ValueError, KeyError, TypeError) as error:
            messagebox.showerror('Invalid configuration', str(error), parent=self.root)

    def load_results(self):
        if self.running:
            return
        path = filedialog.askopenfilename(initialdir=PROJECT/'results', title='Choose summary.json', filetypes=[('JSON results', '*.json')])
        if path:
            try:
                self.show_results(path)
            except (OSError, ValueError, KeyError, TypeError) as error:
                messagebox.showerror('Cannot load results', str(error), parent=self.root)

    def open_output(self):
        if self.output and self.output.exists():
            import webbrowser
            webbrowser.open(self.output.as_uri())


def show_startup(root):
    """Type the wordmark without blocking Tk's event loop, then reveal the app."""
    root.title('Kairos')
    root.geometry('1360x900')
    root.minsize(1180, 760)
    root.configure(bg='#000000')
    splash = tk.Frame(root, bg='#000000')
    splash.place(x=0, y=0, relwidth=1, relheight=1)
    brand = tk.Frame(splash, bg='#000000')
    brand.place(relx=.5, rely=.5, anchor='center')
    splash.logo = load_logo(root, 320)
    root.iconphoto(True, splash.logo)
    tk.Label(brand, image=splash.logo, bg='#000000', bd=0).pack(pady=(0, 20))
    # Original 5x7 dot-matrix lettering: portable, with no font installation.
    glyphs = {
        'k': ('10000','10000','10010','10100','11000','10100','10010'),
        'a': ('00000','00000','01110','00001','01111','10001','01111'),
        'i': ('00100','00000','01100','00100','00100','00100','01110'),
        'r': ('00000','00000','10110','11001','10000','10000','10000'),
        'o': ('00000','00000','01110','10001','10001','10001','01110'),
        's': ('00000','00000','01111','10000','01110','00001','11110'),
    }
    text = 'kairos'
    pitch, radius, margin = 10, 3, 8
    wordmark = tk.Canvas(brand, width=35*pitch+2*margin,
                         height=6*pitch+2*margin, bg='#000000',
                         highlightthickness=0, bd=0)
    wordmark.pack()

    def draw_letters(count):
        wordmark.delete('all')
        for index, letter in enumerate(text[:count]):
            for row, cells in enumerate(glyphs[letter]):
                for column, on in enumerate(cells):
                    if on == '1':
                        x = margin+(index*6+column)*pitch
                        y = margin+row*pitch
                        wordmark.create_oval(x-radius, y-radius, x+radius, y+radius,
                                             fill='#d6af70', outline='')

    def reveal():
        OptimizerApp(root)
        splash.destroy()

    def type_letter(count=1):
        draw_letters(count)
        if count < len(text):
            root.after(150, type_letter, count+1)
        else:
            root.after(650, reveal)

    root.after(250, type_letter)
    return wordmark


def main():
    # Use the same installed solver environment as run_all.py when launched directly.
    python = configured_python()
    if Path(python).absolute() != Path(sys.executable).absolute():
        os.execv(python, [python, str(Path(__file__).resolve()), *sys.argv[1:]])
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/kairos-mpl')
    root = tk.Tk()
    show_startup(root)
    root.mainloop()


if __name__ == '__main__':
    main()
