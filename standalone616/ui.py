from __future__ import annotations

import queue
import threading
import traceback
import tkinter as tk
from dataclasses import asdict
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict, List

from core.reasoning_616.engine import LexiconAdapter, EvidenceAdapter, answer_question
from standalone616.config import load_settings
from standalone616.lexicon import load_lexicon
from standalone616.pipeline import health


class Standalone616App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('Clearbox 6-1-6 Standalone')
        self.root.geometry('980x760')
        self.root.minsize(760, 560)

        self.events: queue.Queue[tuple[str, Dict[str, Any]]] = queue.Queue()
        self.runtime: Dict[str, Any] | None = None

        self.status_var = tk.StringVar(value='Loading lexicon and evidence...')
        self._build_ui()
        self._start_runtime_load()
        self.root.after(120, self._poll_events)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

        frame = ttk.Frame(self.root, padding=14)
        frame.pack(fill='both', expand=True)

        header = ttk.Label(frame, text='6-1-6 Reasoning', font=('Segoe UI', 18, 'bold'))
        header.pack(anchor='w')

        subtitle = ttk.Label(
            frame,
            text='Local-only standalone. Reads the live lexicon, reasons over the standalone evidence store.',
        )
        subtitle.pack(anchor='w', pady=(4, 8))

        status = ttk.Label(frame, textvariable=self.status_var)
        status.pack(anchor='w', pady=(0, 10))

        input_label = ttk.Label(frame, text='Input')
        input_label.pack(anchor='w')

        self.input_text = ScrolledText(frame, height=7, wrap='word', font=('Consolas', 11))
        self.input_text.pack(fill='x', expand=False)
        self.input_text.bind('<Control-Return>', self._submit_from_event)

        controls = ttk.Frame(frame)
        controls.pack(fill='x', pady=(8, 10))

        self.ask_button = ttk.Button(controls, text='Reason', command=self.submit_query, state='disabled')
        self.ask_button.pack(side='left')

        clear_button = ttk.Button(controls, text='Clear', command=self.clear_all)
        clear_button.pack(side='left', padx=(8, 0))

        hint = ttk.Label(controls, text='Ctrl+Enter to run')
        hint.pack(side='left', padx=(12, 0))

        output_label = ttk.Label(frame, text='Output')
        output_label.pack(anchor='w')

        self.output_text = ScrolledText(frame, height=26, wrap='word', font=('Consolas', 11))
        self.output_text.pack(fill='both', expand=True)
        self.output_text.configure(state='disabled')

    def _start_runtime_load(self) -> None:
        thread = threading.Thread(target=self._load_runtime_worker, daemon=True)
        thread.start()

    def _load_runtime_worker(self) -> None:
        try:
            settings = load_settings()
            lexicon = load_lexicon(settings.lexicon_root)
            payload = {
                'settings': settings,
                'lexicon': lexicon,
                'health': health(),
            }
            self.events.put(('loaded', payload))
        except Exception as exc:
            self.events.put(
                (
                    'error',
                    {
                        'stage': 'runtime',
                        'message': str(exc),
                        'traceback': traceback.format_exc(),
                    },
                )
            )

    def _submit_from_event(self, _event=None):
        self.submit_query()
        return 'break'

    def submit_query(self) -> None:
        question = self.input_text.get('1.0', 'end').strip()
        if not question:
            self.status_var.set('Enter a question first.')
            return
        if self.runtime is None:
            self.status_var.set('Still loading runtime...')
            return

        self.ask_button.configure(state='disabled')
        self.status_var.set('Running 6-1-6 reasoning...')
        self._set_output('')
        thread = threading.Thread(target=self._query_worker, args=(question,), daemon=True)
        thread.start()

    def _query_worker(self, question: str) -> None:
        try:
            assert self.runtime is not None
            lexicon = self.runtime['lexicon']
            result = answer_question(question, LexiconAdapter(lexicon), EvidenceAdapter())
            formatted = format_query_result(question, result, lexicon)
            self.events.put(('result', {'question': question, 'text': formatted}))
        except Exception as exc:
            self.events.put(
                (
                    'error',
                    {
                        'stage': 'query',
                        'message': str(exc),
                        'traceback': traceback.format_exc(),
                    },
                )
            )

    def clear_all(self) -> None:
        self.input_text.delete('1.0', 'end')
        self._set_output('')
        if self.runtime is not None:
            summary = runtime_summary(self.runtime)
            self.status_var.set(summary)
        else:
            self.status_var.set('Loading lexicon and evidence...')

    def _set_output(self, text: str) -> None:
        self.output_text.configure(state='normal')
        self.output_text.delete('1.0', 'end')
        self.output_text.insert('1.0', text)
        self.output_text.configure(state='disabled')

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if kind == 'loaded':
                self.runtime = payload
                self.ask_button.configure(state='normal')
                self.status_var.set(runtime_summary(payload))
                self._set_output(
                    'Ready. Enter a question above and press Reason.\n\n'
                    'Example:\n'
                    'describe lexicon'
                )
            elif kind == 'result':
                self.ask_button.configure(state='normal')
                self.status_var.set(runtime_summary(self.runtime or payload))
                self._set_output(payload['text'])
            elif kind == 'error':
                self.ask_button.configure(state='normal')
                message = payload['message']
                stage = payload['stage']
                self.status_var.set(f'{stage} failed: {message}')
                self._set_output(payload['traceback'])

        self.root.after(120, self._poll_events)


def runtime_summary(runtime: Dict[str, Any]) -> str:
    settings = runtime['settings']
    stats = runtime['health']
    evidence = stats.get('evidence', {})
    return (
        f'Lexicon: {runtime["lexicon"].word_count:,} words | '
        f'Evidence: {evidence.get("cells", 0):,} cells | '
        f'Window: {settings.window} | '
        f'Maps: {stats.get("maps", 0):,} | '
        f'Receipts: {stats.get("receipts", 0):,}'
    )


def format_query_result(question: str, result, lexicon) -> str:
    lines: List[str] = []
    lines.append(f'Question: {question}')
    lines.append(f'Type: {result.query_type.value}')
    lines.append(f'Score: {result.score:.4f}')
    lines.append('')
    lines.append('Answer')
    lines.append(result.answer_text or '(no answer)')

    supporting_words = [
        lexicon.entries[symbol].word
        for symbol in result.supporting_symbols
        if symbol in lexicon.entries and lexicon.entries[symbol].word
    ]
    if supporting_words:
        lines.append('')
        lines.append('Supporting words')
        lines.append(', '.join(supporting_words[:20]))

    metrics = asdict(result.support_metrics)
    lines.append('')
    lines.append('Support metrics')
    for key, value in metrics.items():
        lines.append(f'  {key}: {value:.4f}')

    if result.uncertainty_notes:
        lines.append('')
        lines.append('Uncertainty')
        for note in result.uncertainty_notes:
            lines.append(f'  - {note}')

    if result.ranked_matches:
        lines.append('')
        lines.append('Top matches')
        for match in result.ranked_matches[:8]:
            word_text = match.candidate_word
            if not word_text and match.candidate_symbol_id in lexicon.entries:
                word_text = lexicon.entries[match.candidate_symbol_id].word or ''
            lines.append(
                f'  - {word_text or "(unknown)"} | {match.candidate_symbol_id} | '
                f'score={match.final_score:.4f} overlap={match.overlap_score:.4f} '
                f'directional={match.directional_match_score:.4f}'
            )

    return '\n'.join(lines)


def main() -> int:
    root = tk.Tk()
    app = Standalone616App(root)
    app.input_text.focus_set()
    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())