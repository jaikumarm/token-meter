"""Read-only Antigravity CLI usage; conversation content never leaves the parser."""

import json
import os
import re
import tempfile
import time
from bisect import bisect_right
from datetime import datetime
from itertools import islice
from pathlib import Path
from uuid import UUID

from token_meter.contracts import (
    DeletionPlan, DetailLevel, EvidenceBasis, EvidenceValue, ModelRef,
    NormalizedSession, ParseWarning, RuntimeDescriptor, SessionSource,
    SourceLocator, SourceRevision, TimingEvidence, TurnSummary, UsageEvidence,
)

MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MAX_STEPS = 10_000
MAX_SOURCES = 2_000
UNKNOWN_MODEL = 'unknown-model'
# shortcut: exported history has no model names; remove the hook if upstream records them.


def _integer(value):
    return value if type(value) is int and 0 <= value <= 2 ** 53 - 1 else None


def _session_id(value):
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        raise ValueError('Invalid Antigravity conversation identity') from None


def _model(value):
    if not isinstance(value, str) or not re.fullmatch(
            r'(?:gemini|claude|gpt)-[a-z0-9][a-z0-9._-]{0,100}', value):
        return UNKNOWN_MODEL
    return value


def _model_ref(value):
    provider = {'gemini': 'google', 'claude': 'anthropic', 'gpt': 'openai'}.get(
        str(value).split('-')[0], 'unknown-model-provider')
    return ModelRef(provider, _model(value))


def _timestamp(value):
    if not isinstance(value, str):
        return 0.0
    try:
        return max(0.0, datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp())
    except (ValueError, OverflowError, OSError):
        return 0.0


def _date(value):
    return datetime.fromtimestamp(value).astimezone() if value else None


def _signature(path):
    try:
        info = path.stat()
        return str(info.st_mtime_ns), str(info.st_size)
    except OSError:
        return '0', '0'


def record_model(payload, metadata_dir=None, app_data_dir=None):
    """Record only the completed steps observed by a PostInvocation hook."""
    if not isinstance(payload, dict):
        raise ValueError('Invalid Antigravity hook metadata')
    session = _session_id(payload.get('conversationId'))
    start = _integer(payload.get('initialNumSteps'))
    model = _model(payload.get('modelName'))
    if start is None or start >= MAX_STEPS or model == UNKNOWN_MODEL:
        raise ValueError('Invalid Antigravity model boundary')
    root = Path(metadata_dir or Path.home() / '.token-meter/antigravity')
    app_root = Path(app_data_dir or Path.home() / '.gemini/antigravity-cli')
    adapter = AntigravityRuntimeAdapter(app_root, metadata_dir=root)
    rows, warnings = adapter._parsed(
        app_root / 'brain' / session / '.system_generated/logs/transcript.jsonl')
    if any(code != 'model_unavailable' for code in warnings):
        raise ValueError('Incomplete Antigravity invocation evidence')
    completed = [row['index'] for row in rows if row['index'] >= start]
    if not completed:
        raise ValueError('Antigravity invocation has not been exported yet')
    directory = root / session
    if root.is_symlink() or directory.is_symlink():
        raise ValueError('Invalid Antigravity metadata directory')
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    content = json.dumps({'start_step': start, 'end_step': max(completed),
                          'model': model}, sort_keys=True) + '\n'
    descriptor, temporary = tempfile.mkstemp(prefix='.model-', dir=directory)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(content)
        os.replace(temporary, directory / '{}.json'.format(start))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class AntigravityRuntimeAdapter:
    descriptor = RuntimeDescriptor(
        'antigravity', 'Antigravity CLI', frozenset(('sessions', 'models')),
        'runtime.generic', 'runtime-neutral', None,
    )

    def __init__(self, app_data_dir, *, metadata_dir=None, compatibility=None,
                 path_cache=None):
        self.root = Path(app_data_dir).expanduser().absolute()
        self.metadata_dir = Path(metadata_dir or Path.home() / '.token-meter/antigravity')
        self.compatibility = dict(compatibility or {})
        self.path_cache = path_cache

    def _owned(self, path):
        path = Path(path)
        try:
            session = _session_id(path.parents[2].name)
            expected = self.root / 'brain' / session / '.system_generated/logs/transcript.jsonl'
            return (path == expected and not path.is_symlink()
                    and path.resolve().is_relative_to(self.root.resolve()))
        except (ValueError, IndexError, OSError, RuntimeError):
            return False

    def _boundaries(self, session):
        directory = self.metadata_dir / session
        if self.metadata_dir.is_symlink() or directory.is_symlink():
            return ()
        boundaries = []
        try:
            paths = islice(directory.glob('*.json'), MAX_STEPS)
            for path in paths:
                if path.is_symlink() or path.stat().st_size > 512:
                    continue
                try:
                    row = json.loads(path.read_text(encoding='utf-8'))
                    start = _integer(row.get('start_step'))
                    end = _integer(row.get('end_step'))
                    model = _model(row.get('model'))
                    if (start is not None and end is not None and start <= end < MAX_STEPS
                            and model != UNKNOWN_MODEL
                            and path.name == '{}.json'.format(start)):
                        boundaries.append((start, end, model))
                except (ValueError, TypeError, AttributeError, UnicodeError):
                    continue
        except OSError:
            return ()
        return tuple(sorted(boundaries))

    def _revision(self, path):
        path = Path(path)
        return SourceRevision(('antigravity-jsonl', *_signature(path),
            *_signature(self.metadata_dir / path.parents[2].name)))

    def current_revision(self, source):
        path = source.locator.value if isinstance(source, SessionSource) else source['path']
        return self._revision(path)

    def _record(self, path):
        path = Path(path)
        if not self._owned(path) or not path.is_file():
            return None
        boundaries = self._boundaries(path.parents[2].name)
        model = UNKNOWN_MODEL
        if (boundaries and self.metadata_dir.joinpath(path.parents[2].name).stat().st_mtime_ns
                >= path.stat().st_mtime_ns):
            model = boundaries[-1][2]
        return {'provider': 'antigravity', 'client': 'antigravity',
                'label': self.descriptor.label, 'runtime': self.descriptor.label,
                'id': path.parents[2].name, 'session': path.parents[2].name,
                'path': str(path), 'project': '', 'title': 'Antigravity session',
                'metadata_path': str(self.metadata_dir / path.parents[2].name),
                'model': model, 'model_provider': _model_ref(model).provider_id,
                'mtime': path.stat().st_mtime,
                'signature_mtime': path.stat().st_mtime,
                'request_revision': '|'.join(self._revision(path).parts),
                'source_kind': 'antigravity_jsonl'}

    def _paths(self):
        pattern = str(self.root / 'brain/*/.system_generated/logs/transcript.jsonl')
        paths = (self.path_cache.paths(pattern) if self.path_cache else self.root.glob(
            'brain/*/.system_generated/logs/transcript.jsonl'))
        return tuple(str(path) for path in islice(paths, MAX_SOURCES))

    def discover_legacy(self, context):
        del context
        records = []
        for path in self._paths():
            if len(records) >= MAX_SOURCES:
                break
            try:
                record = self._record(path)
                if record:
                    records.append(record)
            except OSError:
                continue
        return tuple(sorted(records, key=lambda row: (-row['mtime'], row['id'])))

    def discover(self, context):
        return tuple(SessionSource(
            runtime_id='antigravity', client_id='antigravity', session_id=row['id'],
            display_label=self.descriptor.label, project=None,
            locator=SourceLocator('jsonl', row['path']), activity_mtime=row['mtime'],
            revision=self._revision(row['path']), model_ref=_model_ref(row['model']),
        ) for row in self.discover_legacy(context))

    def _parsed(self, path):
        path = Path(path)
        if not self._owned(path) or not path.is_file():
            raise ValueError('Antigravity session is unavailable')
        warnings = set()
        steps = {}
        boundaries = self._boundaries(path.parents[2].name)
        boundary_steps = tuple(start for start, end, model in boundaries)
        if path.stat().st_size > MAX_JSON_BYTES:
            return (), ('source_limit',)
        try:
            with path.open('rb') as stream:
                consumed = 0
                rows = 0
                while True:
                    line = stream.readline(MAX_LINE_BYTES + 1)
                    if not line:
                        break
                    consumed += len(line)
                    rows += 1
                    if consumed > MAX_JSON_BYTES or rows > MAX_STEPS * 4:
                        warnings.add('source_limit')
                        break
                    if len(line) > MAX_LINE_BYTES:
                        warnings.add('record_limit')
                        while line and not line.endswith(b'\n'):
                            line = stream.readline(MAX_LINE_BYTES + 1)
                            consumed += len(line)
                            if consumed > MAX_JSON_BYTES:
                                break
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError()
                    except (ValueError, UnicodeError, RecursionError):
                        warnings.add('invalid_record')
                        continue
                    if row.get('type') != 'PLANNER_RESPONSE' or row.get('source') != 'MODEL' or row.get('status') != 'DONE':
                        continue
                    index = _integer(row.get('step_index'))
                    if index is None or index >= MAX_STEPS:
                        warnings.add('invalid_step')
                        continue
                    boundary = bisect_right(boundary_steps, index) - 1
                    model = (boundaries[boundary][2] if boundary >= 0
                             and index <= boundaries[boundary][1] else UNKNOWN_MODEL)
                    # Only allowlisted scalar metadata survives this iteration.
                    steps[index] = {'index': index, 'timestamp': _timestamp(row.get('created_at')),
                        'model': model, **{key: _integer(row.get(key)) for key in
                        ('input_tokens', 'output_tokens', 'cache_read_tokens')}}
        except OSError:
            raise ValueError('Antigravity session is unavailable') from None
        if not steps:
            warnings.add('usage_unavailable')
        if any(row['model'] == UNKNOWN_MODEL for row in steps.values()):
            warnings.add('model_unavailable')
        return tuple(steps[index] for index in sorted(steps)), tuple(sorted(warnings))

    @staticmethod
    def _evidence(rows, field, incomplete=False):
        values = [row[field] for row in rows]
        if incomplete or not values or any(value is None for value in values):
            return EvidenceValue.unavailable()
        return EvidenceValue(sum(values), EvidenceBasis.MEASURED)

    def load(self, source, detail):
        if isinstance(source, dict):
            return self.recompute_legacy(source)
        if not isinstance(source, SessionSource) or source.runtime_id != 'antigravity':
            raise ValueError('Source belongs to another runtime')
        rows, warnings = self._parsed(source.locator.value)
        incomplete = any(code != 'model_unavailable' for code in warnings)
        return NormalizedSession(
            source=source, started_at=_date(min((r['timestamp'] for r in rows), default=0)),
            ended_at=_date(max((r['timestamp'] for r in rows), default=0)),
            usage=UsageEvidence(self._evidence(rows, 'input_tokens', incomplete),
                self._evidence(rows, 'output_tokens', incomplete),
                self._evidence(rows, 'cache_read_tokens', incomplete),
                EvidenceValue.unavailable(), EvidenceValue.unavailable()),
            timing=TimingEvidence.unavailable(), tools=(),
            turns=tuple(TurnSummary(r['index'], _date(r['timestamp']), _date(r['timestamp']),
                self._evidence((r,), 'output_tokens')) for r in rows),
            pricing_basis=None, capabilities=self.descriptor.capabilities,
            warnings=tuple(ParseWarning(code, 'Antigravity evidence is partial or unavailable.') for code in warnings),
            detail=detail)

    def _availability(self, rows, incomplete=False):
        complete = lambda field: not incomplete and bool(rows) and all(r[field] is not None for r in rows)
        input_available = complete('input_tokens') and complete('cache_read_tokens')
        return self.compatibility['metric_availability']('antigravity',
            cost=False, tokens=input_available and complete('output_tokens'),
            input_tokens=input_available, output_tokens=complete('output_tokens'),
            cache=False, timing=False, throughput=False, context=False, tool_results=False)

    def _summary(self, source, rows, incomplete=False):
        compat = self.compatibility
        stats, daily = {}, {}
        for row in rows:
            usage = {'input_tokens': row['input_tokens'] or 0,
                'output_tokens': row['output_tokens'] or 0,
                'cache_read_input_tokens': row['cache_read_tokens'] or 0,
                'input_available': not incomplete and row['input_tokens'] is not None
                    and row['cache_read_tokens'] is not None,
                'output_available': not incomplete and row['output_tokens'] is not None}
            compat['add_model_summary'](stats, row['model'], usage, 0, cost_available=False)
            compat['add_model_daily'](daily, row['model'], usage, 0, row['timestamp'], cost_available=False)
        for values in (*stats.values(), *daily.values()):
            relevant = [r for r in rows if r['model'] == values.get('model')] if 'model' in values else rows
            values['availability'] = self._availability(relevant, incomplete)
        model_tokens = {model: values['tokens'] for model, values in stats.items()}
        timestamps = [r['timestamp'] for r in rows if r['timestamp']]
        return compat['summary_row'](source, 'Antigravity session', 0.0,
            sum(model_tokens.values()), len(rows), set(stats), min(timestamps, default=0),
            max(timestamps, default=0), {}, model_tokens, {}, False,
            input_tokens=sum(v['input_tokens'] for v in stats.values()),
            output_tokens=sum(v['output_tokens'] for v in stats.values()),
            model_stats=stats, model_daily=list(daily.values()),
            availability=self._availability(rows, incomplete))

    def summarize_legacy(self, source, unused=None):
        del unused
        rows, warnings = self._parsed(source['path'])
        summary = self._summary(source, rows, any(code != 'model_unavailable' for code in warnings))
        summary['evidence_warnings'] = list(warnings)
        return summary

    def recompute_legacy(self, source):
        rows, warnings = self._parsed(source['path'])
        incomplete = any(code != 'model_unavailable' for code in warnings)
        compat = self.compatibility
        totals = {'input': sum(r['input_tokens'] or 0 for r in rows),
            'output': sum(r['output_tokens'] or 0 for r in rows),
            'cache_read': sum(r['cache_read_tokens'] or 0 for r in rows), 'cache_write': 0}
        summary = self._summary(source, rows, incomplete)
        series, executions = [], []
        for index, row in enumerate(rows, 1):
            input_tokens, output, cache = (row[key] or 0 for key in
                ('input_tokens', 'output_tokens', 'cache_read_tokens'))
            available = self._availability((row,))
            series.append({'i': index, 'in': input_tokens, 'out': output, 'cost': 0.0,
                'fresh_input': input_tokens, 'cache': cache, 'cache_read': cache,
                'cache_write': 0, 'think': False, 'tools': 0, 'side': False,
                'reasoning': 0, 'reasoning_ms': 0, 'context_pct': None,
                'context_tokens': 0, 'user_message': '', 'user_input': '', 'availability': available})
            executions.append({'id': '{}:{}'.format(source['id'], row['index']), 'idx': index,
                'ts': row['timestamp'], 'time': time.strftime('%H:%M', time.localtime(row['timestamp'])),
                'model': row['model'], 'tokens': {'input': input_tokens, 'output': output,
                    'fresh_input': input_tokens, 'cache': cache, 'cache_read': cache,
                    'cache_write': 0, 'reasoning': 0, 'retrieval': 0, 'total': input_tokens + output + cache},
                'cost': 0.0, 'cost_breakdown': dict.fromkeys(totals, 0.0),
                'tools': [], 'tool_count': 0, 'model_calls': 1, 'reasoning_tokens': 0,
                'reasoning_duration_ms': 0, 'duration_ms': None, 'wait_duration_ms': None,
                'context_tokens': 0, 'context_window': 0, 'context_pct': None,
                'summary': 'Antigravity model invocation', 'user_message': '', 'user_input': '',
                'availability': available})
        analyses = compat['analysis_block'](totals, 0.0, 0, 0, 0.0,
            summary['_model_tok'], dict.fromkeys(summary['_model_tok'], 0.0),
            compat['tool_summary'](executions), 0.0, 0, len(rows))
        timestamps = [r['timestamp'] for r in rows if r['timestamp']]
        state = compat['build_state']({**source, 'cache_savings_available': False},
            totals, dict.fromkeys(totals, 0.0), sum(totals.values()), 0.0, series, executions, [],
            {'reasoning': 0, 'output': totals['output'], 'retrieval': 0, 'coordination': 0},
            analyses, [], min(timestamps, default=0), max(timestamps, default=0), 0, None, 0,
            False, rows[-1]['model'] if rows else UNKNOWN_MODEL,
            'Recorded Antigravity tokens; model names require invocation-hook evidence. '
            'Dollar cost, reasoning-token split, timing and cache-write counts are unavailable.',
            availability=self._availability(rows, incomplete))
        state['evidence_warnings'] = list(warnings)
        return state

    def deletion_plan(self, source):
        del source
        return DeletionPlan.deny('Antigravity session storage is read-only.')


class AntigravityRuntimeAdapterProxy:
    descriptor = AntigravityRuntimeAdapter.descriptor

    def __init__(self, adapter_factory):
        self._adapter_factory = adapter_factory

    def discover(self, context):
        return self._adapter_factory().discover(context)

    def discover_legacy(self, context):
        return self._adapter_factory().discover_legacy(context)

    def current_revision(self, source):
        return self._adapter_factory().current_revision(source)

    def load(self, source, detail):
        return self._adapter_factory().load(source, detail)

    def summarize_legacy(self, source, unused=None):
        return self._adapter_factory().summarize_legacy(source, unused)

    def deletion_plan(self, source):
        return self._adapter_factory().deletion_plan(source)
