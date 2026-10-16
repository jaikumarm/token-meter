import io
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import runpy
import sys
from types import SimpleNamespace
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import meter
from token_meter.contracts import DetailLevel, DiscoveryContext, EvidenceBasis
from token_meter.runtimes.antigravity import AntigravityRuntimeAdapter, record_model


SESSION = '11111111-2222-4333-8444-555555555555'


def step(index=1, **changes):
    row = {'step_index': index, 'type': 'PLANNER_RESPONSE', 'source': 'MODEL',
           'status': 'DONE', 'created_at': '2026-10-14T12:00:00Z',
           'input_tokens': 100, 'output_tokens': 20, 'cache_read_tokens': 200,
           'content': 'PRIVATE_RESPONSE', 'thinking': 'PRIVATE_REASONING'}
    row.update(changes)
    return row


class AntigravityAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / '.gemini/antigravity-cli'
        self.metadata = Path(self.tmp.name) / '.token-meter/antigravity'
        self.path = self.root / 'brain' / SESSION / '.system_generated/logs/transcript.jsonl'
        self.path.parent.mkdir(parents=True)
        self.adapter = AntigravityRuntimeAdapter(self.root, metadata_dir=self.metadata,
            compatibility={name: getattr(meter, name) for name in (
                'summary_row', 'build_state', 'metric_availability', 'analysis_block',
                'tool_summary', 'add_model_summary', 'add_model_daily')})
        self.context = DiscoveryContext(home=self.tmp.name)

    def write(self, *rows):
        self.path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def source(self):
        return self.adapter.discover(self.context)[0]

    def capture(self, start, model, **changes):
        payload = {'conversationId': SESSION, 'initialNumSteps': start,
                   'invocationNum': start, 'modelName': model}
        payload.update(changes)
        return record_model(payload, self.metadata, self.root)

    def test_discovery_ignores_full_transcript_and_other_files(self):
        self.write(step())
        self.path.with_name('transcript_full.jsonl').write_text(json.dumps(step()))
        sources = self.adapter.discover(self.context)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].runtime_id, 'antigravity')
        self.assertEqual(sources[0].model_ref.model_id, 'unknown-model')
        loaded = self.adapter.load(sources[0], DetailLevel.FULL)
        self.assertEqual(loaded.usage.input_tokens.value, 100)
        self.assertEqual(loaded.usage.output_tokens.value, 20)
        self.assertEqual(loaded.usage.cache_read_tokens.value, 200)
        self.assertIsNone(loaded.usage.cost_usd.value)
        self.assertIsNone(loaded.usage.cache_write_tokens.value)
        self.assertNotIn('PRIVATE', repr(loaded))
        self.assertEqual(self.adapter.deletion_plan(sources[0]).targets, ())

    def test_steps_are_per_call_and_duplicates_count_once(self):
        self.write(step(), step(), step(2, input_tokens=10, output_tokens=2))
        loaded = self.adapter.load(self.source(), DetailLevel.SUMMARY)
        self.assertEqual(loaded.usage.input_tokens.value, 110)
        self.assertEqual(loaded.usage.output_tokens.value, 22)
        self.assertEqual(len(loaded.turns), 2)

    def test_active_rows_do_not_replace_completed_steps(self):
        self.write(step(), step(status='ACTIVE', input_tokens=999),
                   step(2, status='ACTIVE'), step(3, type='GENERIC', input_tokens=999))
        loaded = self.adapter.load(self.source(), DetailLevel.SUMMARY)
        self.assertEqual(loaded.usage.input_tokens.value, 100)
        self.assertEqual(len(loaded.turns), 1)

    def test_latest_completed_revision_replaces_same_step(self):
        self.write(step(), step(output_tokens=25))
        self.assertEqual(self.adapter.load(self.source(), DetailLevel.SUMMARY).usage.output_tokens.value, 25)

    def test_missing_invalid_and_zero_usage_stay_distinct(self):
        for value in (None, -1, True, '5', 1.5):
            with self.subTest(value=value):
                self.write(step(input_tokens=value))
                loaded = self.adapter.load(self.source(), DetailLevel.SUMMARY)
                self.assertEqual(loaded.usage.input_tokens.basis, EvidenceBasis.UNAVAILABLE)
                self.assertEqual(loaded.usage.output_tokens.value, 20)
        self.write(step(input_tokens=0))
        self.assertEqual(self.adapter.load(self.source(), DetailLevel.SUMMARY).usage.input_tokens.value, 0)

    def test_hook_boundaries_preserve_unknown_history_and_model_switches(self):
        self.write(step(1), step(3))
        self.capture(3, 'gemini-3.8-flash-medium')
        self.write(step(1), step(3), step(5))
        self.capture(5, 'gemini-3.8-pro')
        source = self.adapter.discover_legacy(self.context)[0]
        summary = self.adapter.summarize_legacy(source)
        models = {row['model']: row for row in summary['model_stats']}
        self.assertEqual(set(models), {'unknown-model', 'gemini-3.8-flash-medium', 'gemini-3.8-pro'})
        self.assertTrue(all(row['tokens'] == 320 for row in models.values()))
        self.assertTrue(all(row['availability']['cost'] is False for row in models.values()))
        self.assertEqual(sum(row['input_tokens'] + row['output_tokens']
                             for row in summary['_model_daily']), 960)
        state = self.adapter.load(source, DetailLevel.FULL)
        self.assertEqual(state['total_tokens'], 960)
        self.assertNotIn('PRIVATE', json.dumps((summary, state)))

    def test_revision_changes_with_usage_and_model_boundary(self):
        self.write(step())
        source = self.source()
        before = self.adapter.current_revision(source)
        self.capture(0, 'gemini-3.8-pro')
        self.assertNotEqual(before, self.adapter.current_revision(source))
        after = self.adapter.current_revision(source)
        self.write(step(), step(2))
        self.assertNotEqual(after, self.adapter.current_revision(source))

    def test_corrupt_and_partial_lines_are_bounded_warnings(self):
        self.write(step())
        with self.path.open('a') as stream:
            stream.write('{invalid}\n{"unfinished":')
        loaded = self.adapter.load(self.source(), DetailLevel.SUMMARY)
        self.assertIsNone(loaded.usage.input_tokens.value)
        self.assertEqual(loaded.turns[0].output_tokens.value, 20)
        self.assertTrue(loaded.warnings)
        source = self.adapter.discover_legacy(self.context)[0]
        summary = self.adapter.summarize_legacy(source)
        state = self.adapter.load(source, DetailLevel.FULL)
        for row in (summary, state, *summary['model_stats'], *summary['_model_daily']):
            self.assertFalse(row['availability']['tokens'])
        self.assertEqual(summary['model_stats'][0]['io_covered_executions'], 0)

    def test_missing_cache_invalidates_inclusive_input_but_zero_is_measured(self):
        for cache, available in ((None, False), (0, True)):
            with self.subTest(cache=cache):
                self.write(step(cache_read_tokens=cache))
                source = self.adapter.discover_legacy(self.context)[0]
                summary = self.adapter.summarize_legacy(source)
                state = self.adapter.load(source, DetailLevel.FULL)
                native = self.adapter.load(self.source(), DetailLevel.SUMMARY)
                self.assertEqual(native.usage.input_tokens.value, 100)
                self.assertEqual(native.usage.cache_read_tokens.value, cache)
                for row in (summary, state, *summary['model_stats'], *summary['_model_daily']):
                    self.assertEqual(row['availability']['tokens'], available)
                    self.assertEqual(row['availability']['input_tokens'], available)
                    self.assertTrue(row['availability']['output_tokens'])
                self.assertEqual(summary['model_stats'][0]['io_covered_executions'], int(available))

    def test_empty_or_missing_usage_is_unavailable(self):
        self.write({'type': 'USER_INPUT', 'step_index': 0, 'content': 'PRIVATE_PROMPT'})
        loaded = self.adapter.load(self.source(), DetailLevel.SUMMARY)
        self.assertIsNone(loaded.usage.input_tokens.value)
        self.assertTrue(loaded.warnings)

    def test_symlinks_and_sources_outside_root_are_rejected(self):
        self.write(step())
        self.path.rename(self.path.with_name('outside.jsonl'))
        self.path.symlink_to(self.path.with_name('outside.jsonl'))
        self.assertEqual(self.adapter.discover(self.context), ())

    def test_missing_file_after_discovery_fails_without_private_error(self):
        self.write(step())
        source = self.source()
        self.path.unlink()
        with self.assertRaises(ValueError):
            self.adapter.load(source, DetailLevel.SUMMARY)

    def test_capture_persists_only_allowlisted_metadata_and_is_idempotent(self):
        self.write(step())
        self.capture(1, 'gemini-3.8-pro', content='PRIVATE', email='PRIVATE',
                     workspacePaths=['PRIVATE'], transcriptPath='PRIVATE')
        self.capture(1, 'gemini-3.8-pro')
        files = list(self.metadata.rglob('*.json'))
        self.assertEqual(len(files), 1)
        saved = json.loads(files[0].read_text())
        self.assertEqual(saved, {'start_step': 1, 'end_step': 1, 'model': 'gemini-3.8-pro'})
        self.assertNotIn('PRIVATE', files[0].read_text())

    def test_capture_rejects_paths_credentials_and_invalid_boundaries(self):
        for changes in ({'conversationId': '../escape'}, {'modelName': 'sk-private'},
                        {'initialNumSteps': -1}, {'initialNumSteps': True},
                        {'modelName': 'gemini-3.8-pro; echo PRIVATE'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.capture(1, 'gemini-3.8-pro', **changes)
        self.assertFalse(self.metadata.exists())

    def test_capture_never_attributes_later_steps_after_hook_stops(self):
        self.write(step())
        self.capture(0, 'gemini-3.8-pro')
        self.write(step(), step(3))
        summary = self.adapter.summarize_legacy(self.adapter.discover_legacy(self.context)[0])
        self.assertEqual(set(summary['models']), {'gemini-3.8-pro', 'unknown-model'})

    def test_model_metadata_creation_triggers_inventory_refresh(self):
        self.write(step())
        source = self.adapter.discover_legacy(self.context)[0]
        targets = meter.source_inventory_probe_targets([source],
            roots=[str(self.root / 'brain'), str(self.metadata)], extra_files=[])
        before = meter.source_inventory_probe_signature(targets, current_path=str(self.path))
        self.capture(0, 'gemini-3.8-pro')
        after = meter.source_inventory_probe_signature(targets, current_path=str(self.path))
        self.assertNotEqual(before, after)
        refreshed = self.adapter.discover_legacy(self.context)[0]
        self.assertNotEqual(meter.source_revision_signature(source), meter.source_revision_signature(refreshed))

    def test_candidate_paths_and_budget_registration_include_antigravity(self):
        self.write(step())
        empty = mock.Mock()
        for name in ('_glob', 'desktop_metadata_paths', '_paths', '_transcript_paths'):
            getattr(empty, name).return_value = ()
        empty.projects_root = empty.sessions_root = Path('/nonexistent')
        empty.agent_storage_root = None
        with ExitStack() as patches:
            for name in ('_claude_native_adapter', '_codex_native_adapter',
                         '_cursor_native_adapter', '_kiro_native_adapter'):
                patches.enter_context(mock.patch.object(meter, name, return_value=empty))
            patches.enter_context(mock.patch.object(meter, '_antigravity_native_adapter', return_value=self.adapter))
            patches.enter_context(mock.patch.object(meter, 'ANTIGRAVITY_DATA_DIR', str(self.root)))
            self.assertEqual(meter.runtime_candidate_paths(), (str(self.path),))
            self.assertIn(str(self.root / 'brain'), meter._source_inventory_roots())
            self.assertIn(str(self.metadata), meter._source_inventory_roots())
        budget = meter.normalize_budget_settings({'allocations': {'antigravity': 5}})
        self.assertEqual(budget['allocations']['antigravity'], 5)

    def test_file_size_limit_marks_usage_unavailable(self):
        self.write(step())
        with mock.patch('token_meter.runtimes.antigravity.MAX_JSON_BYTES', 8):
            loaded = self.adapter.load(self.source(), DetailLevel.SUMMARY)
        self.assertIsNone(loaded.usage.input_tokens.value)
        self.assertTrue(loaded.warnings)

    def test_installed_hook_script_records_metadata_and_returns_empty_response(self):
        self.write(step())
        payload = {'conversationId': SESSION, 'initialNumSteps': 1,
                   'invocationNum': 0, 'modelName': 'gemini-3.8-pro',
                   'content': 'PRIVATE', 'email': 'PRIVATE'}
        output = io.StringIO()
        script = Path(__file__).resolve().parents[2] / 'scripts/record-antigravity-model'
        with mock.patch.object(Path, 'home', return_value=Path(self.tmp.name)), \
                mock.patch.object(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(json.dumps(payload).encode()))), \
                mock.patch.object(sys, 'path', list(sys.path)), redirect_stdout(output):
            runpy.run_path(str(script))
        self.assertEqual(json.loads(output.getvalue()), {})
        saved = list(self.metadata.rglob('*.json'))
        self.assertEqual(len(saved), 1)
        self.assertNotIn('PRIVATE', saved[0].read_text())

    def test_hook_script_fails_open_on_invalid_or_oversized_input(self):
        script = Path(__file__).resolve().parents[2] / 'scripts/record-antigravity-model'
        for raw in (b'{invalid', b'x' * (64 * 1024 + 1)):
            output, errors = io.StringIO(), io.StringIO()
            with self.subTest(size=len(raw)), \
                    mock.patch.object(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(raw))), \
                    mock.patch.object(sys, 'path', list(sys.path)), \
                    redirect_stdout(output), redirect_stderr(errors):
                runpy.run_path(str(script))
            self.assertEqual(json.loads(output.getvalue()), {})
            self.assertIn('could not be recorded', errors.getvalue())
