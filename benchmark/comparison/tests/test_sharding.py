"""Disjoint release-order partitions must reconstruct their complete cohort."""
import unittest

from comparison.run import (experiment_mcqa_cohort, experiment_mcqa_dataset,
                            experiment_tracks, select_tasks)


class ShardingTests(unittest.TestCase):
    def tasks(self, size=19):
        return [{"task_id": f"case-{index}"} for index in range(size)]

    def test_shards_are_disjoint_and_complete(self):
        tasks = self.tasks()
        observed = []
        for shard in range(8):
            selected, metadata = select_tasks(tasks, selection={"shard_count": 8, "shard_index": shard})
            self.assertEqual(metadata["cohort_task_ids"], [task["task_id"] for task in tasks])
            self.assertEqual(metadata["expected_shard_task_ids"], [task["task_id"] for task in selected])
            self.assertFalse(metadata["diagnostic_subset"])
            self.assertTrue(metadata["is_shard"])
            observed.extend(task["task_id"] for task in selected)
        self.assertEqual(len(observed), len(set(observed)))
        self.assertEqual(set(observed), {task["task_id"] for task in tasks})

    def test_explicit_ids_preserve_release_order_before_sharding(self):
        selected, metadata = select_tasks(self.tasks(), selection={"task_ids": ["case-8", "case-2", "case-4"],
                                                                  "shard_count": 2, "shard_index": 0})
        self.assertEqual([task["task_id"] for task in selected], ["case-2", "case-8"])
        self.assertEqual(metadata["eligible_task_ids"], ["case-2", "case-4", "case-8"])
        self.assertTrue(metadata["diagnostic_subset"])

    def test_limit_does_not_reduce_expected_shard_denominator(self):
        selected, metadata = select_tasks(self.tasks(), 1, {"shard_count": 2, "shard_index": 1})
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(metadata["expected_shard_task_ids"]), 9)
        self.assertEqual(len(metadata["cohort_task_ids"]), 19)
        self.assertTrue(metadata["diagnostic_subset"])

    def test_invalid_shards_and_duplicate_ids_rejected(self):
        for selection in ({"shard_count": 0}, {"shard_index": 2, "shard_count": 2},
                          {"shard_count": True}, {"task_ids": ["case-1", "case-1"]}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                select_tasks(self.tasks(), selection=selection)


class ExperimentScopeTests(unittest.TestCase):
    def test_explicit_tracks_intersect_requested_tracks(self):
        self.assertEqual(experiment_tracks({"tracks": ["mcqa"]}, ("mcqa", "sequential")), ("mcqa",))
        self.assertEqual(experiment_tracks({"tracks": ["mcqa"]}, ("sequential",)), ())

    def test_t1_reference_defaults_to_mcqa_only(self):
        self.assertEqual(experiment_tracks({"reporting_scope": "T1_reference"}), ("mcqa",))
        with self.assertRaisesRegex(ValueError, "MCQA-only"):
            experiment_tracks({"reporting_scope": "T1_reference", "tracks": ["sequential"]})

    def test_reference_cohort_and_answer_key_have_matching_t1_ids(self):
        tasks = [{"task_id": "one", "tier": "T1", "split": "train"},
                 {"task_id": "two", "tier": "T2", "split": "test"},
                 {"task_id": "three", "tier": "T1", "split": "test"}]
        answers = {task["task_id"]: {"correct_option_ids": ["A"]} for task in tasks}
        experiment = {"reporting_scope": "T1_reference"}
        cohort = experiment_mcqa_cohort(tasks, "all", experiment)
        self.assertEqual([task["task_id"] for task in cohort], ["one", "three"])
        self.assertEqual([task["task_id"] for task in experiment_mcqa_cohort(tasks, "heldout", experiment)], ["three"])
        grading_tasks, grading_answers = experiment_mcqa_dataset(tasks, answers, experiment)
        self.assertEqual({task["task_id"] for task in grading_tasks}, set(grading_answers))
        self.assertEqual(set(grading_answers), {"one", "three"})
        self.assertEqual(experiment_mcqa_dataset(tasks, answers, {}), (tasks, answers))

    def test_invalid_tracks_and_reporting_scope_rejected(self):
        for experiment in ({"tracks": []}, {"tracks": ["mcqa", "mcqa"]},
                           {"tracks": [{}]}, {"reporting_scope": "unlabelled_reference"}):
            with self.subTest(experiment=experiment), self.assertRaises(ValueError):
                experiment_tracks(experiment)


if __name__ == "__main__":
    unittest.main()
