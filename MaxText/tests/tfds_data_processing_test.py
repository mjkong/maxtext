"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=missing-module-docstring, missing-function-docstring
import os
import sys
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils

import unittest
import tensorflow as tf
import tensorflow_datasets as tfds

import pyconfig
from input_pipeline import _tfds_data_processing
from input_pipeline import input_pipeline_interface


class TfdsDataProcessingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        per_device_batch_size=1,
        run_name="test",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        base_output_directory="gs://max-experiments/",
        dataset_path="gs://maxtext-dataset/",
        tokenizer_path="../assets/tokenizer",
        enable_checkpointing=False,
    )
    os.environ["TFDS_DATA_DIR"] = pyconfig.config.dataset_path
    self.config = pyconfig.config
    self.mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)
    self.read_config = tfds.ReadConfig(
        shuffle_seed=self.config.data_shuffle_seed,
    )
    self.read_config.add_tfds_id = True

    self.train_ds, self.eval_ds = self._get_datasets()
    self.train_iter, self.eval_iter, self.predict_iter = self._get_preprocessed_datasets()

  def _get_datasets(self):
    print("Sharding dataset in ", jax.process_count(), " shards")
    process_indices = input_pipeline_interface.get_process_loading_real_data(self.config, self.mesh)
    train_ds, eval_ds = _tfds_data_processing.get_datasets(
        config=self.config,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        read_config=self.read_config,
    )
    return train_ds, eval_ds

  def _get_preprocessed_datasets(self):
    mesh_shape_1d = (len(jax.devices()),)
    mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape_1d), self.config.mesh_axes)
    sp_tokenizer = input_pipeline_interface.get_tokenizer(self.config.tokenizer_path, add_bos = True, add_eos = False)
    train_iter, eval_iter, test_iter = _tfds_data_processing.preprocess_dataset(
        self.config, mesh, self.train_ds, self.eval_ds, sp_tokenizer
    )
    return train_iter, eval_iter, test_iter

  def test_train_ds(self):
    expected_shape = [jax.device_count(), self.config.max_target_length]
    # For training we pack multiple short examples in one example.
    # *_position and *_segmentation indicate the boundaries.
    batch = next(self.train_iter)
    self.assertEqual(
        {k: list(v.shape) for k, v in batch.items()},
        {
            "inputs": expected_shape,
            "inputs_position": expected_shape,
            "inputs_segmentation": expected_shape,
            "targets": expected_shape,
            "targets_position": expected_shape,
            "targets_segmentation": expected_shape,
        },
    )

  def test_eval_ds(self):
    expected_shape = [jax.device_count(), self.config.max_target_length]
    batch = next(self.eval_iter)
    self.assertEqual(
        {k: list(v.shape) for k, v in batch.items()},
        {
            "inputs": expected_shape,
            "targets": expected_shape,
        },
    )

  def test_predict_ds(self):
    expected_shape = [jax.device_count(), self.config.max_target_length]
    batch = next(self.predict_iter)
    self.assertEqual(
        {k: list(v.shape) for k, v in batch.items()},
        {
            "inputs": expected_shape,
            "targets": expected_shape,
        },
    )

  def test_ds_determinism(self):
    train_ds1 = self.train_ds.batch(64)
    train_ds1 = next(train_ds1.as_numpy_iterator())
    # reset the dataset loading
    train_ds, _ = self._get_datasets()
    train_ds = train_ds.batch(64)
    train_ds2 = next(train_ds.as_numpy_iterator())

    self.assertCountEqual(train_ds1["tfds_id"], train_ds2["tfds_id"])

  def test_batch_determinism(self):
    batch1 = next(self.train_iter)
    self.train_ds, _ = self._get_datasets()
    train_iter2, _, _ = self._get_preprocessed_datasets()
    batch2 = next(train_iter2)
    self.assertTrue(tf.reduce_all(tf.equal(batch1["inputs"], batch2["inputs"])))
    self.assertTrue(tf.reduce_all(tf.equal(batch1["targets"], batch2["targets"])))
    self.assertTrue(tf.reduce_all(tf.equal(batch1["inputs_segmentation"], batch2["inputs_segmentation"])))
    self.assertTrue(tf.reduce_all(tf.equal(batch1["targets_segmentation"], batch2["targets_segmentation"])))
    self.assertTrue(tf.reduce_all(tf.equal(batch1["inputs_position"], batch2["inputs_position"])))
    self.assertTrue(tf.reduce_all(tf.equal(batch1["targets_position"], batch2["targets_position"])))

  def test_for_loop_repeatable(self):
    def get_first_batch(iterator):
      batch = None
      for batch in iterator:
        break
      return batch

    eval_batch1 = get_first_batch(self.eval_iter)
    eval_batch2 = get_first_batch(self.eval_iter)
    self.assertTrue((eval_batch1["inputs"] == eval_batch2["inputs"]).all())
    self.assertTrue((eval_batch1["targets"] == eval_batch2["targets"]).all())


if __name__ == "__main__":
  unittest.main()
