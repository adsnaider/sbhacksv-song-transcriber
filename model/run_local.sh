gcloud ml-engine local train --module-name trainer.gan --package-path trainer -- \
--genre1 gs://song-repo/fma_small/tfrecords/10.tfrecords --genre2 \
gs://song-repo/fma_small/tfrecords/2.tfrecords --job_dir output
