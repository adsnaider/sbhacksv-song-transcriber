gcloud ml-engine jobs submit training $1    --runtime-version 1.12 \
--config config.yaml   --module-name trainer.gan           --staging-bucket gs://song-repo \
--package-path trainer/  -- --job_dir "gs://song-repo/training/$1/" \
--genre1 gs://song-repo/fma_small/tfrecords/15.tfrecords \
--genre2 gs://song-repo/fma_small/tfrecords/17.tfrecords
