FILE_NAME="bplus_hotset_benchmark_1024.json"

python3 daoram/omap/bplus_omap_hot_benchmark.py \
  --workload hotset \
  --num-data 1024 \
  --insert-count 512 \
  --payload-size 64 \
  --seed 42 \
  --warmup-requests 400 \
  --num-queries 4000 \
  --cache-size 16 \
  --hot-threshold 1 \
  --rolling-window 64 \
  --hotset-size 12 \
  --hot-query-probability 0.9 \
  --output-json ${FILE_NAME}


python3 daoram/omap/bplus_omap_hot_benchmark_plot.py \
  ${FILE_NAME} \
  --output-dir bplus_hotset_warmup_demo_plots \
  --plot-bucket-size 32