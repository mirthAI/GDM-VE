export OMP_NUM_THREADS=4

torchrun --standalone --nproc_per_node=1 GDM_main.py "$@"