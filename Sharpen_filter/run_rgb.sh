#!/bin/bash
set -euo pipefail

# ---- settings ----
ACCOUNT="project_2016196"
PARTITION="gpusmall"
GPU_GRES="gpu:a100_1g.5gb:1"
CPUS=4
MEM="17500"
TIME="00:10:00"

EXE="sharpen"
INPUT_PPM="input.ppm"
OUTTXT="output.txt"
OUTPNG="sharpened.png"

# Sharpen strength controls (NEW)
AMOUNT="${AMOUNT:-1.5}"   # try 1.5, 2.0, 3.0
PASSES="${PASSES:-1}"     # try 1 or 2 (more = stronger)
REPEATS="${REPEATS:-20}"
WARMUP="${WARMUP:-5}"

# ---- checks: source files ----
for f in main.cu sharpen_kernel.cu sharpen_kernel.cuh; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Missing $f in $(pwd)"
    exit 1
  fi
done

# ---- choose input image ----
INPUT_IMG="${1:-}"

if [[ -z "$INPUT_IMG" ]]; then
  echo "No input image provided."
  echo "Images found in this directory:"
  mapfile -t imgs < <(ls -1 *.jpg *.jpeg *.png *.bmp *.tif *.tiff 2>/dev/null || true)

  if [[ ${#imgs[@]} -gt 0 ]]; then
    for i in "${!imgs[@]}"; do
      printf "  [%d] %s\n" "$((i+1))" "${imgs[$i]}"
    done
    echo
    read -r -p "Type a number (e.g., 1) OR type a full path to an image: " choice

    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#imgs[@]} )); then
      INPUT_IMG="${imgs[$((choice-1))]}"
    else
      INPUT_IMG="$choice"
    fi
  else
    read -r -p "No images in folder. Type full path to an image: " INPUT_IMG
  fi
fi

if [[ ! -f "$INPUT_IMG" ]]; then
  echo "ERROR: Input image not found: $INPUT_IMG"
  exit 1
fi

echo "Selected input image: $INPUT_IMG"
echo "Sharpen settings: amount=$AMOUNT passes=$PASSES repeats=$REPEATS warmup=$WARMUP"

# ---- compile on login node ----
module purge
module load gcc/11.2.0
module load cuda/11.5.0

echo "[1/3] Compiling main.cu + sharpen_kernel.cu -> $EXE"
nvcc -O3 -lineinfo -Xcompiler="-O3 -fopenmp" \
  main.cu sharpen_kernel.cu -o "$EXE"

# ---- convert input (image -> PPM for compute) ----
echo "[2/3] Converting $INPUT_IMG -> $INPUT_PPM"
module load imagemagick

# Note: using \> prevents upscaling smaller images; remove \> if you want forced 4096x4096.
convert "$INPUT_IMG" -resize 4096x4096\> -depth 8 "$INPUT_PPM"

# ---- run on GPU node via Slurm (save FULL report to output.txt) ----
echo "[3/3] Running on GPU via srun"
srun --account="$ACCOUNT" \
     --partition="$PARTITION" \
     --cpus-per-task="$CPUS" \
     --mem="$MEM" \
     --time="$TIME" \
     --gres="$GPU_GRES" \
     bash -lc "
        set -e
        cd \"$PWD\"
        module purge
        module load gcc/11.2.0
        module load cuda/11.5.0
        echo \"Node: \$(hostname)\"
        nvidia-smi -L || true

        # Save the program's full printed report to output.txt
        ./$EXE --input $INPUT_PPM --repeats $REPEATS --warmup $WARMUP --amount $AMOUNT --passes $PASSES | tee $OUTTXT
     "

# ---- convert ONLY final output to PNG ----
module load imagemagick
echo "Creating output image: $OUTPNG"
convert gpu_out.ppm "$OUTPNG"

# ---- cleanup ----
rm -f "$INPUT_PPM" gpu_out.ppm

echo "DONE."
echo "  Input image:  $INPUT_IMG"
echo "  Output image: $OUTPNG"
echo "  Report saved: $OUTTXT"
