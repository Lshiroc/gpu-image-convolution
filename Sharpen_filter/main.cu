#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
#include <algorithm>

#include "sharpen_kernel.cuh"

static void die(const char* m){ std::cerr << "ERROR: " << m << "\n"; std::exit(1); }
static void ck(cudaError_t e, const char* m){
  if(e!=cudaSuccess){ std::cerr << "CUDA ERROR: " << m << ": " << cudaGetErrorString(e) << "\n"; std::exit(1); }
}

static std::string next_token(std::istream& is) {
  std::string tok;
  while (true) {
    if (!(is >> tok)) return "";
    if (!tok.empty() && tok[0] == '#') { std::string rest; std::getline(is, rest); continue; }
    return tok;
  }
}

static void read_ppm_p6(const std::string& path, int& w, int& h, std::vector<uint8_t>& rgb) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("Cannot open input PPM");
  std::string magic = next_token(f);
  if (magic != "P6") die("Input must be P6 PPM");

  std::string sw = next_token(f), sh = next_token(f), sm = next_token(f);
  if (sw.empty() || sh.empty() || sm.empty()) die("PPM header parse failed");

  w = std::stoi(sw);
  h = std::stoi(sh);
  int maxv = std::stoi(sm);
  if (w <= 0 || h <= 0) die("Invalid PPM dimensions");
  if (maxv != 255) die("PPM maxval must be 255");

  char c; f.get(c);
  rgb.resize((size_t)w * (size_t)h * 3);
  f.read(reinterpret_cast<char*>(rgb.data()), (std::streamsize)rgb.size());
  if ((size_t)f.gcount() != rgb.size()) die("PPM pixel data truncated");
}

static void write_ppm_p6(const std::string& path, int w, int h, const std::vector<uint8_t>& rgb) {
  std::ofstream f(path, std::ios::binary);
  if (!f) die("Cannot open output PPM");
  f << "P6\n" << w << " " << h << "\n255\n";
  f.write(reinterpret_cast<const char*>(rgb.data()), (std::streamsize)rgb.size());
  if (!f) die("Failed writing output PPM");
}

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// CPU Laplacian sharpen (same math as GPU)
static void cpu_sharpen_laplacian(const std::vector<uint8_t>& in,
                                  std::vector<uint8_t>& out,
                                  int w, int h,
                                  float amount) {
  out.resize(in.size());
  auto at = [w](int x,int y,int c){ return (y*w+x)*3+c; };

  for (int y = 0; y < h; ++y) {
    int ym1 = clampi(y-1,0,h-1), yp1 = clampi(y+1,0,h-1);
    for (int x = 0; x < w; ++x) {
      int xm1 = clampi(x-1,0,w-1), xp1 = clampi(x+1,0,w-1);
      for (int c = 0; c < 3; ++c) {
        int center = (int)in[at(x,y,c)];
        int left   = (int)in[at(xm1,y,c)];
        int right  = (int)in[at(xp1,y,c)];
        int up     = (int)in[at(x,ym1,c)];
        int down   = (int)in[at(x,yp1,c)];
        int lap = 4*center - left - right - up - down;

        float vf = (float)center + amount * (float)lap;
        int v = (int)(vf + 0.5f);
        out[at(x,y,c)] = (uint8_t)clampi(v,0,255);
      }
    }
  }
}

static void error_metrics(const std::vector<uint8_t>& a,
                          const std::vector<uint8_t>& b,
                          double& mse, int& max_abs) {
  if (a.size() != b.size()) die("Metric size mismatch");
  long long n = (long long)a.size();
  long long sum_sq = 0;
  int m = 0;
  for (long long i = 0; i < n; ++i) {
    int d = (int)a[i] - (int)b[i];
    int ad = d < 0 ? -d : d;
    if (ad > m) m = ad;
    sum_sq += (long long)d * (long long)d;
  }
  mse = (double)sum_sq / (double)n;
  max_abs = m;
}

static double gbps(size_t bytes, float ms) {
  double sec = (double)ms / 1000.0;
  if (sec <= 0.0) return 0.0;
  return ((double)bytes / 1e9) / sec;
}

int main(int argc, char** argv) {
  std::string input;
  int repeats = 20;
  int warmup  = 5;
  float amount = 1.0f;
  int passes = 1;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--input") && i+1 < argc) input = argv[++i];
    else if (!std::strcmp(argv[i], "--repeats") && i+1 < argc) repeats = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--warmup")  && i+1 < argc) warmup  = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--amount")  && i+1 < argc) amount  = (float)std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--passes")  && i+1 < argc) passes  = std::atoi(argv[++i]);
    else die("Usage: ./sharpen --input input.ppm --repeats N --warmup M --amount A --passes P");
  }
  if (input.empty()) die("Missing --input");
  if (repeats < 1) repeats = 1;
  if (warmup < 0) warmup = 0;
  if (passes < 1) passes = 1;
  if (amount < 0.0f) amount = 0.0f;

  int w=0,h=0;
  std::vector<uint8_t> h_in;
  read_ppm_p6(input, w, h, h_in);
  size_t bytes = h_in.size();

  // CPU reference (passes times, timed as one unit)
  std::vector<uint8_t> cpu_a = h_in, cpu_b;
  auto cpu0 = std::chrono::high_resolution_clock::now();
  for (int p = 0; p < passes; ++p) {
    cpu_sharpen_laplacian(cpu_a, cpu_b, w, h, amount);
    cpu_a.swap(cpu_b);
  }
  auto cpu1 = std::chrono::high_resolution_clock::now();
  std::vector<uint8_t> h_cpu = cpu_a;
  double cpu_ms = std::chrono::duration<double, std::milli>(cpu1 - cpu0).count();

  // GPU alloc: ping-pong buffers
  uint8_t *d_a=nullptr, *d_b=nullptr;
  ck(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
  ck(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");

  cudaEvent_t e0, e1;
  ck(cudaEventCreate(&e0), "event create");
  ck(cudaEventCreate(&e1), "event create");

  // H2D (to d_a)
  ck(cudaEventRecord(e0), "event record H2D start");
  ck(cudaMemcpy(d_a, h_in.data(), bytes, cudaMemcpyHostToDevice), "H2D memcpy");
  ck(cudaEventRecord(e1), "event record H2D stop");
  ck(cudaEventSynchronize(e1), "event sync H2D");
  float h2d_ms = 0.0f;
  ck(cudaEventElapsedTime(&h2d_ms, e0, e1), "elapsed H2D");

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    uint8_t* in_ptr = d_a;
    uint8_t* out_ptr = d_b;
    for (int p = 0; p < passes; ++p) {
      sharpen_rgb_u8(in_ptr, out_ptr, w, h, amount);
      std::swap(in_ptr, out_ptr);
    }
    ck(cudaGetLastError(), "kernel launch warmup");
    ck(cudaDeviceSynchronize(), "sync warmup");
  }

  // Kernel timing over repeats (each repeat runs `passes` passes)
  ck(cudaEventRecord(e0), "event record kernel start");
  for (int i = 0; i < repeats; ++i) {
    uint8_t* in_ptr = d_a;
    uint8_t* out_ptr = d_b;
    for (int p = 0; p < passes; ++p) {
      sharpen_rgb_u8(in_ptr, out_ptr, w, h, amount);
      std::swap(in_ptr, out_ptr);
    }
  }
  ck(cudaEventRecord(e1), "event record kernel stop");
  ck(cudaEventSynchronize(e1), "event sync kernel");

  float kernel_ms_total = 0.0f;
  ck(cudaEventElapsedTime(&kernel_ms_total, e0, e1), "elapsed kernel");
  float kernel_ms_avg = kernel_ms_total / (float)repeats;

  // Final output pointer after `passes` passes
  uint8_t* final_ptr = (passes % 2 == 0) ? d_a : d_b;

  // D2H
  std::vector<uint8_t> h_gpu(bytes);
  ck(cudaEventRecord(e0), "event record D2H start");
  ck(cudaMemcpy(h_gpu.data(), final_ptr, bytes, cudaMemcpyDeviceToHost), "D2H memcpy");
  ck(cudaEventRecord(e1), "event record D2H stop");
  ck(cudaEventSynchronize(e1), "event sync D2H");
  float d2h_ms = 0.0f;
  ck(cudaEventElapsedTime(&d2h_ms, e0, e1), "elapsed D2H");

  // Correctness CPU vs GPU
  double mse = 0.0; int max_abs = 0;
  error_metrics(h_cpu, h_gpu, mse, max_abs);

  // Write ONLY GPU output (temporary PPM for conversion to PNG)
  write_ppm_p6("gpu_out.ppm", w, h, h_gpu);

  // Performance comparison
  double gpu_e2e_ms = (double)h2d_ms + (double)kernel_ms_avg + (double)d2h_ms;
  double speedup_kernel = cpu_ms / (double)kernel_ms_avg;
  double speedup_e2e    = cpu_ms / gpu_e2e_ms;

  std::cout << "=== Sharpen filter executed ===\n";
  std::cout << "CPU: Laplacian sharpen (amount=" << amount << ", passes=" << passes << ") (timed once)\n";
  std::cout << "GPU: Laplacian sharpen (amount=" << amount << ", passes=" << passes << ") "
            << "(avg over " << repeats << " repeats, warmup " << warmup << ")\n\n";

  std::cout << "=== Image / data ===\n";
  std::cout << "W=" << w << " H=" << h << " Channels=3 Bytes=" << bytes << "\n\n";

  std::cout << "=== Timing (ms) ===\n";
  std::cout << "CPU (passes=" << passes << "): " << cpu_ms << "\n";
  std::cout << "H2D memcpy:   " << h2d_ms << "  (" << gbps(bytes, h2d_ms) << " GB/s)\n";
  std::cout << "Kernel avg:   " << kernel_ms_avg << "  (total " << kernel_ms_total << " over " << repeats << ")\n";
  std::cout << "D2H memcpy:   " << d2h_ms << "  (" << gbps(bytes, d2h_ms) << " GB/s)\n\n";

  std::cout << "=== CPU vs GPU correctness ===\n";
  std::cout << "Max abs error: " << max_abs << "\n";
  std::cout << "MSE:           " << mse << "\n\n";

  std::cout << "=== CPU vs GPU performance ===\n";
  std::cout << "GPU end-to-end per pass: " << gpu_e2e_ms
            << "  [H2D " << h2d_ms << " + kernel " << kernel_ms_avg << " + D2H " << d2h_ms << "]\n";
  std::cout << "Speedup (CPU / GPU kernel):   " << speedup_kernel << "x\n";
  std::cout << "Speedup (CPU / GPU end2end):  " << speedup_e2e << "x\n";

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);

  return 0;
}
