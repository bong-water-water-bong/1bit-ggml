// TQ2_0 × Q8_1 mul_mat_vec dispatch — bridges ggml-hip into the halo
// ternary GEMV kernel that lives in the rocm-cpp out-of-tree library.
//
// The kernel itself is at:
//   rocm-cpp/kernels/ternary_gemv_tq2_0.hip  (commit 0dcb709)
//
// Design doc: docs/wiki/TQ2_0-HIP-Port-Plan.md in the halo workspace.
//
// NOTE (build-side): this TU references `ternary_gemv_tq2_0_q8_1_launch`
// as an unresolved `extern "C"` symbol. Downstream projects that bundle
// this ggml submodule (1bit-tts.cpp, stable-diffusion.cpp) are
// responsible for adding `rocm_cpp` (or equivalent) to the link list of
// the final executable. The ggml-cuda static lib compiles cleanly with
// the symbol undefined; it surfaces only at final link time.
//
// Rationale: the halo kernel is the canonical home for ternary HIP
// (Rule B in halo-ai-core/CLAUDE.md). Copy-pasting it into the ggml
// submodule would duplicate ~250 LOC of hand-tuned assembly-adjacent
// code and diverge under maintenance.

#include "ggml.h"
#include "mmvq.cuh"
#include "common.cuh"
#include "quantize.cuh"

#include <cstdint>

#if defined(GGML_USE_HIP)

// The launcher lives in librocm_cpp (out-of-tree). Signature must match
// rocm-cpp/kernels/ternary_gemv_tq2_0.hip verbatim.
extern "C" void ternary_gemv_tq2_0_q8_1_launch(
    const void * vx,      // block_tq2_0 * — [nrows, ncols/QK_K] weight blocks
    const void * vy,      // block_q8_1  * — [ncols/QK8_1] activation blocks
    void       * dst,     // float       * — [nrows]
    int32_t      ncols,   // K, must be a multiple of 256 (QK_K)
    int32_t      nrows,   // M
    void       * stream); // hipStream_t

#endif // GGML_USE_HIP

// Dispatch entry for GGML_OP_MUL_MAT with src0=TQ2_0 and ne11==1.
// Performs its own Q8_1 quantisation of src1, then calls the halo
// launcher. Callers must have already verified:
//   - src0->type == GGML_TYPE_TQ2_0
//   - src1->type == GGML_TYPE_F32
//   - src1->ne[1] == 1 (single-token decode)
//   - src0->ne[0] % QK_K == 0
// Violations are GGML_ASSERT'd here as a last line of defence.
void ggml_cuda_op_mul_mat_vec_tq2_0_q8_1(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         src0,
        const ggml_tensor *         src1,
        ggml_tensor *               dst) {
#if defined(GGML_USE_HIP)
    GGML_ASSERT(src0->type == GGML_TYPE_TQ2_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];   // K
    const int64_t ne01 = src0->ne[1];   // M (nrows)
    const int64_t ne10 = src1->ne[0];   // K (must match ne00)
    const int64_t ne11 = src1->ne[1];   // 1 (decode)
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT(ne11 == 1 && "TQ2_0 HIP GEMV is decode-only (ne11==1)");
    GGML_ASSERT(ne12 == 1 && ne13 == 1 && "TQ2_0 HIP GEMV: batched src1 not supported");
    GGML_ASSERT(ne00 % 256 == 0 && "TQ2_0 requires K divisible by QK_K (256)");
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1 && "TQ2_0 HIP GEMV: batched src0 not supported");
    GGML_ASSERT(dst->ne[0] == ne01);

    cudaStream_t stream = ctx.stream();

    // Quantise src1 (fp32) into a pool-allocated Q8_1 buffer. Mirrors
    // the logic at the top of ggml_cuda_mul_mat_vec_q, but trimmed to
    // the ne11==ne12==ne13==1 case.
    const size_t  ts_src1     = ggml_type_size(src1->type);
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(),
        ne10_padded * sizeof(block_q8_1) / QK8_1);

    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s13 = src1->nb[3] / ts_src1;
    quantize_row_q8_1_cuda(
        (const float *) src1->data, /*ids=*/nullptr, src1_q8_1.get(),
        src0->type, ne10, s11, s12, s13, ne10_padded,
        /*ne1=*/1, /*ne2=*/1, /*ne3=*/1, stream);

    ternary_gemv_tq2_0_q8_1_launch(
        src0->data,
        src1_q8_1.get(),
        dst->data,
        static_cast<int32_t>(ne00),
        static_cast<int32_t>(ne01),
        (void *) stream);
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_ABORT("TQ2_0 HIP path is only available in GGML_USE_HIP builds");
#endif
}
